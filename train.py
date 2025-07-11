# train.py  ────────────────────────────────────────────────────────────────
"""
Improved training workflow:
    • Training: joint supervision of main views and auxiliary views
    • Validation: main views only
    • Save every 10 epochs: train img + val img
"""
import os, gc, torch, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
import lpips
from piq import ssim
from torchmetrics import PeakSignalNoiseRatio as PSNR

from vggt.models.vggt import VGGT
from vggt.heads.gaussian_head import Gaussianhead
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.renderer_gsplat import render_gaussians
from dataloader import build_train_val

# ─────────────── Hyperparameters & Paths ─────────────────────────────────────────────
ROOT_DIR            = "/usr/prakt/s0012/scannetpp/data"
TRAIN_SPLIT_TXT     = "/usr/prakt/s0012/scannetpp/splits/nvs_sem_train.txt"
VAL_SPLIT_TXT       = "/usr/prakt/s0012/scannetpp/splits/nvs_sem_val.txt"

TRAIN_BATCH_SCENES  = 25
VAL_BATCH_SCENES    = 2
IMG_NUM_TRAIN       = 6         # main views per scene
IMG_NUM_AUX         = 2           # auxiliary views per scene
AUX_WEIGHT          = 0.5
IMG_NUM_VAL         = 2           # validation views per scene
STRIDE              = 5
NUM_WORKERS         = 4
NUM_EPOCHS          = 3001

LOG_DIR     = "runs/scannetpp25_12_4metric"
IMG_LOG_DIR = "renders/scannetpp25_12_4metric"
CKPT_DIR    = "checkpoints/scannetpp25_12_4metric"
CKPT_INTERVAL = 100                 # epoch interval for saving checkpoints

# Resume from checkpoint
RESUME_CKPT = "/usr/prakt/s0012/ADL4VC/checkpoints/scannetpp25_12_4metric/epoch_1000.pth"  # e.g. "checkpoints/scannetpp6_2_2metric/best.pth"

os.makedirs(IMG_LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,  exist_ok=True)

# ─────────────── Loss Weights ─────────────────────────────────────────────────────
L_MSE, L_LP, L_SS = 1.0, 0.2, 0.1  # SSIM loss is calculated as (1 - SSIM)

# ─────────────── Device & Precision ───────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = (torch.bfloat16 if device == "cuda"
          and torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
print(f"[Init] device={device}, dtype={dtype}")

# ─────────────── Utility Functions ─────────────────────────────────────────────────
def flatten_gdict(gdict: dict, B: int):
    out = {}
    for k, v in gdict.items():
        C = v.shape[-1] if v.ndim >= 2 else 1
        out[k] = v.reshape(B, -1, C)
    return out

@torch.no_grad()
def build_cache(loader, vggt_model):
    cache = []
    with torch.amp.autocast("cuda", dtype=dtype):
        for imgs, scene_id, img_names in loader:
            imgs = imgs.to(device)
            N, _, H, W = imgs.shape
            tok_list, ps_idx = vggt_model.aggregator(imgs.unsqueeze(1))
            preds            = vggt_model(imgs)
            point_map        = preds["world_points"].unsqueeze(1)
            extr, intr       = pose_encoding_to_extri_intri(preds["pose_enc"], (H, W))

            cache.append(dict(
                imgs      = imgs.cpu(),
                tok_list  = [t.cpu() for t in tok_list],
                ps_idx    = ps_idx,
                point_map = point_map.cpu(),
                intr      = intr.cpu(),
                extr      = extr.cpu(),
            ))
    return cache

# @torch.no_grad()
# def build_cache_with_aux(loader, vggt_model):
#     cache = []
#     with torch.amp.autocast("cuda", dtype=dtype):
#         for imgs_main, imgs_aux, scene_id, names_main, names_aux in loader:
#             imgs_all = torch.cat([imgs_main, imgs_aux], 0).to(device)
#             N_main, _, H, W = imgs_main.shape

#             tok_all, ps_idx = vggt_model.aggregator(imgs_all.unsqueeze(1))
#             preds_all       = vggt_model(imgs_all)
#             point_map_all   = preds_all["world_points"].unsqueeze(1)
#             extr_all, intr_all = pose_encoding_to_extri_intri(preds_all["pose_enc"], (H, W))

#             cache.append(dict(
#                 imgs_main  = imgs_main.cpu(),
#                 imgs_aux   = imgs_aux.cpu(),
#                 tok_list   = [t[:N_main].cpu() for t in tok_all],
#                 ps_idx     = ps_idx,
#                 point_map  = point_map_all[:, :, :N_main].cpu(),
#                 intr_main  = intr_all[:, :N_main].cpu(),
#                 extr_main  = extr_all[:, :N_main].cpu(),
#                 intr_aux   = intr_all[:, N_main:].cpu(),
#                 extr_aux   = extr_all[:, N_main:].cpu(),
#             ))
#     return cache

@torch.no_grad()
def build_cache_with_aux(loader, vggt_model):
    """同时对主/辅助视图进行一次 VGGT 前向，并缓存所有必要数据至 CPU"""
    cache = []
    with torch.amp.autocast("cuda", dtype=dtype):
        for imgs_main, imgs_aux, scene_id, names_main, names_aux in loader:
            imgs_all = torch.cat([imgs_main, imgs_aux], dim=0).to(device)
            # print(f"[Select] scene {scene_id}: Main: {', '.join(names_main)}; Auxiliary: {', '.join(names_aux)}")
            N_main, _, H, W = imgs_main.shape
            tok_list_all, ps_idx_all = vggt_model.aggregator(imgs_all.unsqueeze(1))
            preds_all                = vggt_model(imgs_all)
            point_map_all            = preds_all["world_points"].unsqueeze(1)
            pose_enc_all             = preds_all["pose_enc"]

            # 拆分主/辅助编码 & token
            # pe_main, pe_aux    = pose_enc_all[:N_main], pose_enc_all[N_main:]
            tok_main           = [t[:IMG_NUM_TRAIN] for t in tok_list_all]
            pm_main            = point_map_all[:, :, :N_main, :, :]

            # 解码至同一世界坐标系
            # extr_main, intr_main = pose_encoding_to_extri_intri(pe_main, (H, W))
            # extr_aux,  intr_aux  = pose_encoding_to_extri_intri(pe_aux,  (H, W))
            extr, intr = pose_encoding_to_extri_intri(pose_enc_all, (H, W))
            extr_main, intr_main = extr[:, :N_main, :, :], intr[:, :N_main, :, :]
            extr_aux,  intr_aux  = extr[:, N_main:, :, :], intr[:, N_main:, :, :]

            cache.append({
                'scene_id'   : scene_id,
                'names_main' : names_main,
                'names_aux'  : names_aux,
                'imgs_main'  : imgs_main.cpu(),
                'imgs_aux'   : imgs_aux.cpu(),
                'tok_list'   : [t.cpu() for t in tok_main],
                'ps_idx'     : ps_idx_all,
                'point_map'  : pm_main.cpu(),
                'intr_main'  : intr_main.cpu(),
                'extr_main'  : extr_main.cpu(),
                'intr_aux'   : intr_aux.cpu(),
                'extr_aux'   : extr_aux.cpu(),
            })
    return cache

# ─────────────── Model & Loss Setup ────────────────────────────────────────────────
print("[Init] build Gaussian Head …")
_emb = VGGT.from_pretrained("facebook/VGGT-1B")
EMB_DIM = _emb.embed_dim
del _emb

g_head = Gaussianhead(
    2*EMB_DIM,
    3+3+4+3*(0+1)**2+1,
    activation="exp",
    conf_activation="expp1",
    sh_degree=0
).to(device)

opt = torch.optim.Adam(g_head.parameters(), lr=1e-4)
mse_fn   = nn.MSELoss()
lpips_fn = lpips.LPIPS(net="alex").to(device)
psnr_fn  = PSNR(data_range=1.0).to(device)

# ─────────────── Logger & Checkpoint Resume ────────────────────────────────────────
writer         = SummaryWriter(LOG_DIR)
best_val_loss  = float("inf")
global_step    = 0
start_epoch    = 1

if RESUME_CKPT and os.path.isfile(RESUME_CKPT):
    ckpt = torch.load(RESUME_CKPT, map_location=device)
    g_head.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        opt.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch   = ckpt.get("epoch", 0) + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    global_step   = ckpt.get("global_step", 0)
    print(f"[Resume] Loaded checkpoint '{RESUME_CKPT}' (epoch {start_epoch-1})")


vggt = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)
for p in vggt.parameters(): p.requires_grad_(False)
    
# ======================================================================
#                               Main Loop
# ======================================================================
for epoch in range(start_epoch, NUM_EPOCHS + 1):
    print(f"\n===== Epoch {epoch} =====")

    # 1. Initialize DataLoaders for training and validation
    train_loader, val_loader = build_train_val(
        root_dir           = ROOT_DIR,
        train_split_txt    = TRAIN_SPLIT_TXT,
        val_split_txt      = VAL_SPLIT_TXT,
        train_batch_scenes = TRAIN_BATCH_SCENES,
        train_img_num      = IMG_NUM_TRAIN,
        train_img_aux      = IMG_NUM_AUX,
        val_batch_scenes   = VAL_BATCH_SCENES,
        val_img_num        = IMG_NUM_VAL,
        stride             = STRIDE,
        num_workers        = NUM_WORKERS,
    )

    # 2. Build forward cache using frozen VGGT backbone


    train_cache = build_cache_with_aux(train_loader, vggt)
    val_cache   = build_cache(val_loader, vggt)

    # del vggt

    # ─────────────── TRAINING ───────────────────────────────────────────────
    g_head.train()
    viz_gt_tr, viz_rd_tr = [], []
    loss_train_epoch = 0.0

    for entry in tqdm(train_cache, desc="Train"):
        imgs_main = entry["imgs_main"].to(device)
        N_main, _, H, W = imgs_main.shape
        tok   = [t.to(device) for t in entry["tok_list"]]
        intr_main = entry["intr_main"].to(device).float()
        extr_main = torch.cat(
            [entry["extr_main"].to(device).float(),
             torch.tensor([0,0,0,1], device=device).view(1,1,1,4).expand(1,N_main,1,4)],
            dim=-2
        )

        gdict_raw = g_head(tok, imgs_main.unsqueeze(1),
                           entry["ps_idx"], entry["point_map"].to(device))
        gdict     = {k: (v.view(1,-1,v.shape[-1]) if v.ndim==3 else v.view(1,-1))
                     for k,v in flatten_gdict(gdict_raw, N_main).items()}

        renders_main = render_gaussians(gdict, intr_main, extr_main, H, W)

        # Auxiliary views rendering
        imgs_aux = entry["imgs_aux"].to(device)
        N_aux    = imgs_aux.shape[0]
        intr_aux = entry["intr_aux"].to(device).float()
        extr_aux = torch.cat(
            [entry["extr_aux"].to(device).float(),
             torch.tensor([0,0,0,1], device=device).view(1,1,1,4).expand(1,N_aux,1,4)],
            dim=-2
        )
        renders_aux = render_gaussians(gdict, intr_aux, extr_aux, H, W)

        # Compute losses
        loss_mse_main = mse_fn(renders_main, imgs_main)
        loss_mse_aux  = mse_fn(renders_aux,  imgs_aux)
        loss_lp_main  = lpips_fn(renders_main, imgs_main).mean()
        loss_lp_aux   = lpips_fn(renders_aux,  imgs_aux).mean()
        loss_ss_main  = 1.0 - ssim(renders_main, imgs_main, data_range=1.0).mean()
        loss_ss_aux   = 1.0 - ssim(renders_aux,  imgs_aux,  data_range=1.0).mean()

        loss_total = (
            L_MSE*(loss_mse_main + AUX_WEIGHT*loss_mse_aux) +
            L_LP *(loss_lp_main  + AUX_WEIGHT*loss_lp_aux)  +
            L_SS *(loss_ss_main  + AUX_WEIGHT*loss_ss_aux)
        )

        opt.zero_grad(set_to_none=True)
        loss_total.backward()
        opt.step()

        # Logging & visualization
        loss_train_epoch += loss_total.item()
        writer.add_scalar("loss/train_batch", loss_total.item(), global_step)
        global_step += 1

        viz_gt_tr.append(torch.cat([imgs_main.cpu(),  imgs_aux.cpu()], dim=0))
        viz_rd_tr.append(torch.cat([renders_main.cpu(), renders_aux.cpu()], dim=0))

    loss_train_epoch /= len(train_cache)
    writer.add_scalar("loss/train_epoch", loss_train_epoch, epoch)
    print(f"train loss_total = {loss_train_epoch:.5f}")

    # ─────────────── VALIDATION ─────────────────────────────────────────────
    g_head.eval()
    viz_gt_val, viz_rd_val = [], []
    loss_val_epoch = psnr_val_epoch = ssim_val_epoch = lpips_val_epoch = 0.0

    with torch.no_grad():
        for entry in tqdm(val_cache, desc="Val  "): 
            imgs = entry["imgs"].to(device)
            N, _, H, W = imgs.shape
            tok = [t.to(device) for t in entry["tok_list"]]
            intr = entry["intr"].to(device).float()
            extr = torch.cat(
                [entry["extr"].to(device).float(),
                 torch.tensor([0,0,0,1], device=device).view(1,1,1,4).expand(1,N,1,4)],
                dim=-2
            )

            gdict_raw = g_head(tok, imgs.unsqueeze(1),
                               entry["ps_idx"], entry["point_map"].to(device))
            gdict = {k: (v.view(1,-1,v.shape[-1]) if v.ndim==3 else v.view(1,-1))
                     for k,v in flatten_gdict(gdict_raw, N).items()}

            renders = render_gaussians(gdict, intr, extr, H, W)

            loss_val_epoch  += mse_fn(renders, imgs).item()
            psnr_val_epoch  += psnr_fn(renders, imgs).item()
            ssim_val_epoch  += ssim(renders, imgs, data_range=1.0).mean().item()
            lpips_val_epoch += lpips_fn(renders, imgs).mean().item()

            viz_gt_val.append(imgs.cpu())
            viz_rd_val.append(renders.cpu())

    num_val = len(val_cache)
    loss_val_epoch  /= num_val
    psnr_val_epoch  /= num_val
    ssim_val_epoch  /= num_val
    lpips_val_epoch /= num_val

    writer.add_scalar("loss/val_epoch",    loss_val_epoch,  epoch)
    writer.add_scalar("metric/psnr_epoch", psnr_val_epoch,  epoch)
    writer.add_scalar("metric/ssim_epoch", ssim_val_epoch,  epoch)
    writer.add_scalar("metric/lpips_epoch", lpips_val_epoch, epoch)

    print(f"val loss={loss_val_epoch:.5f} | "
          f"PSNR={psnr_val_epoch:.2f} dB | "
          f"SSIM={ssim_val_epoch:.4f} | "
          f"LPIPS={lpips_val_epoch:.4f}")

    # ─────────────── Visualization Saving ─────────────────────────────────────
    if epoch % 10 == 0:
        grid_tr = torch.cat([torch.cat(viz_gt_tr, 0),
                             torch.cat(viz_rd_tr, 0)], 0)
        save_image(
            grid_tr,
            os.path.join(IMG_LOG_DIR, f"epoch_{epoch:04d}_tr.png"),
            nrow=IMG_NUM_TRAIN+IMG_NUM_AUX,
            normalize=True, value_range=(0,1)
        )

        grid_val = torch.cat([torch.cat(viz_gt_val, 0),
                              torch.cat(viz_rd_val, 0)], 0)
        save_image(
            grid_val,
            os.path.join(IMG_LOG_DIR, f"epoch_{epoch:04d}_val.png"),
            nrow=IMG_NUM_VAL,
            normalize=True, value_range=(0,1)
        )

    # ─────────────── Checkpoint Saving ─────────────────────────────────────────
    if epoch % CKPT_INTERVAL == 0:
        ckpt_path = os.path.join(CKPT_DIR, f"epoch_{epoch:04d}.pth")
        torch.save({
            "epoch"              : epoch,
            "model_state_dict"   : g_head.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "best_val_loss"      : best_val_loss,
            "global_step"        : global_step,
            "loss_train"         : loss_train_epoch,
            "loss_val"           : loss_val_epoch,
        }, ckpt_path)

    if loss_val_epoch < best_val_loss:
        best_val_loss = loss_val_epoch
        torch.save(g_head.state_dict(), os.path.join(CKPT_DIR, "best.pth"))
        print(f"[Checkpoint] Best model updated (val loss {best_val_loss:.5f})")

    # ─────────────── Cleanup ─────────────────────────────────────────────────
    del train_cache, val_cache, viz_gt_tr, viz_rd_tr, viz_gt_val, viz_rd_val
    gc.collect(); torch.cuda.empty_cache()

writer.close()
print("Training complete ✅")