# train.py  ──────────────────────────────────────────────────────────────────
"""
每个 epoch：
    1. 重新构建 train/val DataLoader（随机抽 scene）
    2. VGGT 前向提特征 → 缓存到 CPU
    3. 释放 VGGT 显存
    4. 用 Gaussian Head 训练，再做验证
    5. 保存 (GT|Render) 拼图，数量 = TRAIN_BATCH_SCENES×IMG_NUM_TRAIN
"""

import time
import os, gc, torch, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
import lpips
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

from vggt.models.vggt import VGGT
from vggt.heads.gaussian_head import Gaussianhead
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.renderer_gsplat import render_gaussians
from dataloader import build_train_val          # ← 你的 dataloader_factory.py

run_id = time.strftime("%Y%m%d-%H%M%S")
out_dir = "/home/stud/syua/storage/user/vggt-guassian"
render_dir = os.path.join(out_dir, "renders", run_id)
writer = SummaryWriter(os.path.join(out_dir, f"runs/gauss_train_{run_id}"))
os.makedirs(render_dir, exist_ok=True)

# ----------------- 训练超参 -----------------
ROOT_DIR            = "/usr/prakt/s0012/scannetpp/data"
TRAIN_SPLIT_TXT     = "/usr/prakt/s0012/scannetpp/splits/nvs_sem_train.txt"
VAL_SPLIT_TXT       = "/usr/prakt/s0012/scannetpp/splits/nvs_sem_val.txt"
TRAIN_BATCH_SCENES  = 4           # 每 epoch 随机抽 4 个 scene 训练
VAL_BATCH_SCENES    = 1
IMG_NUM_TRAIN       = 8           # 训练过程中，每 scene 取 几 张图进行训练
IMG_NUM_AUX         = 2           # 训练过程中，每 scene 额外取 几 张图进行额外渲染。 因此一个训练scene中共取 IMG_NUM_TRAIN+IMG_NUM_AUX 张图
AUX_WEIGHT          = 0.5
IMG_NUM_VAL         = 4
STRIDE              = 3
NUM_WORKERS         = 4
NUM_EPOCHS          = 30000
LOG_DIR             = os.path.join(out_dir, f"runs/gauss_train_{run_id}")
IMG_LOG_DIR         = os.path.join(out_dir, "renders", run_id)
IMG_LOG_INTERVAL    = 10
MODEL_PATH          = f"/home/stud/syua/storage/user/vggt-guassian/checkpoints/20250708-165027/gauss_head_ckpt_epoch0050.pth"        # 要加载的预训练模型的文件地址，精确到.pth文件，无预训练模型请输入 None
CKPT_DIR            = os.path.join(out_dir, "checkpoints", run_id)  # ckpt 的保存文件夹
CKPT_INTERVAL       = 25          # 每多少个 epoch 保存一次 ckpt

LR_RATE             = 1e-4

os.makedirs(IMG_LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
print(f"[Init] device={device}, dtype={dtype}")

# ----------------- util -----------------
def flatten_gdict(gdict: dict, B: int):
    out = {}
    for k, v in gdict.items():
        C = v.shape[-1] if v.ndim >= 2 else 1
        out[k] = v.reshape(B, -1, C)
    return out

@torch.no_grad()
def build_cache(loader, vggt_model):
    """把 loader 中所有 batch 通过 VGGT 前向，全部转存到 CPU。"""
    cache = []
    with torch.amp.autocast("cuda", dtype=dtype):
        for imgs, scene_id, img_names in loader:
            # print(f"[Select] scene {scene_id}: {', '.join(img_names)}")
            imgs  = imgs.to(device)                       # (N,3,H,W)
            N, _, H, W = imgs.shape
            imgs_in = imgs.unsqueeze(1)

            tok_list, ps_idx = vggt_model.aggregator(imgs_in)
            preds            = vggt_model(imgs)
            point_map        = preds["world_points"].unsqueeze(1)
            pose_enc         = preds["pose_enc"]
            extr, intr       = pose_encoding_to_extri_intri(pose_enc, (H, W))

            cache.append(dict(
                imgs      = imgs.cpu(),
                tok_list  = [t.cpu() for t in tok_list],
                ps_idx    = ps_idx,
                point_map = point_map.cpu(),
                intr      = intr.cpu(),
                extr      = extr.cpu(),
                name      = scene_id,
                img_names = img_names,
            ))
    return cache

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
# ----------------------------------------

# Gaussian Head（唯一需要训练）
print("[Init] build Gaussian Head …")
_emb = VGGT.from_pretrained("facebook/VGGT-1B"); EMB_DIM = _emb.embed_dim; del _emb
g_head = Gaussianhead(
    2*EMB_DIM, 3+3+4+3*(0+1)**2+1, activation="exp",
    conf_activation="expp1", sh_degree=0
).to(device)

opt = torch.optim.Adam(g_head.parameters(), lr=LR_RATE)
mse_fn  = nn.MSELoss()
lpips_fn= lpips.LPIPS(net="alex").to(device)
ssim_fn = SSIM(data_range=1.0).to(device)
LP_W, SS_W = 0.0, 0.0    # 若需 LPIPS/SSIM 打开权重
writer = SummaryWriter(LOG_DIR)

global_step = 0
start_epoch = 1

if MODEL_PATH is not None:
    print(f"[Init] load ckpt from {MODEL_PATH} …")
    ckpt = torch.load(MODEL_PATH, weights_only=True)
    g_head.load_state_dict(ckpt["model_state_dict"])
    opt.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    last_train_loss  = ckpt.get("loss_train", None)
    last_val_loss    = ckpt.get("loss_val",   None)
    print(f"[Init] Resuming from epoch {ckpt['epoch']}, train_loss={last_train_loss:.5f}, val_loss={last_val_loss:.5f}")
print("[Init] initialization done")

vggt = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)
for p in vggt.parameters(): p.requires_grad = False

for epoch in range(start_epoch, NUM_EPOCHS + 1):
    print(f"\n===== Epoch {epoch} =====")

    # ---- 1. DataLoader 重新随机 ----
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

    # ---- 2. VGGT 单次前向提特征并缓存（主+辅助） ----

    train_cache = build_cache_with_aux(train_loader, vggt)
    val_cache   = build_cache(val_loader,   vggt)
    # torch.cuda.empty_cache()

    # ---- 3. TRAIN ----
    g_head.train()
    viz_gt, viz_rd, viz_gt_val, viz_rd_val = [], [], [], []  # ← 收集可视化
    loss_train_epoch = 0.0
    for entry in tqdm(train_cache, desc="Train"):
        # 主视图数据
        imgs_main = entry['imgs_main'].to(device)            # (N_main,3,H,W)
        N_main, _, H, W = imgs_main.shape
        imgs_main_in = imgs_main.unsqueeze(1)                # (N_main,1,3,H,W)

        tok_list = [t.to(device) for t in entry['tok_list']]# list of length N_layers
        ps_idx    = entry['ps_idx']                          # patch start idx
        point_map = entry['point_map'].to(device)            # (N_main,1,H,W,3)

        intr_main = entry['intr_main'].to(device).float()    # (1,N_main,3,3) or similar
        extr_main = entry['extr_main'].to(device).float()    # (1,N_main,3,4)
        # 补全 extr_main 到 4x4
        pad = torch.tensor([0,0,0,1], device=extr_main.device).view(1,1,1,4)
        extr_main = torch.cat([extr_main, pad.expand_as(extr_main[:,:,:1,:])], dim=-2)

        # Gaussian Head 前向（只用主视图）
        gdict_raw = g_head(tok_list, imgs_main_in, ps_idx, point_map)
        gdict     = flatten_gdict(gdict_raw, B=N_main)

        # 对齐 gdict 每个张量形状
        for k,v in gdict.items():
            if v.ndim == 3: gdict[k] = v.reshape(1,-1,v.shape[-1])
            else: gdict[k] = v.reshape(1,-1)

        # 渲染主视图
        renders_main = render_gaussians(gdict, intr_main, extr_main, H, W)
        viz_gt.append(imgs_main.cpu())
        viz_rd.append(renders_main.cpu())

        # 辅助视图数据
        imgs_aux = entry['imgs_aux'].to(device)               # (N_aux,3,H,W)
        intr_aux = entry['intr_aux'].to(device).float()       # (1,N_aux,3,3)
        extr_aux = entry['extr_aux'].to(device).float()       # (1,N_aux,3,4)
        pad_aux  = torch.tensor([0,0,0,1], device=extr_aux.device).view(1,1,1,4)
        extr_aux = torch.cat([extr_aux, pad_aux.expand_as(extr_aux[:,:,:1,:])], dim=-2)

        # 渲染辅助视图
        renders_aux = render_gaussians(gdict, intr_aux, extr_aux, H, W)
        viz_gt.append(imgs_aux.cpu())
        viz_rd.append(renders_aux.cpu())

        # 计算并累加损失
        loss_main = mse_fn(renders_main, imgs_main)
        loss_aux  = mse_fn(renders_aux, imgs_aux)
        loss = loss_main + AUX_WEIGHT * loss_aux

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_train_epoch += loss.item()
        writer.add_scalar("loss/train_batch", loss.item(), global_step)
        global_step += 1

    # epoch 级别日志
    loss_train_epoch /= len(train_cache)
    writer.add_scalar("loss/train_epoch", loss_train_epoch, epoch)
    print(f"train loss = {loss_train_epoch:.5f}")

    # ---- 4. VALIDATION ----
    g_head.eval(); loss_val_epoch = 0.0
    with torch.no_grad():
        for entry in tqdm(val_cache, desc="Val  "):
            imgs = entry["imgs"].to(device); N, _, H, W = imgs.shape
            imgs_in = imgs.unsqueeze(1)
            tok = [t.to(device) for t in entry["tok_list"]]
            intr = entry["intr"].to(device).float()
            extr = entry["extr"].to(device).float()
            pad  = torch.tensor([0,0,0,1], device=extr.device).view(1,1,1,4)
            extr = torch.cat([extr, pad.expand_as(extr[:,:,:1,:])], dim=-2)

            gdict_raw = g_head(tok, imgs_in, entry["ps_idx"], entry["point_map"].to(device))
            gdict = flatten_gdict(gdict_raw, B=N)
            for k,v in gdict.items():
                if v.ndim == 3: gdict[k] = v.reshape(1,-1,v.shape[-1])
                else: gdict[k] = v.reshape(1,-1)

            renders = render_gaussians(gdict, intr, extr, H, W)
            viz_gt_val.append(imgs.cpu())
            viz_rd_val.append(renders.cpu())
            loss_val_epoch += mse_fn(renders, imgs).item()

    loss_val_epoch /= len(val_cache)
    writer.add_scalar("loss/val_epoch", loss_val_epoch, epoch)
    print(f"val   loss = {loss_val_epoch:.5f}")

    # ---- 5. 保存 (GT|Render) 拼图 以及 ckpt ----
    if epoch % IMG_LOG_INTERVAL == 0:
        grid = torch.cat([torch.cat(viz_gt, 0), torch.cat(viz_rd, 0)], 0)
        save_image(
            grid,
            os.path.join(IMG_LOG_DIR, f"epoch_{epoch:04d}_tr.png"),
            nrow=IMG_NUM_TRAIN+IMG_NUM_AUX,           # 每行放 IMG_NUM_TRAIN 张
            normalize=True, value_range=(0,1)
        )

        grid_val = torch.cat([torch.cat(viz_gt_val, 0), torch.cat(viz_rd_val, 0)], 0)
        save_image(
            grid_val,
            os.path.join(IMG_LOG_DIR, f"epoch_{epoch:04d}_val.png"),
            nrow=IMG_NUM_VAL,           # 每行放 IMG_NUM_TRAIN 张
            normalize=True, value_range=(0,1)
        )

    if epoch % CKPT_INTERVAL == 0:
        ckpt = {
            "epoch":                epoch,                   # 当前 epoch
            "model_state_dict":     g_head.state_dict(),     # Gaussian Head 权重
            "optimizer_state_dict": opt.state_dict(),        # 优化器状态
            "loss_train":           loss_train_epoch,        # 最后一个 epoch 训练损失
            "loss_val":             loss_val_epoch,          # 最后一个 epoch 验证损失
        }
        ckpt_path = os.path.join(CKPT_DIR, f"gauss_head_ckpt_epoch{epoch:04d}.pth")
        torch.save(ckpt, ckpt_path)
        print(f"[Checkpoint] Full checkpoint saved to {ckpt_path}")


    # ---- 6. 清理 ----
    # del train_cache, val_cache, viz_gt, viz_rd, viz_gt_val, viz_rd_val
    # gc.collect(); torch.cuda.empty_cache()

writer.close()
print("Training complete ✅")