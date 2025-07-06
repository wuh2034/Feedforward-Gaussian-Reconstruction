# train.py  ──────────────────────────────────────────────────────────────────
"""
每个 epoch：
    1. 重新构建 train/val DataLoader（随机抽 scene）
    2. VGGT 前向提特征 → 缓存到 CPU
    3. 释放 VGGT 显存
    4. 用 Gaussian Head 训练，再做验证
    5. 保存 (GT|Render) 拼图，数量 = TRAIN_BATCH_SCENES×IMG_NUM_TRAIN
"""

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

# ----------------- 训练超参 -----------------
ROOT_DIR            = "/usr/prakt/s0012/scannetpp/data"
TRAIN_SPLIT_TXT     = "/usr/prakt/s0012/scannetpp/splits/nvs_sem_train.txt"
VAL_SPLIT_TXT       = "/usr/prakt/s0012/scannetpp/splits/nvs_sem_val.txt"
TRAIN_BATCH_SCENES  = 4           # 每 epoch 随机抽 4 个 scene 训练
VAL_BATCH_SCENES    = 2
IMG_NUM_TRAIN       = 4           # 每 scene 取 4 张图
IMG_NUM_VAL         = 4
STRIDE              = 3
NUM_WORKERS         = 4
NUM_EPOCHS          = 30000
LOG_DIR             = "runs/scannetpp"
IMG_LOG_DIR         = "renders/scannetpp"
os.makedirs(IMG_LOG_DIR, exist_ok=True)

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
            print(f"[Select] scene {scene_id}: {', '.join(img_names)}")
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
# ----------------------------------------

# Gaussian Head（唯一需要训练）
print("[Init] build Gaussian Head …")
_emb = VGGT.from_pretrained("facebook/VGGT-1B"); EMB_DIM = _emb.embed_dim; del _emb
g_head = Gaussianhead(
    2*EMB_DIM, 3+3+4+3*(0+1)**2+1, activation="exp",
    conf_activation="expp1", sh_degree=0
).to(device)
opt = torch.optim.Adam(g_head.parameters(), lr=1e-4)
mse_fn  = nn.MSELoss()
lpips_fn= lpips.LPIPS(net="alex").to(device)
ssim_fn = SSIM(data_range=1.0).to(device)
LP_W, SS_W = 0.0, 0.0    # 若需 LPIPS/SSIM 打开权重
writer = SummaryWriter(LOG_DIR)

global_step = 0
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n===== Epoch {epoch} =====")

    # ---- 1. DataLoader 重新随机 ----
    train_loader, val_loader = build_train_val(
        root_dir           = ROOT_DIR,
        train_split_txt    = TRAIN_SPLIT_TXT,
        val_split_txt      = VAL_SPLIT_TXT,
        train_batch_scenes = TRAIN_BATCH_SCENES,
        train_img_num      = IMG_NUM_TRAIN,
        val_batch_scenes   = VAL_BATCH_SCENES,
        val_img_num        = IMG_NUM_VAL,
        stride             = STRIDE,
        num_workers        = NUM_WORKERS,
    )

    # ---- 2. VGGT 前向提特征并缓存 ----
    vggt = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)
    for p in vggt.parameters(): p.requires_grad = False

    train_cache = build_cache(train_loader, vggt)
    val_cache   = build_cache(val_loader,   vggt)

    del vggt
    torch.cuda.empty_cache()

    # ---- 3. TRAIN ----
    g_head.train()
    viz_gt, viz_rd = [], []             # ← 收集可视化
    loss_train_epoch = 0.0
    for entry in tqdm(train_cache, desc="Train"):
        imgs   = entry["imgs"].to(device)
        N, _, H, W = imgs.shape
        imgs_in = imgs.unsqueeze(1)

        tok = [t.to(device) for t in entry["tok_list"]]
        intr = entry["intr"].to(device).float()
        extr = entry["extr"].to(device).float()
        pad  = torch.tensor([0,0,0,1], device=extr.device).view(1,1,1,4)
        extr = torch.cat([extr, pad.expand_as(extr[:,:,:1,:])], dim=-2)

        gdict_raw = g_head(tok, imgs_in, entry["ps_idx"], entry["point_map"].to(device))
        gdict     = flatten_gdict(gdict_raw, B=N)
        for k,v in gdict.items():
            if v.ndim == 3: gdict[k] = v.reshape(1,-1,v.shape[-1])
            else: gdict[k] = v.reshape(1,-1)

        renders = render_gaussians(gdict, intr, extr, H, W)

        viz_gt.append(imgs.cpu())
        viz_rd.append(renders.cpu())

        loss = mse_fn(renders, imgs)  # + LP_W*lpips ... 如需要
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        loss_train_epoch += loss.item()
        writer.add_scalar("loss/train_batch", loss.item(), global_step)
        global_step += 1

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
            loss_val_epoch += mse_fn(renders, imgs).item()

    loss_val_epoch /= len(val_cache)
    writer.add_scalar("loss/val_epoch", loss_val_epoch, epoch)
    print(f"val   loss = {loss_val_epoch:.5f}")

    # ---- 5. 保存 (GT|Render) 拼图 ----
    if epoch % 10 == 0:
        grid = torch.cat([torch.cat(viz_gt, 0), torch.cat(viz_rd, 0)], 0)
        save_image(
            grid,
            os.path.join(IMG_LOG_DIR, f"epoch_{epoch:04d}.png"),
            nrow=IMG_NUM_TRAIN,           # 每行放 IMG_NUM_TRAIN 张
            normalize=True, value_range=(0,1)
        )

    # ---- 6. 清理 ----
    del train_cache, val_cache, viz_gt, viz_rd
    gc.collect(); torch.cuda.empty_cache()

writer.close()
print("Training complete ✅")