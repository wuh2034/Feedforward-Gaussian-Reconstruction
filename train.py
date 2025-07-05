# train.py ─――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
"""
训练流程（满足最新需求）：
    for epoch:
        1. 调 build_train_val 重新生成 train/val DataLoader
           （内部 RandomSampler 保证每 epoch 抽不同 scene）
        2. ──【特征提取阶段】─────────────────────────────
           2.1 实例化 VGGT (fp16/bf16) → GPU
           2.2 遍历 train_loader  →   cache_train  (放 CPU)
               遍历 val_loader    →   cache_val
           2.3 del vggt; torch.cuda.empty_cache()  # 释放显存
        3. ──【优化阶段】───────────────────────────────
           3.1 使用 cache_train 做前向 + BP，更新 Gaussian Head
           3.2 使用 cache_val   只做前向，统计验证损失
        4. log 结果 & 保存可视化
"""

import os, gc, torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm
import lpips
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

from vggt.models.vggt import VGGT
from vggt.heads.gaussian_head import Gaussianhead
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.renderer_gsplat import render_gaussians

from dataloader import build_train_val

# =============== hyper params ===============================================
ROOT_DIR            = "/usr/prakt/s0012/scannetpp/data"
TRAIN_SPLIT_TXT     = "/usr/prakt/s0012/scannetpp/splits/nvs_sem_train.txt"
VAL_SPLIT_TXT       = "/usr/prakt/s0012/scannetpp/splits/nvs_sem_val.txt"
TRAIN_BATCH_SCENES  = 4
VAL_BATCH_SCENES    = 2
IMG_NUM_TRAIN       = 4
IMG_NUM_VAL         = 4
STRIDE              = 3
NUM_WORKERS         = 4
NUM_EPOCHS          = 30000
LOG_DIR             = "runs/scannetpp"
IMG_LOG_DIR         = "renders/scannetpp"
os.makedirs(IMG_LOG_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if (device=="cuda" and torch.cuda.get_device_capability()[0]>=8) else torch.float16
print(f"[Init] device={device}, dtype={dtype}")

# ------------------- helper --------------------------------------------------
def flatten_gdict(gdict: dict, B: int):
    out = {}
    for k, v in gdict.items():
        C = v.shape[-1] if v.ndim >= 2 else 1
        out[k] = v.reshape(B, -1, C)
    return out

@torch.no_grad()
def build_cache(loader, vggt_model):
    """对 loader 进行 VGGT 前向，返回 list(dict)，并全部搬到 CPU。"""
    cache = []
    with torch.amp.autocast("cuda", dtype=dtype):
        for imgs, scene_id, img_names in loader:              # imgs:(N,3,H,W)
            print(f"[Select] scene {scene_id}: {', '.join(img_names)}")
            imgs = imgs.to(device)
            N, _, H, W = imgs.shape
            imgs_in = imgs.unsqueeze(1)           # (N,1,3,H,W)

            tok_list, ps_idx = vggt_model.aggregator(imgs_in)
            preds            = vggt_model(imgs)
            point_map        = preds["world_points"].unsqueeze(1)
            pose_enc         = preds["pose_enc"]
            extr, intr       = pose_encoding_to_extri_intri(pose_enc, (H, W))

            cache.append({
                "imgs"     : imgs.cpu(),
                "tok_list" : [t.cpu() for t in tok_list],
                "ps_idx"   : ps_idx,
                "point_map": point_map.cpu(),
                "intr"     : intr.cpu(),
                "extr"     : extr.cpu(),
                "name"     : scene_id,
                "img_names": img_names,
            })
    return cache
# -----------------------------------------------------------------------------

# ------------------- Gaussian Head & loss ------------------------------------
print("[Init] building Gaussian Head …")
embed_demo = VGGT.from_pretrained("facebook/VGGT-1B")
embed_dim  = embed_demo.embed_dim
del embed_demo
out_dim = 3 + 3 + 4 + 3*(0+1)**2 + 1
g_head  = Gaussianhead(2*embed_dim, out_dim,
                       activation="exp", conf_activation="expp1",
                       sh_degree=0).to(device)
opt        = torch.optim.Adam(g_head.parameters(), lr=1e-4)
crit_mse   = nn.MSELoss()
lpips_fn   = lpips.LPIPS(net="alex").to(device)
ssim_fn    = SSIM(data_range=1.0).to(device)
LP_W, SS_W = 0.2, 0.2
writer     = SummaryWriter(LOG_DIR)

# ============================================================================#
global_step = 0
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n========== Epoch {epoch} ==========")

    # 1. 重新生成 loader（保证每 epoch 随机抽 scene）
    train_loader, val_loader = build_train_val(
        root_dir           = ROOT_DIR,
        train_split_txt    = TRAIN_SPLIT_TXT,
        val_split_txt      = VAL_SPLIT_TXT,
        train_batch_scenes = TRAIN_BATCH_SCENES,
        train_img_num      = IMG_NUM_TRAIN,
        val_batch_scenes   = VAL_BATCH_SCENES,
        val_img_num        = IMG_NUM_VAL,
        stride             = STRIDE,
        num_workers        = NUM_WORKERS, # num of cpu
    )

    # 2. 特征提取阶段：实例化 VGGT → cache → 释放
    print("[Epoch] VGGT forward & caching …")
    vggt = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)
    for p in vggt.parameters(): p.requires_grad = False

    train_cache = build_cache(train_loader, vggt)
    val_cache   = build_cache(val_loader,   vggt)

    del vggt
    torch.cuda.empty_cache()

    # 3. TRAIN -----------------------------------------------------------------
    g_head.train()
    loss_train_epoch = 0.0
    for entry in tqdm(train_cache, desc="Train"):
        imgs      = entry["imgs"].to(device)
        point_map = entry["point_map"].to(device)
        tok_list  = [t.to(device) for t in entry["tok_list"]]
        ps_idx    = entry["ps_idx"]
        intr      = entry["intr"].to(device).float()  # (1,N,3,3)
        extr      = entry["extr"].to(device).float()  # (1,N,3,4)

        # pad extr to 4×4
        pad = torch.tensor([0,0,0,1], device=extr.device).view(1,1,1,4)
        extr = torch.cat([extr, pad.expand_as(extr[:,:,:1,:])], dim=-2)

        N, _, H, W = imgs.shape
        imgs_in = imgs.unsqueeze(1)

        gdict_raw = g_head(tok_list, imgs_in, ps_idx, point_map)
        gdict     = flatten_gdict(gdict_raw, B=N)
        for k,v in gdict.items():
            if v.ndim == 3: gdict[k] = v.reshape(1,-1,v.shape[-1])
            else:           gdict[k] = v.reshape(1,-1)

        renders = render_gaussians(gdict, intr, extr, H, W)

        with torch.amp.autocast("cuda", dtype=dtype):
            mse   = crit_mse(renders, imgs)
            # lp    = lpips_fn(renders*2-1, imgs*2-1).mean()
            # ssim  = 1 - ssim_fn(renders, imgs)
            # loss  = mse + LP_W*lp + SS_W*ssim
            loss  = mse

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_train_epoch += loss.item()
        writer.add_scalar("loss/train_batch", loss.item(), global_step)
        global_step += 1

    loss_train_epoch /= len(train_cache)
    writer.add_scalar("loss/train_epoch", loss_train_epoch, epoch)
    print(f"[Epoch] train loss = {loss_train_epoch:.5f}")

    # 4. VALIDATION ------------------------------------------------------------
    g_head.eval()
    loss_val_epoch = 0.0
    with torch.no_grad():
        for entry in tqdm(val_cache, desc="Val  "):
            imgs      = entry["imgs"].to(device)
            point_map = entry["point_map"].to(device)
            tok_list  = [t.to(device) for t in entry["tok_list"]]
            ps_idx    = entry["ps_idx"]
            intr      = entry["intr"].to(device).float()
            extr      = entry["extr"].to(device).float()
            pad       = torch.tensor([0,0,0,1], device=extr.device).view(1,1,1,4)
            extr      = torch.cat([extr, pad.expand_as(extr[:,:,:1,:])], dim=-2)

            N, _, H, W = imgs.shape
            imgs_in = imgs.unsqueeze(1)

            gdict_raw = g_head(tok_list, imgs_in, ps_idx, point_map)
            gdict     = flatten_gdict(gdict_raw, B=N)
            for k,v in gdict.items():
                if v.ndim == 3: gdict[k] = v.reshape(1,-1,v.shape[-1])
                else:           gdict[k] = v.reshape(1,-1)

            renders = render_gaussians(gdict, intr, extr, H, W)

            mse   = crit_mse(renders, imgs)
            # lp    = lpips_fn(renders*2-1, imgs*2-1).mean()
            # ssim  = 1 - ssim_fn(renders, imgs)
            # loss  = mse + LP_W*lp + SS_W*ssim
            loss = mse
            loss_val_epoch += loss.item()

    loss_val_epoch /= len(val_cache)
    writer.add_scalar("loss/val_epoch", loss_val_epoch, epoch)
    print(f"[Epoch] val   loss = {loss_val_epoch:.5f}")

    # 5. 可视化
    if epoch % 10 == 0:
        save_image(
            torch.cat([imgs, renders], 0).cpu(),
            os.path.join(IMG_LOG_DIR, f"epoch_{epoch:05d}.png"),
            normalize=True, value_range=(0,1)
        )

    # 6. 清理
    del train_cache, val_cache
    gc.collect()
    torch.cuda.empty_cache()

writer.close()
print("Training complete ✅")