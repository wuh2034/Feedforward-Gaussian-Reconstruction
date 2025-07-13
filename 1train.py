# 1train.py ────────────────────────────────────────────────────────────────
"""
训练脚本（按-batch 缓存 + 双验证策略 + 可视化 + per-batch 指标）
预热：前 WARMUP_EPOCHS 仅用 MSE；之后恢复 MSE + LPIPS + SSIM
"""

import os, gc, random, torch, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import lpips
from piq import ssim
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchvision.utils import save_image

from vggt.models.vggt import VGGT
from vggt.heads.gaussian_head import Gaussianhead
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.renderer_gsplat import render_gaussians
from Newdataloader import build_train_val

# ──────────── Config ────────────
ROOT_DIR        = "/usr/prakt/s0012/scannetpp/data"
TRAIN_SPLIT     = "/usr/prakt/s0012/scannetpp/splits/nvs_sem_train_800.txt"
VAL_SPLIT       = "/usr/prakt/s0012/scannetpp/splits/nvs_sem_val.txt"

SCENES_PER_BATCH = 20
IMG_NUM_MAIN     = 4
IMG_NUM_AUX      = 2
STRIDE           = 5

VAL_EVERY_BATCH  = True
QUICK_VAL_SCENES = 2
QUICK_VAL_IMGS   = 2

VAL_SCENE_LIMIT  = 5
VAL_IMG_NUM_MAIN = IMG_NUM_MAIN
VAL_IMG_NUM_AUX  = IMG_NUM_AUX

ENABLE_AUX   = False
NUM_EPOCHS   = 3001
NUM_WORKERS  = 4
VIS_EPOCH    = 1
CKPT_INTERVAL= 5
WARMUP_EPOCHS= 10                      ##### ★ 预热 epoch 数 ★#####

LOG_DIR  = "runs/withoutAUX_20_4"
CKPT_DIR = "checkpoints/withoutAUX_20_4"
IMG_DIR  = "renders/withoutAUX_20_4"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

L_MSE, L_LP, L_SS = 1.0, 0.2, 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
print(f"[Init] device={device}, dtype={dtype}")

# ──────────── Utils ────────────
def flatten_gdict(gdict: dict, B: int):
    return {k: v.reshape(B, -1, v.shape[-1] if v.ndim >= 2 else 1) for k, v in gdict.items()}

@torch.no_grad()
def build_cache(loader, vggt_model, use_aux: bool, desc: str):
    """前向 VGGT → CPU 缓存"""
    cache = []
    with torch.amp.autocast("cuda", dtype=dtype):
        for batch in tqdm(loader, desc=desc, leave=False):
            if use_aux:
                imgs_m, imgs_a, *_ = batch
                cat = torch.cat([imgs_m, imgs_a], 0).to(device)
                N_m, _, H, W = imgs_m.shape
                tok_all, ps   = vggt_model.aggregator(cat.unsqueeze(1))
                preds_all     = vggt_model(cat)
                pmap_all      = preds_all["world_points"].unsqueeze(1)
                extr, intr    = pose_encoding_to_extri_intri(preds_all["pose_enc"], (H, W))
                cache.append(dict(
                    imgs_main=imgs_m.cpu(), imgs_aux=imgs_a.cpu(),
                    tok_list=[t[:N_m].cpu() for t in tok_all], ps_idx=ps,
                    point_map=pmap_all[:, :, :N_m].cpu(),
                    intr_main=intr[:, :N_m].cpu(), extr_main=extr[:, :N_m].cpu(),
                    intr_aux=intr[:, N_m:].cpu(),  extr_aux=extr[:, N_m:].cpu(),
                ))
            else:
                imgs, *_ = batch
                imgs = imgs.to(device)
                N, _, H, W = imgs.shape
                tok_list, ps = vggt_model.aggregator(imgs.unsqueeze(1))
                preds        = vggt_model(imgs)
                pmap         = preds["world_points"].unsqueeze(1)
                extr, intr   = pose_encoding_to_extri_intri(preds["pose_enc"], (H, W))
                cache.append(dict(
                    imgs=imgs.cpu(), tok_list=[t.cpu() for t in tok_list],
                    ps_idx=ps, point_map=pmap.cpu(),
                    intr=intr.cpu(), extr=extr.cpu(),
                ))
    return cache

# ──────────── Model & Loss ────────────
print("[Init] Gaussian Head …")
EMB_DIM = VGGT.from_pretrained("facebook/VGGT-1B").embed_dim
g_head  = Gaussianhead(
    2*EMB_DIM,
    3+3+4+3*(0+1)**2+1,
    activation="exp",
    conf_activation="expp1",
    sh_degree=0,
).to(device)

opt     = torch.optim.Adam(g_head.parameters(), 1e-4)
mse_fn  = nn.MSELoss()
lpips_fn= lpips.LPIPS(net="alex").to(device)
psnr_fn = PSNR(data_range=1.0).to(device)

writer, step = SummaryWriter(LOG_DIR), 0

# ──────────── Helper: val 评估 ────────────
@torch.no_grad()
def evaluate_cache(val_cache, use_aux=False, return_visual=False):   ##### ★ 新增参数 ★#####
    mse_acc = psnr_acc = ssim_acc = lpips_acc = 0.0
    first_gt = first_rd = None                                       ##### ★
    for idx, e in enumerate(val_cache):
        if use_aux and "imgs_aux" in e:
            imgs_m = e["imgs_main"].to(device)
            imgs_a = e["imgs_aux"].to(device)
            N, _, H, W = imgs_m.shape
            tok   = [t.to(device) for t in e["tok_list"]]

            extr_m = torch.cat([e["extr_main"].to(device).float(),
                                torch.tensor([0,0,0,1], device=device).view(1,1,1,4)
                                .expand(1,N,1,4)], -2)
            intr_m = e["intr_main"].to(device).float()
            gd_raw = g_head(tok, imgs_m.unsqueeze(1), e["ps_idx"], e["point_map"].to(device))
            gd     = {k:(v.view(1,-1,v.shape[-1]) if v.ndim==3 else v.view(1,-1))
                      for k,v in flatten_gdict(gd_raw, N).items()}
            rd_m   = render_gaussians(gd, intr_m, extr_m, H, W)

            N_a = imgs_a.shape[0]
            extr_a = torch.cat([e["extr_aux"].to(device).float(),
                                torch.tensor([0,0,0,1], device=device).view(1,1,1,4)
                                .expand(1,N_a,1,4)], -2)
            intr_a = e["intr_aux"].to(device).float()
            rd_a   = render_gaussians(gd, intr_a, extr_a, H, W)

            rd_stack = torch.cat([rd_m, rd_a], 0)
            gt_stack = torch.cat([imgs_m, imgs_a], 0)
            mse_acc += mse_fn(rd_m, imgs_m).item() + 0.5*mse_fn(rd_a, imgs_a).item()
        else:
            imgs = e["imgs"].to(device)
            N, _, H, W = imgs.shape
            tok  = [t.to(device) for t in e["tok_list"]]
            extr = torch.cat([e["extr"].to(device).float(),
                              torch.tensor([0,0,0,1], device=device).view(1,1,1,4)
                              .expand(1,N,1,4)], -2)
            intr = e["intr"].to(device).float()
            gd_raw = g_head(tok, imgs.unsqueeze(1), e["ps_idx"], e["point_map"].to(device))
            gd = {k:(v.view(1,-1,v.shape[-1]) if v.ndim==3 else v.view(1,-1))
                  for k,v in flatten_gdict(gd_raw, N).items()}
            rd = render_gaussians(gd, intr, extr, H, W)

            rd_stack, gt_stack = rd, imgs
            mse_acc += mse_fn(rd, imgs).item()

        psnr_acc += psnr_fn(rd_stack, gt_stack).item()
        ssim_acc += ssim(rd_stack.cpu(), gt_stack.cpu(), data_range=1.0).mean().item()
        lpips_acc+= lpips_fn(rd_stack, gt_stack).mean().item()

        if return_visual and first_gt is None:                ##### ★ 记录首个场景样张 ★#####
            first_gt = gt_stack.cpu()
            first_rd = rd_stack.cpu()

    n = len(val_cache)
    out = (mse_acc/n, psnr_acc/n, ssim_acc/n, lpips_acc/n)
    if return_visual:
        return (*out, first_gt, first_rd)
    return out

# ──────────── Training Loop ────────────
for ep in range(1, NUM_EPOCHS + 1):
    print(f"\n======= Epoch {ep} =======")

    lp_w = 0.0 if ep <= WARMUP_EPOCHS else L_LP
    ss_w = 0.0 if ep <= WARMUP_EPOCHS else L_SS

    with open(TRAIN_SPLIT) as f:
        train_ids_all = [l.strip() for l in f if l.strip()]
    random.shuffle(train_ids_all)

    total_batches = len(train_ids_all) // SCENES_PER_BATCH
    vggt = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)
    for p in vggt.parameters(): p.requires_grad_(False)

    if VAL_EVERY_BATCH:
        with open(VAL_SPLIT) as f:
            quick_ids = [l.strip() for i, l in enumerate(f) if i < QUICK_VAL_SCENES]
        _, quick_val_loader = build_train_val(
            ROOT_DIR, TRAIN_SPLIT, VAL_SPLIT,
            None, quick_ids,
            IMG_NUM_MAIN, QUICK_VAL_IMGS,
            stride=STRIDE, num_workers=NUM_WORKERS)

    save_vis = (ep % VIS_EPOCH == 0)
    vis_gt, vis_rd = [], []
    val_vis_gt = val_vis_rd = None                           ##### ★ 新增 val 可视化容器 ★#####

    for bi in range(total_batches):
        batch_ids = train_ids_all[bi*SCENES_PER_BATCH:(bi+1)*SCENES_PER_BATCH]
        train_loader, _ = build_train_val(
            ROOT_DIR, TRAIN_SPLIT, VAL_SPLIT,
            batch_ids, None,
            IMG_NUM_MAIN, IMG_NUM_MAIN,
            stride=STRIDE, num_workers=NUM_WORKERS)

        cache = build_cache(train_loader, vggt, ENABLE_AUX, f"⚙️ {bi+1}/{total_batches}")
        opt.zero_grad(set_to_none=True)
        loss_sum = psnr_sum = ssim_sum = lpips_sum = 0.0

        for e in cache:
            if ENABLE_AUX:
                imgs_m = e["imgs_main"].to(device)
                imgs_a = e["imgs_aux"].to(device)
                N, _, H, W = imgs_m.shape
                tok = [t.to(device) for t in e["tok_list"]]

                extr_m = torch.cat([e["extr_main"].to(device).float(),
                                    torch.tensor([0,0,0,1], device=device).view(1,1,1,4)
                                    .expand(1,N,1,4)], -2)
                intr_m = e["intr_main"].to(device).float()
                gd_raw = g_head(tok, imgs_m.unsqueeze(1),
                                e["ps_idx"], e["point_map"].to(device))
                gd = {k:(v.view(1,-1,v.shape[-1]) if v.ndim==3 else v.view(1,-1))
                      for k,v in flatten_gdict(gd_raw, N).items()}
                rd_m = render_gaussians(gd, intr_m, extr_m, H, W)

                N_a = imgs_a.shape[0]
                extr_a = torch.cat([e["extr_aux"].to(device).float(),
                                    torch.tensor([0,0,0,1], device=device).view(1,1,1,4)
                                    .expand(1,N_a,1,4)], -2)
                intr_a = e["intr_aux"].to(device).float()
                rd_a   = render_gaussians(gd, intr_a, extr_a, H, W)

                loss = (L_MSE*(mse_fn(rd_m, imgs_m) + 0.5*mse_fn(rd_a, imgs_a))
                        + lp_w *(lpips_fn(rd_m, imgs_m).mean() + 0.5*lpips_fn(rd_a, imgs_a).mean())
                        + ss_w *((1-ssim(rd_m, imgs_m, data_range=1.0).mean())
                                 +0.5*(1-ssim(rd_a, imgs_a, data_range=1.0).mean())))

                rd_stack = torch.cat([rd_m, rd_a], 0)
                gt_stack = torch.cat([imgs_m, imgs_a], 0)

                if save_vis and bi == total_batches-1:
                    vis_gt.append(gt_stack.cpu())
                    vis_rd.append(rd_stack.detach().cpu())
            else:
                imgs = e["imgs"].to(device)
                N, _, H, W = imgs.shape
                tok = [t.to(device) for t in e["tok_list"]]
                extr = torch.cat([e["extr"].to(device).float(),
                                  torch.tensor([0,0,0,1], device=device).view(1,1,1,4)
                                  .expand(1,N,1,4)], -2)
                intr = e["intr"].to(device).float()
                gd_raw = g_head(tok, imgs.unsqueeze(1),
                                e["ps_idx"], e["point_map"].to(device))
                gd = {k:(v.view(1,-1,v.shape[-1]) if v.ndim==3 else v.view(1,-1))
                      for k,v in flatten_gdict(gd_raw, N).items()}
                rd = render_gaussians(gd, intr, extr, H, W)
                loss = L_MSE * mse_fn(rd, imgs)

                rd_stack, gt_stack = rd, imgs
                if save_vis and bi == total_batches-1:
                    vis_gt.append(gt_stack.cpu())
                    vis_rd.append(rd_stack.detach().cpu())

            loss.backward()
            loss_sum  += loss.item()
            psnr_sum  += psnr_fn(rd_stack, gt_stack).item()
            ssim_sum  += ssim(rd_stack.cpu(), gt_stack.cpu(), data_range=1.0).mean().item()
            lpips_sum += lpips_fn(rd_stack, gt_stack).mean().item()

        opt.step()
        n_scene = len(cache)
        train_loss = loss_sum / n_scene
        psnr_t = psnr_sum / n_scene
        ssim_t = ssim_sum / n_scene
        lpips_t= lpips_sum / n_scene

        writer.add_scalars("loss", {"train": train_loss}, step)
        writer.add_scalar("metric/psnr_train", psnr_t, step)
        writer.add_scalar("metric/ssim_train", ssim_t, step)
        writer.add_scalar("metric/lpips_train", lpips_t, step)

        if VAL_EVERY_BATCH:
            q_cache = build_cache(quick_val_loader, vggt, ENABLE_AUX,
                                  f"🔍 Val {bi+1}/{total_batches}")
            val_loss, psnr_v, ssim_v, lpips_v, v_gt, v_rd = evaluate_cache(   ##### ★
                q_cache, ENABLE_AUX, return_visual=True)                      ##### ★
            if save_vis and bi == total_batches-1:                            ##### ★
                val_vis_gt, val_vis_rd = v_gt, v_rd                           ##### ★

            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, step)
            writer.add_scalar("metric/psnr_val",  psnr_v, step)
            writer.add_scalar("metric/ssim_val",  ssim_v, step)
            writer.add_scalar("metric/lpips_val", lpips_v, step)
            print(f"[B {bi+1}/{total_batches}] "
                  f"train={train_loss:.4f} | val={val_loss:.4f} "
                  f"(PSNR {psnr_t:.2f}/{psnr_v:.2f})")
            del q_cache
        else:
            print(f"[B {bi+1}/{total_batches}] train={train_loss:.4f} "
                  f"(PSNR {psnr_t:.2f})")

        step += 1
        del cache; gc.collect(); torch.cuda.empty_cache()

    # ---------- 保存可视化 ----------
    if save_vis and vis_gt:
        grid_tr = torch.cat([torch.cat(vis_gt, 0), torch.cat(vis_rd, 0)], 0)
        per_scene_tr = IMG_NUM_MAIN + (IMG_NUM_AUX if ENABLE_AUX else 0)
        save_image(grid_tr, f"{IMG_DIR}/epoch_{ep:04d}_train.png",
                   nrow=per_scene_tr, normalize=True, value_range=(0,1))
        print(f"[Vis] saved renders/epoch_{ep:04d}_train.png")

        if val_vis_gt is not None:
            per_scene_val = QUICK_VAL_IMGS if VAL_EVERY_BATCH else VAL_IMG_NUM_MAIN
            per_scene_val += (VAL_IMG_NUM_AUX if ENABLE_AUX else 0)
            grid_val = torch.cat([val_vis_gt, val_vis_rd], 0)
            save_image(grid_val, f"{IMG_DIR}/epoch_{ep:04d}_val.png",
                       nrow=per_scene_val, normalize=True, value_range=(0,1))
            print(f"[Vis] saved renders/epoch_{ep:04d}_val.png")

    # ---------- epoch-end 验证 ----------
    if not VAL_EVERY_BATCH:
        val_ids=[]
        with open(VAL_SPLIT) as f:
            for i, l in enumerate(f):
                if i>=VAL_SCENE_LIMIT: break
                val_ids.append(l.strip())
        _, val_loader = build_train_val(
            ROOT_DIR, TRAIN_SPLIT, VAL_SPLIT,
            None, val_ids,
            VAL_IMG_NUM_MAIN, VAL_IMG_NUM_MAIN,
            val_img_aux=VAL_IMG_NUM_AUX if ENABLE_AUX else 0,
            stride=STRIDE, num_workers=NUM_WORKERS)
        val_cache = build_cache(val_loader, vggt, ENABLE_AUX, "🔍 Val epoch")
        v_loss, psnr_v, ssim_v, lpips_v, v_gt, v_rd = evaluate_cache(      ##### ★
            val_cache, ENABLE_AUX, return_visual=True)                     ##### ★
        writer.add_scalars("loss", {"val": v_loss}, step-1)
        writer.add_scalar("metric/psnr_val",  psnr_v, step-1)
        writer.add_scalar("metric/ssim_val",  ssim_v, step-1)
        writer.add_scalar("metric/lpips_val", lpips_v, step-1)
        if save_vis:
            val_vis_gt, val_vis_rd = v_gt, v_rd                             ##### ★ 保存可视化 ★#####
        print(f"[Epoch {ep}] val={v_loss:.4f}")
        del val_cache

    if ep % CKPT_INTERVAL == 0:
        torch.save({"epoch": ep,
                    "model": g_head.state_dict(),
                    "opt": opt.state_dict()},
                   f"{CKPT_DIR}/epoch_{ep:04d}.pth")

    del vggt; gc.collect(); torch.cuda.empty_cache()

writer.close()
print("Training complete ✅")