#!/usr/bin/env python
# inference.py  ────────────────────────────────────────────────────────────
"""
单-Scene 推理脚本（随机或指定 scene）
────────────────────────────────────
• 从 nvs_test.txt 随机/指定 scene
• 连续抽取 num_imgs 帧 → 渲染
• 抽取的 GT 单独拷到父目录 /infer/
• 输出一张 2×N 网格：上 = GT， 下 = Render
• 导出 PLY 与指标
"""

import os, random, json, argparse, torch, torch.nn as nn
from torchvision.utils import save_image
import lpips
from piq import ssim
from torchmetrics import PeakSignalNoiseRatio as PSNR

from dataloader import build_loader, _read_split
from vggt.models.vggt import VGGT
from vggt.heads.gaussian_head import Gaussianhead
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.renderer_gsplat import render_gaussians

# ════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir",   default="/usr/prakt/s0012/scannetpp/data")
    p.add_argument("--test_split", default="/usr/prakt/s0012/scannetpp/splits/nvs_test.txt")
    p.add_argument("--ckpt",       default="/usr/prakt/s0012/ADL4VC/checkpoints/scannetpp25_12_4metric/epoch_0800.pth")
    p.add_argument("--num_imgs",   type=int, default=8)
    p.add_argument("--scene_id",   default=None)
    p.add_argument("--out_base",   default="/usr/prakt/s0012/ADL4VC/infer")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",       type=int, default=None)
    return p.parse_args()

# --------------------------------------------------------------------
def flatten_gdict(gdict: dict, B: int):
    return {k: v.reshape(B, -1, v.shape[-1] if v.ndim >= 2 else 1)
            for k, v in gdict.items()}

def _first_key(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None

def save_gaussians_to_ply(gdict: dict, path: str):
    xyz   = _first_key(gdict, ["means3D", "means"])
    if xyz is None:
        raise ValueError("gdict 缺少 means/means3D")
    rgb   = _first_key(gdict, ["rgb", "colors"])
    opa   = _first_key(gdict, ["opacity", "opacities"])
    scale = gdict.get("scales")

    xyz   = xyz[0].cpu().float()
    N     = xyz.size(0)
    rgb   = (rgb[0].cpu()*255).clamp(0,255).to(torch.uint8) if rgb is not None else torch.zeros_like(xyz,dtype=torch.uint8)
    opa   = opa[0].cpu().float() if opa is not None else torch.zeros(N,1)
    scale = scale[0].cpu().float() if scale is not None else torch.zeros_like(xyz)
    if scale.ndim == 1:
        scale = scale.unsqueeze(1).repeat(1,3)

    header = [
        "ply","format ascii 1.0",f"element vertex {N}",
        "property float x","property float y","property float z",
        "property uchar red","property uchar green","property uchar blue",
        "property float opacity",
        "property float scale_x","property float scale_y","property float scale_z",
        "end_header"
    ]
    body = [f"{x} {y} {z} {r} {g} {b} {a} {sx} {sy} {sz}"
            for (x,y,z),(r,g,b),a,(sx,sy,sz)
            in zip(xyz.tolist(), rgb.tolist(), opa.squeeze(1).tolist(), scale.tolist())]
    with open(path,"w") as f:
        f.write("\n".join(header+body))
    print(f"[PLY] saved {N} points → {path}")

# ════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    scene_id = args.scene_id or random.choice(_read_split(args.test_split))
    print(f"[Scene] {scene_id}")

    out_dir = os.path.join(args.out_base, scene_id)
    os.makedirs(out_dir, exist_ok=True)

    device = args.device
    dtype  = torch.bfloat16 if device=="cuda" and torch.cuda.get_device_capability()[0]>=8 else torch.float16
    print(f"[Init] device={device}, dtype={dtype}")

    # -------- Data --------
    loader = build_loader(
        root_dir=args.root_dir,
        scene_ids=[scene_id],
        batch_scenes=1,
        img_num=args.num_imgs,
        stride=1,
        shuffle=False,
        num_workers=2,
        verbose=True,
    )
    imgs, _, img_names = next(iter(loader))   # (N,3,H,W)
    imgs = imgs.to(device)
    N, _, H, W = imgs.shape

    # --- 把抽取的 GT 存到父目录 ------------------------
    for i,(im,name) in enumerate(zip(imgs, img_names)):
        save_image(im.cpu(),
                   os.path.join(args.out_base, f"{scene_id}_input_{i:02d}.png"),
                   normalize=True, value_range=(0,1))

    # -------- Model --------
    vggt = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)
    for p in vggt.parameters(): p.requires_grad_(False)
    g_head = Gaussianhead(
        2*vggt.embed_dim,
        3+3+4+3*(0+1)**2+1,
        activation="exp",
        conf_activation="expp1",
        sh_degree=0
    ).to(device)

    g_head.load_state_dict(torch.load(args.ckpt,map_location=device).get("model_state_dict", torch.load(args.ckpt,map_location=device)), strict=True)
    g_head.eval()

    # -------- Inference --------
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        tok, ps_idx      = vggt.aggregator(imgs.unsqueeze(1))
        preds            = vggt(imgs)
        point_map        = preds["world_points"].unsqueeze(1)
        extr, intr       = pose_encoding_to_extri_intri(preds["pose_enc"], (H,W))

        gdict_raw = g_head(tok, imgs.unsqueeze(1), ps_idx, point_map)
        gdict     = {k:(v.view(1,-1,v.shape[-1]) if v.ndim==3 else v.view(1,-1))
                     for k,v in flatten_gdict(gdict_raw, B=N).items()}
        gdict.setdefault("means3D", gdict.get("means"))
        gdict.setdefault("rgb",     gdict.get("colors"))
        gdict.setdefault("opacity", gdict.get("opacities"))

        renders = render_gaussians(
            gdict,
            intr.to(device).float(),
            torch.cat([extr.to(device).float(),
                       torch.tensor([0,0,0,1],device=device).view(1,1,1,4).expand(1,N,1,4)],dim=-2),
            H, W)

    # -------- Metrics --------
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    psnr_fn  = PSNR(data_range=1.0).to(device)
    metrics  = []
    for i in range(N):
        gt = imgs[i:i+1]; rd = renders[i:i+1]
        metrics.append(dict(
            idx=i, name=img_names[i],
            psnr=psnr_fn(rd,gt).item(),
            ssim=ssim(rd,gt,data_range=1.0).mean().item(),
            lpips=lpips_fn(rd,gt).mean().item()
        ))

    avg = {k: sum(m[k] for m in metrics)/N for k in ("psnr","ssim","lpips")}

    # -------- 生成 2×N 网格（GT 在上，Render 在下） --------
    grid_tensor = torch.cat([imgs.cpu(), renders.cpu()], 0)  # (2N,3,H,W)
    grid_path_scene = os.path.join(out_dir, "grid_gt_render.png")
    grid_path_root  = os.path.join(args.out_base, f"{scene_id}_composite.png")
    save_image(grid_tensor, grid_path_scene, nrow=N, normalize=True, value_range=(0,1))
    save_image(grid_tensor, grid_path_root,  nrow=N, normalize=True, value_range=(0,1))
    print(f"[Grid] saved → {grid_path_scene}\n        saved → {grid_path_root}")

    # -------- PLY & metrics --------
    save_gaussians_to_ply(gdict, os.path.join(out_dir, "gaussians.ply"))
    with open(os.path.join(out_dir, "metrics.txt"),"w") as f:
        for m in metrics:
            f.write(f'{m["idx"]:2d} {m["name"]:<20} '
                    f'PSNR={m["psnr"]:.2f} SSIM={m["ssim"]:.4f} LPIPS={m["lpips"]:.4f}\n')
        f.write("\n—— AVERAGE ——\n")
        f.write(f'PSNR={avg["psnr"]:.2f}  SSIM={avg["ssim"]:.4f}  LPIPS={avg["lpips"]:.4f}\n')

    print(json.dumps({"scene":scene_id, **avg}, indent=2))
    print(f"[Done] all results → {out_dir}")

if __name__ == "__main__":
    main()