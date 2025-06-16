# Copyright © 2025.  Released under MIT.
# Minimal end-to-end VGGT × 3-DGS 训练脚本，无真相机标签

from __future__ import annotations
import argparse, pathlib
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.io as tvio

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri        # ← 官方 util
from render import VGGSplatRenderer

# -------------------------------------------------------------
class ImageFolderDataset(Dataset):
    """只读图像序列，用 PNG/JPG 排好序即可。"""
    def __init__(self, root: str, img_size: int = 518):
        self.paths = sorted(pathlib.Path(root).glob("*.png")) + \
                     sorted(pathlib.Path(root).glob("*.jpg"))
        self.H = self.W = img_size

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = tvio.read_image(str(self.paths[idx])).float() / 255.0   # C,H,W
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(self.H, self.W),
                                              mode="bilinear", align_corners=False)[0]
        return img


# -------------------------------------------------------------
def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- Data ----------
    ds = ImageFolderDataset(cfg.data, cfg.img)
    dl = DataLoader(ds, batch_size=cfg.batch, shuffle=True, num_workers=4)

    # ---------- Model & Renderer ----------
    net       = VGGT(img_size=cfg.img).to(device)
    renderer  = VGGSplatRenderer(cfg.img, cfg.img, sh_degree=0).to(device)
    optim     = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    step = 0
    for epoch in range(cfg.epoch):
        for imgs in dl:                        # imgs: [B,3,H,W]
            step += 1
            imgs = imgs.to(device).unsqueeze(1)  # → [B,1,3,H,W]   (S=1)

            # ---------------- 前向 ----------------
            preds = net(imgs)
            pose_enc = preds["pose_enc"].unsqueeze(1)  # [B,1,9] → 保持维度
            extri, intri = pose_encoding_to_extri_intri(
                pose_enc, (cfg.img, cfg.img)
            )                                           # 各 [B,1,...]

            # 展平批次与帧维
            view = torch.cat([
                torch.nn.functional.pad(extri[i,0], (0,1))    # → 4×4
                for i in range(extri.shape[0])
            ]).view(extri.shape[0], 4, 4).to(device)

            Ks   = intri[:,0].to(device)                      # [B,3,3]

            # 处理高斯
            g = preds["gaussian"]           # dict, 每个 [B,1,H,W,*]
            B, S, H, W, _ = g["means"].shape
            def flat(x): return x.view(B*S*H*W, -1)
            gauss_flat = {k: flat(v) for k, v in g.items()}

            rgb_pred, _ = renderer(gauss_flat, view, Ks)      # (B,3,H,W)
            loss = renderer.loss(rgb_pred, imgs[:,0], gauss_flat)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            if step % 25 == 0:
                print(f"step {step:06d} | loss {loss.item():.4f} | N {gauss_flat['means'].shape[0]}")

            if step % cfg.densify == 0: renderer.densify(gauss_flat)
            if step % cfg.prune   == 0: renderer.prune(gauss_flat)

        # --- checkpoint each epoch ---
        (pathlib.Path(cfg.ckpt)/"").mkdir(parents=True, exist_ok=True)
        torch.save({"net": net.state_dict(), "optim": optim.state_dict()},
                   pathlib.Path(cfg.ckpt)/f"vggsplat_e{epoch:03d}.pt")

# -------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True, help="包含 *.png/jpg 的文件夹")
    p.add_argument("--ckpt",   default="ckpt")
    p.add_argument("--img",    type=int, default=518)
    p.add_argument("--batch",  type=int, default=1)
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--epoch",  type=int, default=1)
    p.add_argument("--densify",type=int, default=200)
    p.add_argument("--prune",  type=int, default=500)
    cfg = p.parse_args()
    train(cfg)
