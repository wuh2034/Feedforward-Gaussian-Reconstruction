#!/usr/bin/env python
"""
End‑to‑end pipeline that
1. Loads images -> VGGT backbone  -> point / camera tokens
2. Runs a Gaussian Head to regress full Gaussian parameters
3. Converts outputs to a gsplat.GaussianModel and renders RGB images

* Author: ChatGPT (OpenAI)
* Requirements:
    pip install torch torchvision tqdm tensorboard Pillow gsplat
* Usage:
    python vggt_gsplat_pipeline.py --image_dir images --epochs 5
"""
import argparse
import glob
import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.heads.gaussian_head import Gaussianhead




################################################################################
# Dataset
################################################################################
class ImageDataset(Dataset):
    """Simple dataset that loads images from a folder."""

    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform or transforms.ToTensor()

        if len(self.img_paths) == 0:
            raise RuntimeError("No images found for dataset!")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        # Support the VGGT helper function directly
        if self.transform == load_and_preprocess_images:
            return self.transform([self.img_paths[idx]])
        return self.transform(img)

################################################################################
# Camera & Gaussian helpers
################################################################################

def extrinsic_to_viewmat(extri_row_3x4: torch.Tensor) -> torch.Tensor:
    """Convert 3x4 world→camera extrinsic to 4x4 OpenGL‑style view matrix."""
    assert extri_row_3x4.shape == (3, 4), "Expected (3,4) extrinsic matrix"
    bottom = torch.tensor([[0, 0, 0, 1]], dtype=extri_row_3x4.dtype, device=extri_row_3x4.device)
    viewmat = torch.cat([extri_row_3x4, bottom], dim=0)
    return viewmat  # shape (4,4)


def build_gaussians(gaussian_map: torch.Tensor, point_map: torch.Tensor) -> GaussianModel:
    """Flatten per‑pixel Gaussian fields into a gsplat.GaussianModel."""
    # gaussian_map: (B, C, H, W)
    # point_map   : (B, 3, H, W)
    B, C, H, W = gaussian_map.shape
    assert point_map.shape[:3] == (B, 3, H), "Gaussian‑map and point‑map dims disagree"

    # Flatten spatial dims
    params = gaussian_map.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)
    points = point_map.permute(0, 2, 3, 1).reshape(-1, 3)     # (N, 3)

    # ----- Channel assignment -------------------------------------------------
    xyz_offsets   = params[:, 0:3]
    scales        = params[:, 3:6]
    quats         = params[:, 6:10]
    sh_coeff      = params[:, 10:-1] if C > 11 else None
    opacity       = params[:, -1:].sigmoid()                  # Clamp to 0–1

    # Estimated mean = point + offset (unit in metres as VGGT)
    means = points + xyz_offsets

    # Create & populate GaussianModel
    gs_model = GaussianModel(max_gaussians=means.shape[0]).to(means.device)
    gs_model.update(
        xyz=means,
        scales=scales,
        quats=quats,
        features=sh_coeff,
        opacities=opacity,
    )
    return gs_model

################################################################################
# Main
################################################################################

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True  # speed on Ampere+

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print(f"Device: {device} | mixed‑precision: {dtype}")

    # 1) Discover images
    patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png"]
    img_paths = []
    for p in patterns:
        img_paths += glob.glob(os.path.join(args.image_dir, p))
    img_paths = sorted(img_paths)
    print(f"Found {len(img_paths)} image(s) in {args.image_dir}")

    # 2) Dataset & Loader
    dataset = ImageDataset(img_paths, transform=load_and_preprocess_images)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 3) VGGT backbone (frozen) & Gaussian head (trainable)
    print("Loading VGGT …")
    vggt = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    for p in vggt.parameters():
        p.requires_grad = False

    sh_degree = args.sh_degree
    gaussian_channels = 3 + 3 + 4 + 3 * (sh_degree + 1) ** 2 + 1
    g_head = Gaussianhead(
        dim_in=2 * vggt.embed_dim,
        output_dim=gaussian_channels,
        activation="exp",
        conf_activation="expp1",
    ).to(device)

    # (Optional) optimisation setup — here we demonstrate a quick over‑fit
    optim_g = torch.optim.Adam(g_head.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    writer = SummaryWriter("runs/vggt_gsplat")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for imgs in loop:
            imgs = imgs.to(device)
            if imgs.dim() == 4:  # Add sequence dim (S=1)
                imgs = imgs.unsqueeze(1)

            B, S, C, H, W = imgs.shape  # VGGT expects BSx3xHxW when aggregator called sequentially
            imgs_flat = imgs.view(B * S, C, H, W)

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                tokens, ps_idx = vggt.aggregator(imgs_flat)
                depth_map, _ = vggt.depth_head(tokens, imgs_flat, ps_idx)
                point_map, _ = vggt.point_head(tokens, imgs_flat, ps_idx)
                pose_enc = vggt.camera_head(tokens)[-1]
                extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, (H, W))  # shapes: (B*S,3,4)/(B*S,3,3)

            # Forward Gaussian head
            gaussian_map = g_head(tokens, imgs_flat, ps_idx, point_map)

            # Build gsplat model & render
            batches_rgb = []
            for i in range(B * S):
                gs_model = build_gaussians(gaussian_map[i:i+1], point_map[i:i+1])

                viewmat = extrinsic_to_viewmat(extrinsics[i]).to(device)
                K       = intrinsics[i].to(device)

                rgba, _ = rasterize(
                    gaussians=gs_model,
                    viewmats=viewmat[None],
                    Ks=K[None],
                    H=H,
                    W=W,
                    bg=torch.tensor([1, 1, 1], dtype=torch.float32, device=device),
                )
                batches_rgb.append(rgba[:, :3])  # keep RGB

            renders = torch.cat(batches_rgb, dim=0).view(B, S, 3, H, W)

            # Simple photometric loss vs input (could also compare to ground truth novel view)
            loss = criterion(renders, imgs[..., :3, :, :])
            optim_g.zero_grad()
            loss.backward()
            optim_g.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            loop.set_postfix(loss=batch_loss)

            writer.add_scalar("Loss/train_batch", batch_loss, global_step)
            global_step += 1

        avg = epoch_loss / len(dataloader)
        writer.add_scalar("Loss/train_epoch", avg, epoch)
        print(f"Epoch {epoch} • avg photometric loss: {avg:.4f}")

    writer.close()
    print("Training + rendering finished. Outputs stored in runs/vggt_gsplat")

################################################################################
# CLI
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGGT + Gaussian Splatting renderer")
    parser.add_argument("--image_dir", type=str, default="images", help="Folder of training images")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sh_degree", type=int, default=0, help="Spherical harmonic degree in Gaussian head")
    args = parser.parse_args()

    main(args)
