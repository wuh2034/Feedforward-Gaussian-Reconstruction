import os
import gc
import random
import torch
import torch.nn as nn
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

# Configuration
ROOT_DIR = "/usr/prakt/s0012/scannetpp/data"
TRAIN_SPLIT = "/usr/prakt/s0012/scannetpp/splits/nvs_sem_train_800.txt"
VAL_SPLIT = "/usr/prakt/s0012/scannetpp/splits/nvs_sem_val.txt"

SCENES_PER_BATCH = 80  # Number of scenes per training batch
IMG_NUM_MAIN = 10      # Number of main images per scene
IMG_NUM_AUX = 2        # Number of auxiliary images per scene
STRIDE = 5             # Sampling stride for image selection

VAL_EVERY_BATCH = True    # Validate after every training batch
QUICK_VAL_SCENES = 2      # Number of scenes for quick validation
QUICK_VAL_IMGS = 2        # Number of images per scene for quick validation

VAL_SCENE_LIMIT = 5       # Maximum number of scenes for full validation
VAL_IMG_NUM_MAIN = IMG_NUM_MAIN  # Number of main images for full validation
VAL_IMG_NUM_AUX = IMG_NUM_AUX    # Number of auxiliary images for full validation

ENABLE_AUX = False    # Whether to use auxiliary images
NUM_EPOCHS = 3001     # Total number of training epochs
NUM_WORKERS = 4       # Number of data loading workers
VIS_EPOCH = 1         # Epoch interval for generating visual outputs
CKPT_INTERVAL = 5     # Epoch interval for saving checkpoints
WARMUP_EPOCHS = 10    # Number of warm-up epochs using only MSE loss

LOG_DIR = "runs/withoutAUX_50_10"
CKPT_DIR = "checkpoints/withoutAUX_50_10"
IMG_DIR = "renders/withoutAUX_50_10"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Loss weights: MSE, LPIPS, SSIM
L_MSE, L_LP, L_SS = 1.0, 0.2, 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
print(f"[Init] device={device}, dtype={dtype}")

# Utility to reshape Gaussian dictionary outputs
def flatten_gdict(gdict: dict, batch_size: int):
    return {
        key: value.reshape(batch_size, -1, value.shape[-1] if value.ndim >= 2 else 1)
        for key, value in gdict.items()
    }

# Build a cache of features and poses for a data loader
@torch.no_grad()
def build_cache(loader, vggt_model, use_aux: bool, description: str):
    cache = []
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        for batch in tqdm(loader, desc=description, leave=False):
            if use_aux:
                main_imgs, aux_imgs, *_ = batch
                combined = torch.cat([main_imgs, aux_imgs], dim=0).to(device)
                num_main, _, height, width = main_imgs.shape
                token_list, pose_indices = vggt_model.aggregator(combined.unsqueeze(1))
                predictions = vggt_model(combined)
                point_map = predictions["world_points"].unsqueeze(1)
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], (height, width))
                cache.append({
                    'main_images': main_imgs.cpu(),
                    'aux_images': aux_imgs.cpu(),
                    'token_list': [t[:num_main].cpu() for t in token_list],
                    'pose_indices': pose_indices,
                    'point_map': point_map[:, :, :num_main].cpu(),
                    'intrinsics_main': intrinsics[:, :num_main].cpu(),
                    'extrinsics_main': extrinsics[:, :num_main].cpu(),
                    'intrinsics_aux': intrinsics[:, num_main:].cpu(),
                    'extrinsics_aux': extrinsics[:, num_main:].cpu(),
                })
            else:
                images, *_ = batch
                images = images.to(device)
                batch_size, _, height, width = images.shape
                token_list, pose_indices = vggt_model.aggregator(images.unsqueeze(1))
                predictions = vggt_model(images)
                point_map = predictions["world_points"].unsqueeze(1)
                extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], (height, width))
                cache.append({
                    'images': images.cpu(),
                    'token_list': [t.cpu() for t in token_list],
                    'pose_indices': pose_indices,
                    'point_map': point_map.cpu(),
                    'intrinsics': intrinsics.cpu(),
                    'extrinsics': extrinsics.cpu(),
                })
    return cache

# Initialize Gaussian head and optimizer
print("[Init] Gaussian Head initialization...")
EMBED_DIM = VGGT.from_pretrained("facebook/VGGT-1B").embed_dim
gaussian_head = Gaussianhead(
    input_dim=2 * EMBED_DIM,
    output_dim=3 + 3 + 4 + 3 * (0 + 1) ** 2 + 1,
    activation="exp",
    confidence_activation="expp1",
    spherical_harmonics_degree=0,
).to(device)
optimizer = torch.optim.Adam(gaussian_head.parameters(), lr=1e-4)

# Loss functions
mse_loss = nn.MSELoss()
lpips_loss = lpips.LPIPS(net="alex").to(device)
psnr_metric = PSNR(data_range=1.0).to(device)

# TensorBoard writer
writer = SummaryWriter(LOG_DIR)
step = 0

# Evaluation helper: compute validation metrics and optional visualization
@torch.no_grad()
def evaluate_cache(cache, use_aux=False, return_visual=False):
    total_mse = total_psnr = total_ssim = total_lpips = 0.0
    first_ground_truth = first_render = None
    for entry in cache:
        if use_aux and 'aux_images' in entry:
            main = entry['main_images'].to(device)
            aux = entry['aux_images'].to(device)
            batch_size, _, h, w = main.shape
            tokens = [t.to(device) for t in entry['token_list']]

            extr_main = torch.cat([
                entry['extrinsics_main'].to(device).float(),
                torch.tensor([0,0,0,1], device=device).view(1,1,1,4).expand(1, batch_size, 1, 4)
            ], dim=-2)
            intr_main = entry['intrinsics_main'].to(device).float()
            gaussian_output = gaussian_head(tokens, main.unsqueeze(1), entry['pose_indices'], entry['point_map'].to(device))
            gaussian_flat = {k: v.view(1, -1, v.shape[-1]) if v.ndim == 3 else v.view(1, -1) for k, v in flatten_gdict(gaussian_output, batch_size).items()}
            render_main = render_gaussians(gaussian_flat, intr_main, extr_main, h, w)

            batch_aux = aux.shape[0]
            extr_aux = torch.cat([
                entry['extrinsics_aux'].to(device).float(),
                torch.tensor([0,0,0,1], device=device).view(1,1,1,4).expand(1, batch_aux, 1, 4)
            ], dim=-2)
            render_aux = render_gaussians(gaussian_flat, entry['intrinsics_aux'].to(device).float(), extr_aux, h, w)

            rendered = torch.cat([render_main, render_aux], dim=0)
            ground_truth = torch.cat([main, aux], dim=0)
            total_mse += mse_loss(render_main, main).item() + 0.5 * mse_loss(render_aux, aux).item()
        else:
            images = entry['images'].to(device)
            batch_size, _, h, w = images.shape
            tokens = [t.to(device) for t in entry['token_list']]
            extr = torch.cat([
                entry['extrinsics'].to(device).float(),
                torch.tensor([0,0,0,1], device=device).view(1,1,1,4).expand(1, batch_size, 1, 4)
            ], dim=-2)
            intr = entry['intrinsics'].to(device).float()
            gaussian_output = gaussian_head(tokens, images.unsqueeze(1), entry['pose_indices'], entry['point_map'].to(device))
            gaussian_flat = {k: v.view(1, -1, v.shape[-1]) if v.ndim == 3 else v.view(1, -1) for k, v in flatten_gdict(gaussian_output, batch_size).items()}
            rendered = render_gaussians(gaussian_flat, intr, extr, h, w)
            ground_truth = images
            total_mse += mse_loss(rendered, images).item()

        total_psnr += psnr_metric(rendered, ground_truth).item()
        total_ssim += ssim(rendered.cpu(), ground_truth.cpu(), data_range=1.0).mean().item()
        total_lpips += lpips_loss(rendered, ground_truth).mean().item()

        if return_visual and first_ground_truth is None:
            first_ground_truth = ground_truth.cpu()
            first_render = rendered.cpu()

    num_entries = len(cache)
    avg_metrics = (total_mse / num_entries, total_psnr / num_entries, total_ssim / num_entries, total_lpips / num_entries)
    if return_visual:
        return (*avg_metrics, first_ground_truth, first_render)
    return avg_metrics

# Main training loop
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n===== Epoch {epoch} =====")

    # Adjust loss weights after warm-up period
    lp_weight = 0.0 if epoch <= WARMUP_EPOCHS else L_LP
    ss_weight = 0.0 if epoch <= WARMUP_EPOCHS else L_SS

    # Load and shuffle training scene IDs
    with open(TRAIN_SPLIT) as f:
        scene_ids = [line.strip() for line in f if line.strip()]
    random.shuffle(scene_ids)
    total_batches = len(scene_ids) // SCENES_PER_BATCH

    # Prepare frozen feature extractor
    feature_extractor = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)
    for param in feature_extractor.parameters():
        param.requires_grad_(False)

    # Prepare quick validation if enabled
    if VAL_EVERY_BATCH:
        with open(VAL_SPLIT) as f:
            quick_ids = [line.strip() for idx, line in enumerate(f) if idx < QUICK_VAL_SCENES]
        _, quick_val_loader = build_train_val(
            ROOT_DIR, TRAIN_SPLIT, VAL_SPLIT,
            None, quick_ids,
            IMG_NUM_MAIN, QUICK_VAL_IMGS,
            stride=STRIDE, num_workers=NUM_WORKERS
        )

    save_visual = (epoch % VIS_EPOCH == 0)
    train_vis_gt = []
    train_vis_rd = []
    val_vis_gt = val_vis_rd = None

    for batch_idx in range(total_batches):
        batch_ids = scene_ids[batch_idx*SCENES_PER_BATCH:(batch_idx+1)*SCENES_PER_BATCH]
        train_loader, _ = build_train_val(
            ROOT_DIR, TRAIN_SPLIT, VAL_SPLIT,
            batch_ids, None,
            IMG_NUM_MAIN, IMG_NUM_MAIN,
            stride=STRIDE, num_workers=NUM_WORKERS
        )

        cache = build_cache(train_loader, feature_extractor, ENABLE_AUX, f"Batch {batch_idx+1}/{total_batches}")
        optimizer.zero_grad(set_to_none=True)

        sum_loss = sum_psnr = sum_ssim = sum_lpips = 0.0

        for entry in cache:
            # Compute rendering and loss similar to validation logic...
            # Code omitted for brevity
            pass

        optimizer.step()
        # Logging to TensorBoard, validation, and step increment omitted for brevity

    # Save visual outputs and checkpoints as configured
print("Training complete.")