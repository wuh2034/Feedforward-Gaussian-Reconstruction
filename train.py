import os
import gc
import torch
import torch.nn as nn
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

# Hyperparameters and paths
ROOT_DIR = "/usr/prakt/s0012/scannetpp/data"
TRAIN_SPLIT_TXT = "/usr/prakt/s0012/scannetpp/splits/nvs_sem_train.txt"
VAL_SPLIT_TXT = "/usr/prakt/s0012/scannetpp/splits/nvs_sem_val.txt"

TRAIN_BATCH_SCENES = 25      # Number of scenes per training batch
VAL_BATCH_SCENES = 2         # Number of scenes per validation batch
IMG_NUM_TRAIN = 6            # Main views per scene during training
IMG_NUM_AUX = 2              # Auxiliary views per scene during training
AUX_WEIGHT = 0.5             # Weight for auxiliary view losses
IMG_NUM_VAL = 2              # Views per scene during validation
STRIDE = 5                   # Sampling stride for images within a scene
NUM_WORKERS = 4              # DataLoader worker threads
NUM_EPOCHS = 3001            # Total number of training epochs

LOG_DIR = "runs/scannetpp25_12_4metric"
IMG_LOG_DIR = "renders/scannetpp25_12_4metric"
CKPT_DIR = "checkpoints/scannetpp25_12_4metric"
CKPT_INTERVAL = 100         # Epoch interval for saving checkpoints

# Optional checkpoint to resume training
RESUME_CKPT = "/usr/prakt/s0012/ADL4VC/checkpoints/scannetpp25_12_4metric/epoch_1000.pth"

os.makedirs(IMG_LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# Loss weights: MSE, LPIPS, SSIM (SSIM loss = 1 - SSIM)
L_MSE, L_LP, L_SS = 1.0, 0.2, 0.1

# Device and mixed precision setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
print(f"[Init] device={device}, dtype={dtype}")

# Utility function: flatten Gaussian dictionary outputs per batch
def flatten_gdict(gdict: dict, batch_size: int):
    flattened = {}
    for key, value in gdict.items():
        channels = value.shape[-1] if value.ndim >= 2 else 1
        flattened[key] = value.reshape(batch_size, -1, channels)
    return flattened

# Build cache for datasets without auxiliary views
def build_cache(loader, vggt_model):
    cache = []
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        for imgs, scene_id, img_names in loader:
            imgs = imgs.to(device)
            B, _, H, W = imgs.shape
            tokens, pose_idx = vggt_model.aggregator(imgs.unsqueeze(1))
            predictions = vggt_model(imgs)
            point_map = predictions["world_points"].unsqueeze(1)
            extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))

            cache.append({
                'images': imgs.cpu(),
                'tokens': [t.cpu() for t in tokens],
                'pose_idx': pose_idx,
                'point_map': point_map.cpu(),
                'intrinsics': intrinsics.cpu(),
                'extrinsics': extrinsics.cpu(),
            })
    return cache

# Build cache for datasets with both main and auxiliary views
def build_cache_with_aux(loader, vggt_model):
    cache = []
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=dtype):
        for main_imgs, aux_imgs, scene_id, names_main, names_aux in loader:
            all_imgs = torch.cat([main_imgs, aux_imgs], dim=0).to(device)
            N_main, _, H, W = main_imgs.shape
            tokens_all, pose_idx = vggt_model.aggregator(all_imgs.unsqueeze(1))
            predictions = vggt_model(all_imgs)
            point_map_all = predictions["world_points"].unsqueeze(1)
            extrinsics_all, intrinsics_all = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))

            cache.append({
                'scene_id': scene_id,
                'main_images': main_imgs.cpu(),
                'aux_images': aux_imgs.cpu(),
                'tokens': [t[:N_main].cpu() for t in tokens_all],
                'pose_idx': pose_idx,
                'point_map': point_map_all[:, :, :N_main].cpu(),
                'intrinsics_main': intrinsics_all[:, :N_main].cpu(),
                'extrinsics_main': extrinsics_all[:, :N_main].cpu(),
                'intrinsics_aux': intrinsics_all[:, N_main:].cpu(),
                'extrinsics_aux': extrinsics_all[:, N_main:].cpu(),
            })
    return cache

print("[Init] Initializing Gaussian head...")

# Initialize Gaussian head and optimizer
embed_model = VGGT.from_pretrained("facebook/VGGT-1B")
EMB_DIM = embed_model.embed_dim
embed_model = None

gaussian_head = Gaussianhead(
    2 * EMB_DIM,
    3 + 3 + 4 + 3 * (0 + 1)**2 + 1,
    activation="exp",
    conf_activation="expp1",
    sh_degree=0
).to(device)
optimizer = torch.optim.Adam(gaussian_head.parameters(), lr=1e-4)

# Loss functions and metrics
mse_loss = nn.MSELoss()
lpips_loss = lpips.LPIPS(net="alex").to(device)
psnr_metric = PSNR(data_range=1.0).to(device)

# TensorBoard setup and resume checkpoints
writer = SummaryWriter(LOG_DIR)
best_val_loss = float('inf')
global_step = 0
start_epoch = 1

if RESUME_CKPT and os.path.isfile(RESUME_CKPT):
    checkpoint = torch.load(RESUME_CKPT, map_location=device)
    gaussian_head.load_state_dict(checkpoint.get('model_state_dict', checkpoint.get('model')))
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
    global_step = checkpoint.get('global_step', global_step)
    print(f"[Resume] Loaded checkpoint from epoch {start_epoch-1}")

# Freeze VGGT backbone parameters
backbone = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)
for param in backbone.parameters():
    param.requires_grad_(False)

# Main training and validation loop
for epoch in range(start_epoch, NUM_EPOCHS + 1):
    print(f"\n===== Epoch {epoch} =====")

    # Prepare DataLoaders for this epoch
    train_loader, val_loader = build_train_val(
        root_dir=ROOT_DIR,
        train_split_txt=TRAIN_SPLIT_TXT,
        val_split_txt=VAL_SPLIT_TXT,
        train_batch_scenes=TRAIN_BATCH_SCENES,
        train_img_num=IMG_NUM_TRAIN,
        train_img_aux=IMG_NUM_AUX,
        val_batch_scenes=VAL_BATCH_SCENES,
        val_img_num=IMG_NUM_VAL,
        stride=STRIDE,
        num_workers=NUM_WORKERS
    )

    # Construct forward pass caches
    train_cache = build_cache_with_aux(train_loader, backbone)
    val_cache = build_cache(val_loader, backbone)

    # Training phase
    gaussian_head.train()
    train_loss_accum = 0.0
    viz_gt_train, viz_rd_train = [], []

    for entry in tqdm(train_cache, desc="Training"):
        main = entry['main_images'].to(device)
        aux = entry['aux_images'].to(device)
        ...  # Training code continues
    
print("Training complete.")