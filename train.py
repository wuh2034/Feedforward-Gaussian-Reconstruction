import os, glob, torch, time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.heads.gaussian_head import Gaussianhead
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.renderer import render_gaussians  # differentiable rendering

# Perceptual and SSIM losses
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

# Dataset
class ImageDataset(Dataset):
    def __init__(self, paths, tfm=None):  # paths: list of image file paths; tfm: preprocessing transform or None
        self.paths, self.tfm = paths, tfm or transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.tfm is load_and_preprocess_images:
            img = self.tfm([self.paths[idx]])[0]  # returns tensor shape (3, H, W)
        else:
            img = self.tfm(img)
        return img

# Utilities
def flatten_gdict(gdict: dict, B: int):
    """
    Flatten each value in gdict from shape (B, ..., C) to (B, N, C) for batch processing,
    preserving dimensions: batch, flattened elements, and channels.
    """
    out = {}
    for k, v in gdict.items():                   # v.shape = (B, ..., C)
        C = v.shape[-1] if v.ndim >= 2 else 1    # determine channel count C
        out[k] = v.reshape(B, -1, C)             # reshape to (B, N, C)
    return out

# Select device and precision
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if (device=="cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
print(f"Device: {device},  Mixed-Precision dtype: {dtype}")

# 1) Load images
img_dir  = "large_images"
patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
paths = sorted(p for pat in patterns for p in glob.glob(os.path.join(img_dir, pat)))
print("Found", len(paths), "images")

dataset    = ImageDataset(paths, tfm=load_and_preprocess_images)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 2) Model setup
vggt = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)
for p in vggt.parameters():
    p.requires_grad = False

sh_degree = 0
# sh_degree = 2  # enable higher-order spherical harmonics
out_dim = 3 + 3 + 4 + 3 * (sh_degree + 1) ** 2 + 1
g_head = Gaussianhead(
    2 * vggt.embed_dim,
    output_dim=out_dim,
    activation="exp",
    conf_activation="expp1",
    sh_degree=sh_degree
).to(device)

opt = torch.optim.Adam(g_head.parameters(), lr=1e-4)
crit = nn.MSELoss()
lpips_fn = lpips.LPIPS(net='alex').to(device)
lpips_weight = 0.2
ssim_fn = SSIM(data_range=1.0).to(device)
ssim_weight = 0.2

# Create a run-specific logging folder
run_id = time.strftime("%Y%m%d-%H%M%S")
render_dir = os.path.join("renders", run_id)
writer = SummaryWriter(f"runs/gauss_train_{run_id}")
os.makedirs(render_dir, exist_ok=True)

global_step = 0
for epoch in range(1, 30000):
    loop, epoch_loss = tqdm(dataloader, f"Epoch {epoch}/30000"), 0.0
    for imgs in loop:
        imgs = imgs.to(device)                # input batch shape (B, 3, H, W)
        B, C, H, W = imgs.shape
        imgs_in = imgs.unsqueeze(1)           # reshape to (B, 1, 3, H, W)

        # VGGT forward pass (frozen)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            tok_list, ps_idx = vggt.aggregator(imgs_in)
            depth_map, _ = vggt.depth_head(tok_list, imgs_in, ps_idx)
            point_map, _ = vggt.point_head(tok_list, imgs_in, ps_idx)
            pose_enc = vggt.camera_head(tok_list)[-1]
            extr, intr = pose_encoding_to_extri_intri(pose_enc, (H, W))

        # Gaussian head predictions
        gdict_raw = g_head(tok_list, imgs_in, ps_idx, point_map)
        gdict = flatten_gdict(gdict_raw, B)
        print("α:", gdict["opacities"].mean().item(), "σ:", gdict["scales"].mean().item())

        # Differentiable rendering
        renders = render_gaussians(gdict, intr, extr, H, W)  # (B, 3, H, W)

        # Compute loss and backpropagate
        with torch.amp.autocast("cuda", dtype=dtype):
            mse_loss = crit(renders, imgs)
            # LPIPS expects inputs in [-1,1]
            renders_norm = renders * 2 - 1
            imgs_norm = imgs * 2 - 1
            lpips_loss = lpips_fn(renders_norm, imgs_norm).mean()
            ssim_val = ssim_fn(renders, imgs)
            ssim_loss = 1 - ssim_val
            if epoch <= 0:
                loss = mse_loss
            else:
                loss = mse_loss + lpips_weight * lpips_loss + ssim_weight * ssim_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Logging
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        writer.add_scalar("loss/batch", loss.item(), global_step)
        writer.add_scalar("mse/batch", mse_loss.item(), global_step)
        writer.add_scalar("lpips/batch", lpips_loss.item(), global_step)
        writer.add_scalar("ssim/batch", ssim_loss.item(), global_step)

        if global_step % 50 == 0:
            save_image(
                torch.cat([imgs, renders], 0),
                os.path.join(render_dir, f"ep{epoch:02d}_it{global_step:06d}.png"),
                normalize=True, value_range=(0,1)
            )
        global_step += 1

    writer.add_scalar("loss/epoch", epoch_loss / len(dataloader), epoch)

writer.close()