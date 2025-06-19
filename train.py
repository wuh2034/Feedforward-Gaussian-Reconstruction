# train.py  ────────────────────────────────────────────────────────────────────
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
from vggt.utils.renderer import render_gaussians            # gsplat 渲染

# --------------------------- dataset -----------------------------------------
class ImageDataset(Dataset):
    def __init__(self, paths, tfm=None):
        self.paths, self.tfm = paths, tfm or transforms.ToTensor()

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.tfm is load_and_preprocess_images:
            img = self.tfm([self.paths[idx]])[0]             # (3,H,W)   
        else:
            img = self.tfm(img)
        return img
# -----------------------------------------------------------------------------


# --------------------------- util --------------------------------------------
def flatten_gdict(gdict: dict, B: int):
    """
    把 (B,S,H,W,C) 统一 flatten → (B, N, C)，同时确保仍有 3 个维度
    """
    out = {}
    for k, v in gdict.items():                   # v.shape = (B, ..., C)
        C = v.shape[-1] if v.ndim >= 2 else 1
        out[k] = v.reshape(B, -1, C)             # (B,N,C)             
    return out
# -----------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if (device=="cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
print(f"Device: {device},  Mixed-Precision dtype: {dtype}")

# 1) 读图
img_dir  = "images"
patterns = ["*.jpg","*.JPG","*.jpeg","*.JPEG"]
paths = sorted(p for pat in patterns for p in glob.glob(os.path.join(img_dir, pat)))
print("Found", len(paths), "images")

dataset    = ImageDataset(paths, tfm=load_and_preprocess_images)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 2) 模型
vggt = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)
for p in vggt.parameters(): p.requires_grad = False

sh_degree  = 0
out_dim    = 3+3+4+3*(sh_degree+1)**2+1
g_head     = Gaussianhead(2*vggt.embed_dim, output_dim=out_dim,
                          activation="exp", conf_activation="expp1",
                          sh_degree=sh_degree).to(device)

opt  = torch.optim.Adam(g_head.parameters(), lr=1e-4)
crit = nn.MSELoss()

writer = SummaryWriter("runs/gauss_train")
os.makedirs("renders", exist_ok=True)

global_step = 0
for epoch in range(1, 50):
    loop, epoch_loss = tqdm(dataloader, f"Epoch {epoch}/5"), 0.0
    for imgs in loop:
        imgs = imgs.to(device)                          # (B,3,H,W)
        B,C,H,W = imgs.shape
        imgs_in = imgs.unsqueeze(1)                     # (B,1,3,H,W)      

        # ----- VGGT 前向 (冻结) -----------------------------------------------
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):       
            tok_list, ps_idx = vggt.aggregator(imgs_in)
            depth_map, _   = vggt.depth_head(tok_list, imgs_in, ps_idx)
            point_map, _   = vggt.point_head(tok_list, imgs_in, ps_idx)
            pose_enc       = vggt.camera_head(tok_list)[-1]
            extr, intr     = pose_encoding_to_extri_intri(pose_enc, (H, W))

        # ----- Gaussian Head -------------------------------------------------
        gdict_raw = g_head(tok_list, imgs_in, ps_idx, point_map)             
        gdict = flatten_gdict(gdict_raw, B)                                  
        print("α:", gdict["opacities"].mean().item(),
        "σ:", gdict["scales"].mean().item())

        # ----- 可微渲染 -------------------------------------------------------
        renders = render_gaussians(gdict, intr, extr, H, W)                  # (B,3,H,W)

        # ----- 损失 & BP ------------------------------------------------------
        with torch.amp.autocast("cuda", dtype=dtype):
            loss = crit(renders, imgs)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # ----- 日志 -----------------------------------------------------------
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        writer.add_scalar("loss/batch", loss.item(), global_step)

        if global_step % 5 == 0:
            save_image(torch.cat([imgs, renders], 0),
                       f"renders/ep{epoch:02d}_it{global_step:06d}.png",
                       normalize=True, value_range=(0,1))
        global_step += 1

    writer.add_scalar("loss/epoch", epoch_loss/len(dataloader), epoch)
writer.close()
