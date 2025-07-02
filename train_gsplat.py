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
from vggt.utils.renderer_gsplat import render_gaussians            # gsplat 渲染
import lpips
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import gc; gc.collect()

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

writer_path = "runs/gauss_train_9"
img_log_path = "renders/gauss_train_9"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if (device=="cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
print(f"Device: {device},  Mixed-Precision dtype: {dtype}")

# 1) 读图
img_dir  = "images"
patterns = ["*.jpg","*.JPG","*.jpeg","*.JPEG"]
paths = sorted(p for pat in patterns for p in glob.glob(os.path.join(img_dir, pat)))
print("Found", len(paths), "images")

dataset    = ImageDataset(paths, tfm=load_and_preprocess_images)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# 2) 模型
print("loading model")
vggt = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)
for p in vggt.parameters(): p.requires_grad = False
print("model loaded")


cache = []
with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    for imgs in tqdm(dataloader, desc="Caching VGGT outputs"):
        imgs = imgs.to(device)                      # (B,3,H,W)
        B, C, H, W = imgs.shape
        imgs_in = imgs.unsqueeze(1)                 # (B,1,3,H,W)

        # --- VGGT 前向 ---
        tok_list, ps_idx  = vggt.aggregator(imgs_in)
       
        # --- VGGT 前向（与 demo_viser 接口保持一致） ---
        preds = vggt(imgs)                       # dict with "pose_enc", "world_points", etc.
        print("preds keys:", preds.keys())
        # VGGT forward returns "world_points" of shape (B,H,W,3); reshape to (B,1,H,W,3)
        point_map = preds["world_points"].unsqueeze(1)
        pose_enc  = preds["pose_enc"]            # last iteration encoding
        extr, intr = pose_encoding_to_extri_intri(pose_enc, (H, W))
        
        # --- 缓存到 CPU ---
        cache.append({
            "imgs"     : imgs.detach().cpu(),
            "tok_list" : [t.detach().cpu() for t in tok_list],
            "ps_idx"   : ps_idx,
            "point_map": point_map.detach().cpu(),
            "intr"     : intr.detach().cpu(),
            "extr"     : extr.detach().cpu(),
        })


# 保存嵌入维度后释放 VGGT
embed_dim = vggt.embed_dim
del vggt

torch.cuda.empty_cache()


print("creating gaussian head")
sh_degree  = 0
out_dim    = 3+3+4+3*(sh_degree+1)**2+1
g_head     = Gaussianhead(2*embed_dim, output_dim=out_dim,
                          activation="exp", conf_activation="expp1",
                          sh_degree=sh_degree).to(device)
print("gaussian head created")

import torch # ensure torch is imported for optimizer param grouping
# 3）定义loss与optimizer策略
warmup_epochs = 1000  # 设定前多少个epoch只用MSE做预热

opt = torch.optim.Adam(g_head.parameters(), lr=1e-4)
crit = nn.MSELoss()
lpips_fn = lpips.LPIPS(net='alex').to(device)
lpips_weight = 0.2  # LPIPS 的权重，可根据需要调节
ssim_fn = SSIM(data_range=1.0).to(device)
ssim_weight = 0.2  # SSIM 的权重，可根据需要调节


writer = SummaryWriter(writer_path)
os.makedirs(img_log_path, exist_ok=True)

global_step = 0
for epoch in range(1, 30000):
    epoch_loss = 0.0
    for entry in tqdm(cache, desc=f"Epoch {epoch}/30000"):
        imgs = entry["imgs"].to(device)               # directly retrieve tensor
        tok_list  = [t.to(device) for t in entry["tok_list"]]
        ps_idx    = entry["ps_idx"]
        point_map = entry["point_map"].to(device)
        intr      = entry["intr"].to(device) #(1,3,3,3)
        extr      = entry["extr"].to(device)
        B, C, H, W = imgs.shape
        imgs_in   = imgs.unsqueeze(1)     

        # ---- 组织为  B=1, C=B  -------------------------------------------
        # extr : (1, B, 3, 4)  → pad → (1, B, 4, 4)
        pad_row = torch.tensor([0, 0, 0, 1],
                               device=extr.device,
                               dtype=torch.float32).view(1, 1, 1, 4).expand(extr.shape[0], extr.shape[1], -1, -1)
        extr = torch.cat([extr.float(), pad_row], dim=-2)          # (1, B, 4, 4)

        # intr : (1, B, 3, 3)  保持不变，只需转 fp32
        intr = intr.float()
        
        # ----- Gaussian Head -------------------------------------------------
        gdict_raw = g_head(tok_list, imgs_in, ps_idx, point_map)
        gdict = flatten_gdict(gdict_raw, B)
        # 把 batch 维合并，使 gdict 形状变成 (1, N_tot, C) ——与 (B=1, C=B) 对齐
        for k, v in gdict.items():
            if v.ndim == 3:
                gdict[k] = v.reshape(1, -1, v.shape[-1])
            else:                  # opacities (...,1) 等情况
                gdict[k] = v.reshape(1, -1)
        # print("α:", gdict["opacities"].mean().item(),
        # "σ:", gdict["scales"].mean().item())

        # ----- 可微渲染 -------------------------------------------------------
        renders = render_gaussians(gdict, intr, extr, H, W)                  # (B,3,H,W)

        # # Diagnostic: per-channel MSE before backward
        # mse_per_ch = ((renders - imgs) ** 2).mean(dim=[0,2,3])
        # print(f"MSE per channel: {mse_per_ch.tolist()}")

        # ----- 损失 & BP ------------------------------------------------------
        with torch.amp.autocast("cuda", dtype=dtype):
            # apply channel weights to MSE
            weights = torch.tensor([1.0, 1.0, 1.0], device=device).view(1,3,1,1)
            mse_loss = ((weights * (renders - imgs)**2).mean())
            # 计算LPIPS, LPIPS 需要 [-1,1] 范围
            renders_norm = renders * 2 - 1
            imgs_norm = imgs * 2 - 1
            lpips_loss = lpips_fn(renders_norm, imgs_norm).mean()

            # 计算SSIM
            ssim_val = ssim_fn(renders, imgs)
            ssim_loss = 1 - ssim_val

            # 计算总loss
            if epoch <= warmup_epochs:
                loss = mse_loss
            else:
                loss = mse_loss + lpips_weight * lpips_loss + ssim_weight * ssim_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # # Diagnostic: gradient of color bias in output_conv2
        # color_end = color_start + 3
        # grads = g_head.scratch.output_conv2[-1].bias.grad
        # print(f"Bias grads for color channels: {grads[color_start:color_end].tolist()}")
        # gradient clip
        # torch.nn.utils.clip_grad_norm_(g_head.parameters(), max_norm=1.0)
        opt.step()
        # # clamp green channel bias to [-1,1]
        # g_bias = g_head.scratch.output_conv2[-1].bias
        # with torch.no_grad():
        #     g_bias[color_start+1].clamp_(-1.0, +1.0)

        # for name, param in g_head.named_parameters():
        #     if param.grad is None:
        #         print(f"[No grad] {name}")
        #     else:
        #         print(f"{name}: grad norm = {param.grad.norm().item():.4e}")

        # ----- 日志 -----------------------------------------------------------
        epoch_loss += loss.item()
        writer.add_scalar("loss/batch", loss.item(), global_step)
        writer.add_scalar("mse/batch", mse_loss.item(), global_step)
        writer.add_scalar("lpips/batch", lpips_loss.item(), global_step)
        writer.add_scalar("ssim/batch", ssim_loss.item(), global_step)

        if global_step % 100 == 0:
            output_path = os.path.join(
                        img_log_path,
                        f"ep{epoch:02d}_it{global_step:06d}.png"
                    )
            save_image(torch.cat([imgs, renders], 0),
                       output_path,
                       normalize=True, value_range=(0,1))
        global_step += 1

    writer.add_scalar("loss/epoch", epoch_loss/len(cache), epoch)
writer.close()