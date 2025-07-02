import os, glob, torch, time
import gc
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
from vggt.utils.renderer_diff_gaussian import render_gaussians  # differentiable rendering


from einops import rearrange

# Perceptual and SSIM losses
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

# -----------------------------------------------------------------------------
# 1) Dataset & DataLoader
# -----------------------------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, paths, tfm=None):
        self.paths, self.tfm = paths, tfm or transforms.ToTensor()
    def __len__(self): return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.tfm is load_and_preprocess_images:
            img = self.tfm([self.paths[idx]])[0]
        else:
            img = self.tfm(img)
        return img

def flatten_gdict(gdict: dict, B: int):
    out = {}
    for k, v in gdict.items():
        # Determine channel count
        C = v.shape[-1] if v.ndim >= 2 else 1
        # 如果传入sh_degree 就要使用这里
        if k == 'sh':
            # For spherical harmonics: reshape to (B, N, 1, C)
            out[k] = v.reshape(B, -1, 1, C)
        else:
            # For all other keys: reshape to (B, N, C)
            out[k] = v.reshape(B, -1, C)
        # out[k] = v.reshape(B, -1, C)
    return out



device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if (device=="cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
print(f"Device: {device},  dtype: {dtype}")

# load image paths
# img_dir  = "large_images"
img_dir  = "images"
patterns = ["*.jpg","*.JPG","*.jpeg","*.JPEG"]
paths = sorted(p for pat in patterns for p in glob.glob(os.path.join(img_dir, pat)))
print("Found", len(paths), "images")

dataset    = ImageDataset(paths, tfm=load_and_preprocess_images)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)  # no shuffle for caching

# -----------------------------------------------------------------------------
# 2) 初始化 VGGT 并缓存特征
# -----------------------------------------------------------------------------
vggt = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)
for p in vggt.parameters(): p.requires_grad = False

cache = []  # 每个元素是 dict: { 'imgs', 'tok_list', 'ps_idx', 'point_map', 'intr', 'extr' }
with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    for imgs in tqdm(dataloader, desc="Caching VGGT outputs"):
        imgs = imgs.to(device)              # (B,3,H,W)
        B,C,H,W = imgs.shape
        imgs_in  = imgs.unsqueeze(1)        # (B,1,3,H,W)
        tok_list, ps_idx = vggt.aggregator(imgs_in)
        _, _          = vggt.depth_head(tok_list, imgs_in, ps_idx)  # 可以丢掉深度输出
        point_map, _  = vggt.point_head(tok_list, imgs_in, ps_idx)
        pose_enc      = vggt.camera_head(tok_list)[-1]
        extr, intr    = pose_encoding_to_extri_intri(pose_enc, (H, W))
        cache.append({
            'imgs':      imgs,
            'tok_list':  tok_list,
            'ps_idx':    ps_idx,
            'point_map': point_map,
            'intr':      intr,
            'extr':      extr,
        })

    # Save the embedding dimension before deleting the VGGT object
    embed_dim = vggt.embed_dim
# ---------- 释放显存 ----------
for entry in cache:
    for k, v in entry.items():
        if torch.is_tensor(v):
            entry[k] = v.detach().cpu()
        elif isinstance(v, list):
            entry[k] = [t.detach().cpu() if torch.is_tensor(t) else t for t in v]
del vggt                      # 释放 VGGT 模型占用的显存
torch.cuda.empty_cache()
gc.collect()
# ---------- 释放显存 ----------

# -----------------------------------------------------------------------------
# 3) 构造 Gaussian Head & 训练循环
# -----------------------------------------------------------------------------
sh_degree = 0
out_dim   = 3 + 3 + 4 + 3*(sh_degree+1)**2 + 1
g_head = Gaussianhead(
    2*embed_dim,
    output_dim=out_dim,
    activation="exp",
    conf_activation="expp1",
    sh_degree=sh_degree
).to(device)

opt       = torch.optim.Adam(g_head.parameters(), lr=1e-4)
crit      = nn.MSELoss()
lpips_fn  = lpips.LPIPS(net='alex').to(device)
ssim_fn   = SSIM(data_range=1.0).to(device)
lpips_w, ssim_w = 0.2, 0.2

run_id    = time.strftime("%Y%m%d-%H%M%S")
render_dir= os.path.join("renders", run_id)
writer    = SummaryWriter(f"runs/gauss_train_{run_id}")
os.makedirs(render_dir, exist_ok=True)

global_step = 0
for epoch in range(1, 30000):
    epoch_loss = 0.0
    for item in tqdm(cache, desc=f"Epoch {epoch}"):
        imgs      = item['imgs'].to(device)
        tok_list  = [t.to(device) for t in item['tok_list']]
        ps_idx    = item['ps_idx']
        # ps_idx = torch.tensor(item['ps_idx'], device=device)
        point_map = item['point_map'].to(device)
        intr      = item['intr'].to(device)
        extr      = item['extr'].to(device)
        B,_,H,W   = imgs.shape

        # Gaussian head 前向
        gdict_raw = g_head(tok_list, imgs.unsqueeze(1), ps_idx, point_map)
        gdict     = flatten_gdict(gdict_raw, B)
        #         # ========= 方案 B：在线随机 drop 高斯 =========
        # drop_ratio = 0.5                                    # 想保留多少比例就调这里
        # N_total     = gdict["opacities"].shape[1]           # 每张图当前的高斯数
        # keep_mask   = torch.rand(
        #     N_total, device=gdict["opacities"].device
        # ) >= drop_ratio                                     # True 表示保留

        # # 对 gdict 里所有键统一子采样：
        # # - 非 SH 通道形状 (B, N, C)           → gdict[k][:, keep_mask,  ...]
        # # - SH  通道形状 (B, N, K, 3)          → gdict[k][:, keep_mask, ...]
        # for k in gdict:
        #     gdict[k] = gdict[k][:, keep_mask, ...]
        # # =============================================
        # renders   = render_gaussians(gdict, sh_degree,intr, extr, H, W)

        # # 损失
        # mse_loss   = crit(renders, imgs)
        # with torch.amp.autocast("cuda", dtype=dtype):
        #     renders_n  = renders*2-1
        #     imgs_n = imgs*2-1
        #     lpips_loss = lpips_fn(renders_n, imgs_n).mean()
        #     ssim_val   = ssim_fn(renders, imgs); ssim_loss = 1-ssim_val
        #     loss = mse_loss + lpips_w*lpips_loss + ssim_w*ssim_loss

        # opt.zero_grad(set_to_none=True)
        # with torch.autograd.detect_anomaly():
        #     loss.backward()
        # opt.step()
        
        opt.zero_grad(set_to_none=True)
        with torch.autograd.detect_anomaly():                # ⬅️ 前向也包进来
            renders = render_gaussians(gdict, sh_degree, intr, extr, H, W)

            # ---------- 损失 ----------
            mse_loss = crit(renders, imgs)
            with torch.amp.autocast("cuda", dtype=dtype):
                renders_n  = renders * 2 - 1
                imgs_n     = imgs * 2 - 1
                lpips_loss = lpips_fn(renders_n, imgs_n).mean()
                ssim_val   = ssim_fn(renders, imgs)
                ssim_loss  = 1 - ssim_val
                loss = mse_loss + lpips_w * lpips_loss + ssim_w * ssim_loss

            loss.backward()

        # --------- 新增梯度裁剪 ----------
        torch.nn.utils.clip_grad_norm_(g_head.parameters(), 1.0)
        opt.step()

        epoch_loss += loss.item()
        writer.add_scalar("loss/batch",   loss.item(),   global_step)
        writer.add_scalar("mse/batch",    mse_loss.item(),global_step)
        writer.add_scalar("lpips/batch",  lpips_loss.item(),global_step)
        writer.add_scalar("ssim/batch",   ssim_loss.item(),global_step)

        if global_step % 300 == 0:
            save_image(
                torch.cat([imgs, renders], 0),
                os.path.join(render_dir, f"ep{epoch:02d}_it{global_step:06d}.png"),
                normalize=True, value_range=(0,1)
            )
        global_step += 1

    writer.add_scalar("loss/epoch", epoch_loss/len(cache), epoch)

writer.close()