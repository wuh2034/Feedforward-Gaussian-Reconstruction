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
        C = v.shape[-1] if v.ndim >= 2 else 1
        out[k] = v.reshape(B, -1, C)
        # out[k] = v.reshape(-1, C) #展开成(N, C)
    return out

# @MODIFIED
def reg_dense_sh(sh):
    """
    Apply PixelSplat's spherical harmonic postprocessing
    """
    # sh = rearrange(sh, '... (xyz d_sh) -> ... xyz d_sh', xyz=3) #重排sh张量,从(..., RGB * d_sh) -> (..., RGB, d_sh)
    sh = rearrange(sh, '... (xyz d_sh) -> ... d_sh xyz', xyz=3)#重排成(..., d_sh, RGB)
    
    return sh

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if (device=="cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
print(f"Device: {device},  dtype: {dtype}")

# load image paths
img_dir  = "large_images"
patterns = ["*.jpg","*.JPG","*.jpeg","*.JPEG"]
paths = sorted(p for pat in patterns for p in glob.glob(os.path.join(img_dir, pat)))
print("Found", len(paths), "images")

dataset    = ImageDataset(paths, tfm=load_and_preprocess_images)
loader     = DataLoader(dataset, batch_size=8, shuffle=False)  # no shuffle for caching

# -----------------------------------------------------------------------------
# 2) 初始化 VGGT 并缓存特征
# -----------------------------------------------------------------------------
vggt = VGGT.from_pretrained("facebook/VGGT-1B").eval().to(device)
for p in vggt.parameters(): p.requires_grad = False

cache = []  # 每个元素是 dict: { 'imgs', 'tok_list', 'ps_idx', 'point_map', 'intr', 'extr' }
with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    for imgs in tqdm(loader, desc="Caching VGGT outputs"):
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

# -----------------------------------------------------------------------------
# 3) 构造 Gaussian Head & 训练循环
# -----------------------------------------------------------------------------
sh_degree = 0
out_dim   = 3 + 3 + 4 + 3*(sh_degree+1)**2 + 1
g_head = Gaussianhead(
    2*vggt.embed_dim,
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

run_id = time.strftime("%Y%m%d-%H%M%S")
out_dir = "/home/stud/syua/storage/user/vggt-guassian"
render_dir = os.path.join(out_dir, "renders", run_id)
writer = SummaryWriter(os.path.join(out_dir, f"runs/gauss_train_{run_id}"))
os.makedirs(render_dir, exist_ok=True)

global_step = 0
for epoch in range(1, 30000):
    epoch_loss = 0.0
    for item in tqdm(cache, desc=f"Epoch {epoch}"):
        imgs      = item['imgs']
        tok_list  = item['tok_list']
        ps_idx    = item['ps_idx']
        point_map = item['point_map']
        intr      = item['intr']
        extr      = item['extr']
        B,_,H,W   = imgs.shape

        # Gaussian head 前向
        gdict_raw = g_head(tok_list, imgs.unsqueeze(1), ps_idx, point_map)
        gdict     = flatten_gdict(gdict_raw, B)
        gdict["sh"]= reg_dense_sh(gdict['sh'])
        renders   = render_gaussians(gdict, sh_degree,intr, extr, H, W)

        # 损失
        mse_loss   = crit(renders, imgs)
        with torch.amp.autocast("cuda", dtype=dtype):
            renders_n  = renders*2-1; imgs_n = imgs*2-1
            lpips_loss = lpips_fn(renders_n, imgs_n).mean()
            ssim_val   = ssim_fn(renders, imgs); ssim_loss = 1-ssim_val
            loss = mse_loss + lpips_w*lpips_loss + ssim_w*ssim_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        writer.add_scalar("loss/batch",   loss.item(),   global_step)
        writer.add_scalar("mse/batch",    mse_loss.item(),global_step)
        writer.add_scalar("lpips/batch",  lpips_loss.item(),global_step)
        writer.add_scalar("ssim/batch",   ssim_loss.item(),global_step)

        if global_step % 1000 == 0:
            save_image(
                torch.cat([imgs, renders], 0),
                os.path.join(render_dir, f"ep{epoch:02d}_it{global_step:06d}.png"),
                normalize=True, value_range=(0,1)
            )

            ### Log params
            gt_colors_mean = imgs.mean(dim=[2, 3])                 # (B, 3)
            pred_colors_mean = gdict_raw["sh"].mean(dim=[1,2]) # (B, 3)
            render_colors_mean = renders.mean(dim=[2, 3])          # (B, 3)

            txt_logfile = os.path.join(render_dir, "log.txt")

            with open(txt_logfile, "a") as f:
                for b in range(B):
                    f.write(
                        f"epoch:{epoch:04d} step:{global_step:06d} img_idx:{b:02d} "
                        f"GT:[{gt_colors_mean[b,0]:.4f}, {gt_colors_mean[b,1]:.4f}, {gt_colors_mean[b,2]:.4f}] "
                        f"SH:[{pred_colors_mean[b,0]:.4f}, {pred_colors_mean[b,1]:.4f}, {pred_colors_mean[b,2]:.4f}] "
                        f"RENDER:[{render_colors_mean[b,0]:.4f}, {render_colors_mean[b,1]:.4f}, {render_colors_mean[b,2]:.4f}] \n"
                    )
                f.write("\n")
        global_step += 1

    writer.add_scalar("loss/epoch", epoch_loss/len(cache), epoch)

writer.close()