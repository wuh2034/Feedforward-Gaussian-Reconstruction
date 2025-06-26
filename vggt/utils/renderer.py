# vggt/utils/renderer.py
import torch
import gsplat


@torch.amp.autocast("cuda", enabled=False)   # 渲染阶段保持 fp32
def render_gaussians(
    gdict: dict,   # keys = means / scales / rotations / opacities
    sh_degree: int,
    Ks: torch.Tensor,         # (B,3,3)  or (B,1,3,3)  or (B,4), intr
    viewmats: torch.Tensor,   # (B,4,4)  or (B,1,3,4) / (B,3,4), extr
    H: int,
    W: int,
    bg: str | None = "white",
    tile: int = 16,

) -> torch.Tensor:
    """
    Returns: (B, 3, H, W)  —  RGB in [0,1]
    """

    B, N, _ = gdict["means"].shape  # (B,N,3)

    # ------------------------------------------------------------------
    # 0. 处理 viewmats 形状与 dtype  → (B,1,4,4) float32
    # ------------------------------------------------------------------
    if viewmats.ndim == 4 and viewmats.shape[1] == 1:  # (B,1,3,4)
        viewmats = viewmats.squeeze(1)                 # (B,3,4)

    if viewmats.ndim == 3 and viewmats.shape[1:] == (3, 4):  # (B,3,4)
        pad_row = torch.tensor([0, 0, 0, 1],
                               device=viewmats.device,
                               dtype=torch.float32).view(1, 1, 4).expand(B, -1, -1)
        viewmats = torch.cat([viewmats.float(), pad_row], dim=1)  # (B,4,4)

    if viewmats.ndim == 3:  # (B,4,4)
        viewmats = viewmats.unsqueeze(1)               # (B,1,4,4)
    viewmats = viewmats.float()                        # 确保 fp32

    # ------------------------------------------------------------------
    # 1. 处理 Ks 形状与 dtype         → (B,1,3,3) float32
    # ------------------------------------------------------------------
    if Ks.ndim == 4 and Ks.shape[1] == 1:              # (B,1,3,3)
        Ks = Ks.float()
    elif Ks.ndim == 3:                                 # (B,3,3)
        Ks = Ks.unsqueeze(1).float()
    elif Ks.ndim == 2 and Ks.shape[1] == 4:            # (B,4)
        fx, fy, cx, cy = Ks.t()
        Ks = torch.stack([
                torch.stack([fx, torch.zeros_like(fx), cx], dim=-1),
                torch.stack([torch.zeros_like(fy), fy, cy], dim=-1),
                torch.tensor([0., 0., 1.], device=Ks.device).expand(B, -1)
             ], dim=1).unsqueeze(1).float()            # (B,1,3,3)
    else:
        raise ValueError(f"Unexpected Ks shape {Ks.shape}")

    # ------------------------------------------------------------------
    # 2. 准备颜色常量 & 调用 rasterization
    # ------------------------------------------------------------------
    # colors = torch.ones(B, N, 3, device=Ks.device, dtype=torch.float32)

    # def flatten_gdict(gdict: dict, B: int):
    #     out = {}
    #     for k, v in gdict.items():
    #         C = v.shape[-1] if v.ndim >= 2 else 1
    #         out[k] = v.reshape(B, -1, C)
    #         # out[k] = v.reshape(-1, C) #展开成(N, C)
    #     return out
    
    # for k, v in gdict.items():
    #     # 0 表示第一个维度，1 表示第二个维度
    #     gdict[k] = v.flatten(0, 1)  
    #     # 把第 0 到第 1 维都扁平化了，等价于 reshape(-1, *v.shape[2:])
    
    
    
    
    rgb, _, _ = gsplat.rasterization(
        sh_degree  = sh_degree,#传入sh阶数
        means      = gdict["means"].float(),
        quats      = gdict["rotations"].float(),
        scales     = gdict["scales"].float(),
        opacities  = gdict["opacities"].squeeze(-1).float(),
        colors     = gdict['sh'].float(),
        viewmats   = viewmats,   # (B,1,4,4)
        Ks         = Ks,         # (B,1,3,3)
        width      = W,
        height     = H,
        rasterize_mode = "antialiased",
        tile_size  = tile,
        backgrounds = None if bg is None else
                      torch.tensor([1.0, 1.0, 1.0],
                                   device=Ks.device,
                                   dtype=torch.float32),
    )  # → (B,1,H,W,3)

    return rgb.squeeze(1).permute(0, 3, 1, 2).clamp(0, 1)  # (B,3,H,W)
