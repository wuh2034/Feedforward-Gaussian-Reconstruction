# vggt/utils/renderer.py  ────────────────────────────────────────────────────
import torch, gsplat

@torch.amp.autocast("cuda", enabled=False)
def render_gaussians(gdict, Ks, viewmats, H: int, W: int,
                     bg: str | None = "white", tile: int = 16):

    B, N, _ = gdict["means"].shape

    # ---------- 0.  修补 viewmats 形状 (B,3,4) → (B,4,4) -------------------
    if viewmats.ndim == 3 and viewmats.shape[1:] == (3, 4):
        pad = torch.tensor([0, 0, 0, 1], dtype=viewmats.dtype,
                           device=viewmats.device).view(1, 1, 4).expand(B, -1, -1)
        viewmats = torch.cat([viewmats, pad], dim=1)        # (B,4,4)  <<< NEW

    # ---------- 1. 统一 Ks → (B,3,3)（与之前代码一致） ---------------------
    if Ks.ndim == 4:                           # (B,1,3,3)
        Ks = Ks.squeeze(1)
    elif Ks.ndim == 2 and Ks.shape[1] == 4:    # (B,4) = fx,fy,cx,cy
        fx, fy, cx, cy = Ks.t()
        Ks = torch.stack([
            torch.stack([fx, torch.zeros_like(fx), cx], dim=-1),
            torch.stack([torch.zeros_like(fy), fy, cy], dim=-1),
            torch.tensor([0, 0, 1], device=Ks.device).expand(B, -1)
        ], dim=1)                              # (B,3,3)

    # ---------- 2. 常量颜色 & 调用 rasterization ---------------------------
    colors = torch.ones(B, N, 3, device=Ks.device)

    rgb, _, _ = gsplat.rasterization(
        means      = gdict["means"],
        quats      = gdict["rotations"],
        scales     = gdict["scales"],
        opacities  = gdict["opacities"].squeeze(-1),
        colors     = colors,
        viewmats   = viewmats,                 # 现在一定 (B,4,4)
        Ks         = Ks,                       # (B,3,3)
        width      = W,
        height     = H,
        rasterize_mode = "antialiased",
        tile_size  = tile,
        backgrounds = None if bg is None else
                      torch.tensor([1,1,1], device=Ks.device),
    )                                           # (B,H,W,3)

    return rgb.permute(0,3,1,2).clamp(0,1)      # → (B,3,H,W)
