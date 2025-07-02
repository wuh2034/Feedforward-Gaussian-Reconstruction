# """
# 彻底使用 diff_gaussian_rasterization；不再含 gsplat 回退逻辑。
# 需要先编译安装：
#     git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting
#     pip install gaussian-splatting/submodules/diff-gaussian-rasterization
# """

# from __future__ import annotations
# import torch
# from diff_gaussian_rasterization import ( 
#     GaussianRasterizationSettings,
#     GaussianRasterizer,
# )

# # ────────────────────────── util ──────────────────────────
# def _prep_K(K: torch.Tensor) -> torch.Tensor:               # → (B,3,3)
#     if K.ndim == 4:                                         # (B,1,3,3)
#         K = K.squeeze(1)
#     elif K.ndim == 2 and K.shape[1] == 4:                   # (B,4)  fx,fy,cx,cy
#         fx, fy, cx, cy = K.T
#         K = torch.stack([
#             torch.stack([fx, torch.zeros_like(fx), cx], -1),
#             torch.stack([torch.zeros_like(fy), fy, cy], -1),
#             torch.tensor([0., 0., 1.], device=K.device).expand(K.shape[0], -1)
#         ], 1)
#     return K.float()

# def _prep_ext(ext: torch.Tensor) -> torch.Tensor:           # → (B,4,4) world2cam
#     if ext.ndim == 4 and ext.shape[1] == 1:                 # (B,1,3,4)
#         ext = ext.squeeze(1)
#     if ext.ndim == 3 and ext.shape[1:] == (3, 4):
#         pad = torch.tensor([0,0,0,1], device=ext.device,
#                            dtype=ext.dtype).view(1,1,4).expand(ext.shape[0],-1,-1)
#         ext = torch.cat([ext, pad], 1)
#     return ext.float()

# def _tanfov(K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#     fx, fy = K[:,0,0], K[:,1,1]
#     return 0.5 / fx, 0.5 / fy                              # 见官方 train.py
# # ──────────────────────────────────────────────────────────


# @torch.amp.autocast(device_type="cuda", enabled=False)      # 保持 fp32
# def render_gaussians(
#     gdict: dict[str, torch.Tensor],   # (B,N,·)
#     Ks: torch.Tensor,                 # intrinsics
#     viewmats: torch.Tensor,           # world2cam
#     H: int, W: int,
#     bg: str | None = "white",
#     # tile: int = 16                    # 官方也支持，可保留
# ) -> torch.Tensor:
#     """
#     Returns: (B,3,H,W)  ∈ [0,1]  — 支持梯度反传
#     """
#     Ks    = _prep_K(Ks)
#     view  = _prep_ext(viewmats)
#     B, N, _ = gdict["means"].shape

#     # --- Insert cam2world, campos, sh_degree before constructing settings ---
#     cam2world = torch.inverse(view)        # (B,4,4)
#     campos    = cam2world[:, :3, 3]        # (B,3)
#     sh_degree = 0                          # current pipeline only DC

#     # —— 扁平化到 (M,·) 并预乘 α ————————————————————————————
#     means      = gdict["means"].reshape(-1, 3) #(B*N, ...)
#     scales     = gdict["scales"].reshape(-1, 3)
#     rotations  = gdict["rotations"].reshape(-1, 4)
#     #(B, N, 3)*(B, N, 1).reshape(-1, 3)=(B*N, 3)
#     opacities  = gdict["opacities"]
#     colors     = (gdict["colors"] * opacities).reshape(-1, 3)   # 官方约定
#     opacities  = opacities.reshape(-1, 1)
#     # —— 渲染设置 ————————————————————————————————————————————
#     tanx, tany = _tanfov(Ks)
#     settings = GaussianRasterizationSettings(
#         image_height   = H,
#         image_width    = W,
#         tanfovx        = tanx,
#         tanfovy        = tany,
#         bg             = 1.0 if bg == "white" else 0.0,
#         scale_modifier = 1.0,
#         viewmatrix     = view,      # world→cam (B,4,4)
#         projmatrix     = Ks,        # (B,3,3) intrinsics
#         sh_degree      = sh_degree,
#         campos         = campos,
#         prefiltered    = False,
#         debug          = False,
#         antialiasing   = True,
#     )

#     # —— 调用自定义 autograd.Function ————————————————
#     rgb, _, _ = GaussianRasterizer.apply(
#         means, colors, opacities,
#         scales, rotations,
#         torch.arange(B, device=means.device).repeat_interleave(N),  # batch ids
#         settings
#     )  # → (B,H,W,3)

#     return rgb.permute(0,3,1,2).clamp(0.0, 1.0)

"""
vggt/utils/renderer.py
----------------------

仅使用官方 diff_gaussian_rasterization 渲染器；
一次 forward 以 Python for-loop 方式逐视图渲染，
无需 batch_ids，也不会触发参数数量错误。
"""

from __future__ import annotations
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

# ────────────────────── 辅助函数 ──────────────────────
def _prep_K(K: torch.Tensor) -> torch.Tensor:
    """将内参归一为 (B,3,3)。支持 (B,3,3)/(B,1,3,3)/(B,4:fx,fy,cx,cy)。"""
    if K.ndim == 4 and K.shape[1] == 1:
        K = K.squeeze(1)
    elif K.ndim == 2 and K.shape[1] == 4:  # fx,fy,cx,cy
        fx, fy, cx, cy = K.T
        K = torch.stack(
            [
                torch.stack([fx, torch.zeros_like(fx), cx], -1),
                torch.stack([torch.zeros_like(fy), fy, cy], -1),
                torch.tensor([0.0, 0.0, 1.0], device=K.device).expand(K.shape[0], -1),
            ],
            1,
        )
    assert K.shape[-2:] == (3, 3), f"Intrinsics 形状应为 (B,3,3)，得到 {K.shape}"
    return K.float()


def _prep_ext(ext: torch.Tensor) -> torch.Tensor:
    """将外参归一为 (B,4,4) world→cam。支持 (B,4,4)/(B,3,4)/(B,1,3,4)。"""
    if ext.ndim == 4 and ext.shape[1] == 1:  # (B,1,3,4)
        ext = ext.squeeze(1)
    if ext.ndim == 3 and ext.shape[1:] == (3, 4):
        pad = torch.tensor([0, 0, 0, 1], dtype=ext.dtype, device=ext.device)\
                 .view(1, 1, 4).expand(ext.shape[0], -1, -1)
        ext = torch.cat([ext, pad], 1)
    assert ext.shape[-2:] == (4, 4), f"Extrinsics 形状应为 (B,4,4)，得到 {ext.shape}"
    return ext.float()


def _tanfov(K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """由内参矩阵计算 tan(fovx), tan(fovy)。"""
    fx, fy = K[:, 0, 0], K[:, 1, 1]
    return 0.5 / fx, 0.5 / fy


# ────────────────────── 主渲染函数 ─────────────────────
@torch.amp.autocast(device_type="cuda", enabled=False)   # 全程 fp32 更稳
def render_gaussians(
    gdict: dict[str, torch.Tensor],   # (B,N,·)
    Ks: torch.Tensor,                 # (B,3,3) 或可解析形式
    viewmats: torch.Tensor,           # (B,4,4) world→cam 或可解析形式
    H: int,
    W: int,
    bg: str | None = "white",
) -> torch.Tensor:
    """
    返回 (B,3,H,W) ∈ [0,1]，支持梯度回传。
    逐视图渲染，兼容多场景混合 batch。
    """
    Ks   = _prep_K(Ks)
    view = _prep_ext(viewmats)
    B, N, _ = gdict["means"].shape
    outputs = []

    for b in range(B):
        # 取第 b 张图对应数据
        means_b  = gdict["means"][b]           # (N,3)
        scales_b = gdict["scales"][b]
        rots_b   = gdict["rotations"][b]
        opa_b    = gdict["opacities"][b]       # (N,1)
        col_b    = gdict["colors"][b]            # (N,3)
       
        # 构造 rasterizer 设置
        tanx, tany = _tanfov(Ks[b:b+1])
        campos     = torch.inverse(view[b:b+1])[:, :3, 3]  # (1,3)
        bg_tensor  = torch.tensor([1.0, 1.0, 1.0],  # 或 torch.tensor(1.0)
                                device=means_b.device, dtype=torch.float32)
        settings = GaussianRasterizationSettings(
            image_height   = H,
            image_width    = W,
            tanfovx        = tanx,
            tanfovy        = tany,
            bg             = bg_tensor,
            scale_modifier = 0.6, # 1.0->0.6
            viewmatrix     = view[b:b+1],  # (1,4,4)
            projmatrix     = Ks[b:b+1],    # (1,3,3)
            sh_degree      = 0,
            campos         = campos,
            prefiltered    = False,
            debug          = False,
            antialiasing   = True,
        )

        rasterizer = GaussianRasterizer(settings)
        rgb_b, _, _ = rasterizer(
            means3D        = means_b,
            means2D        = torch.empty(0, device=means_b.device),
            opacities      = opa_b,
            colors_precomp = col_b,         # 预乘 α
            scales         = scales_b,
            rotations      = rots_b,
        )  # → (3,H,W)

        rgb_b = rgb_b.unsqueeze(0)           # 变成 (1,3,H,W)
#       rgb_b = rgb_b.permute(0, 3, 1, 2)    # 变成 (1,3,H,W)
        outputs.append(rgb_b)

    # 拼接并整理通道顺序
    rgb = torch.cat(outputs, dim=0)            # (B,H,W,3)
    return rgb.clamp(0.0, 1.0)