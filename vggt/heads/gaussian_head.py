# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import os
from typing import List, Dict, Tuple, Union
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed

class Gaussianhead(nn.Module):#继承父类
    """
    DPT  Head for dense prediction tasks.

    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
    backbone and produces dense predictions by fusing multi-scale features.

    Args:
        dim_in (int): Input dimension (channels).
        patch_size (int, optional): Patch size. Default is 14.
        output_dim (int, optional): Number of output channels. Default is 4.
        activation (str, optional): Activation type. Default is "inv_log".
        conf_activation (str, optional): Confidence activation type. Default is "expp1".
        features (int, optional): Feature channels for intermediate representations. Default is 256.
        out_channels (List[int], optional): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
        pos_embed (bool, optional): Whether to use positional embedding. Default is True.
        feature_only (bool, optional): If True, return features only without the last several layers and activation head. Default is False.
        down_ratio (int, optional): Downscaling factor for the output resolution. Default is 1.
    """

    def __init__(#先让 子类 完成自身初始化
        self,
        dim_in: int,# Transformer输出通道数 张量形状(B, 197, 768) 一个batch中B张图片,一个图片分成14x14+1个patch, 一个patch 16x16x3
        #Transformer 输出 token 张量的形状是 (B, N, dim_in)
        patch_size: int = 14,
        output_dim: int = 4, # number of output channels
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,#选取的中间层token经过处理后有不同的C, 还需要把他们统一到相同的C才能拼接,这里选取的C=256
        out_channels: List[int] = [256, 512, 1024, 1024],#这里的意思是指从tranformer的不同层中取出中间的token,然后用1x1 conv得到不同channel的特征图
        #比如从4, 11, 17, 23层分别取出形状(B,N,D)的token后,用conv 1x1分别映射成(BxCxHxW)的空间特征图
        #低层映射的C小,侧重捕捉局部纹理、边缘等低级视觉特征,中层（如第 8 层）开始体现小范围图案与形状；深层（如第12层）则包含全局语义和高阶抽象信息
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],#选取不同层的tranformer
        pos_embed: bool = True,#是否在self.projection空间特征图上加入positional encoding
        feature_only: bool = False,#是否只输出融合后的空间特征图，不再进行高斯参数的回归预测。
        #当你只想把 Gaussianhead 当作一个多尺度特征融合模块（而非最终的高斯参数头）来使用时，可将 feature_only=True。
        # 这时，网络会在融合骨架（fusion）之后直接使用 output_conv1（一个 3×3 卷积）输出通道数为 features 的特征图，并不执行后续的 output_conv2、gaussian_postprocess 等高斯参数计算逻辑。
        down_ratio: int = 1,#如果 down_ratio=1，融合后的特征图会被插值回与原始图像相同的分辨率。例如 down_ratio=2 时，输出分辨率是输入宽高的一半。
        sh_degree: int = 0
    ) -> None:
        super(Gaussianhead, self).__init__()#super(ChildClass, instance),调用父类构造函数（__init__）
        #接着再做子类自己的初始化
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        self.sh_degree = sh_degree

        self.norm = nn.LayerNorm(dim_in)#对输入的token做layernorm, 在(B,S,D)中的D上做归一化
        #nn.LayerNorm 是一个类（class），位于 torch.nn 模块下。D表示对某个通道做归一化


        # Projection layers for each output channel from tokens. 1x1 conv
        #这里处理的是transformer中间层的patch token(B, S, D), dim_in = D, 经过重塑后变为(B, D, Hi, Wi)作为输入
        #输出: (B, oc, Hi, Wi)对于不同中间层的patch token, 提取不同的C表达不同层级的特征
        self.projects = nn.ModuleList(# nn.ModuleList是一个容器, 里面可以有不同的子模块nn.Conv2d, nn.Idnetity, nn.ConvTranspose2d,还可以用for loop迭代使用
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in out_channels
            ]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(#把Projection后的token size变为统一的 H x W,提供了四种方法统一分辨率
            [
                nn.ConvTranspose2d(#上采样
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),# x4上采样
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ), # x2上采样
                nn.Identity(),#保持分辨率
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ), # 1/2下采样
            ]
        )

        self.scratch = _make_scratch(#调用_make_scratch方法, 对齐所有空间特征图的C = features
            out_channels,
            features,
            expand=False,
        )

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32

        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

        with torch.no_grad():
        # 最后 conv2 输出通道顺序 = [offset(3) | scale(3) | rot(4) | sh | opacity(1)]
            self.scratch.output_conv2[-1].bias[..., -1]   = 2.0          # σ⁻¹(0.88) ≈ α0.88
            self.scratch.output_conv2[-1].bias[..., -4:-1] = math.log(0.03)  # σ ≈0.03
                # 新增：初始化颜色 bias（3 个通道）到 [-2, 2]
        color_start = 3 + 3 + 4                # offset(3)+scale(3)+rot(4)
        color_end   = color_start + 3          # 只针对 sh_degree=0 的 3 通道
        nn.init.uniform_(self.scratch.output_conv2[-1].bias[color_start:color_end], -2.0, 2.0)


    def forward(#切分→调度→拼接
        self,
        aggregated_tokens_list: List[torch.Tensor],#从 Transformer 各中间层抽出的 token 序列，元素形状均为 (B, S, D)
        images: torch.Tensor,#输入图像序列，B 批次大小，S 帧数，H×W 空间分辨率(B是一次输入的样本多少, S是一个样本有多少帧, HxW是一帧的分辨率)
        patch_start_idx: int,#patch_start_idx 标记 patch token 在序列中的起始下标
        point_map: torch.Tensor,# 用于后续偏移计算
        frames_chunk_size: int = 8,#控制分块处理帧数
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:# ->表示函数返回值提示, Union表示返回值可以是 类型 A 或 类型 B
        """
        Forward pass through the DPT head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 8.

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - If feature_only=True: Feature maps with shape [B, S, C, H, W]
                - Otherwise: Tuple of (predictions, confidence) both with shape [B, S, 1, H, W]
        """
        B, S, _, H, W = images.shape# C固定为3, 这里不使用,仅占位

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        # S 可能很大,直接送入_forward_impl处理内存占据大, 可以分块进行, 设置一个frames_chunk_size < S
        # 如果不分块, 或者分块尺寸>S,就一次都送入_forward_impl处理
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, point_map, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_preds = [] #定义最后的predictions
        all_conf = []
        #生成chunk的idx, 同时保证frames_end_idx不超过S
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # Process batch of frames 分块处理
            if self.feature_only:#只返回特征图 (B, S_chunk, C, H, W)，追加到 all_preds。
                chunk_output = self._forward_impl(
                    aggregated_tokens_list, images, point_map, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_output)
            else:#返回高斯参数和置信度
                chunk_preds, chunk_conf = self._forward_impl(
                    aggregated_tokens_list, images, point_map, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)

        # Concatenate results along the sequence dimension
        if self.feature_only:
            return torch.cat(all_preds, dim=1)# 在S上拼接所有chunk
        else:
            return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)

    def _forward_impl(#实际的特征生成与高斯参数回归
    #整条流水线从最初的输入 (B,196,768) patch token序列，
    #到最终的输出 (B·196, C_out, 16,16) 融合特征图，完成了序列→空间→多尺度融合→平滑的全部操作。
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        point_map: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0

        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.view(B * S, -1, x.shape[-1])#这里是把(B, S, D)变成(B*S, D, H, W)的空间特征图然后进行projection
                                            # S是patch的数量
            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x) # (B·S, oc_i, h_i, w_i)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[dpt_idx](x) # → (B·S, oc_i, H, W) 同一 H×W

            out.append(x)#这里得到的features是具有不同C的空间特征图(B, oc, H, W)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        # Interpolate fused output to match target image resolution.
        #将 DPT 融合后的低分辨率特征图恢复到与原始图像或指定缩放比例相同的大小，便于做后续像素级的回归或可微渲染
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        out = self.scratch.output_conv2(out)
        out = self.gaussian_postprocess(out, point_map)

        return out
    
    def gaussian_postprocess(self, out: torch.Tensor, point_map: torch.Tensor, use_offsets: bool=True):
        out = out.permute(0, 2, 3, 1) # shape: (B,D,H,W) -> (B,H,W,D)
        point_map = point_map.squeeze(1) # shape: (B,1,H,W,D) -> (B,H,W,D)
        
        offset, scales, rotations, sh, opacities = torch.split(out, [3, 3, 4, 3*(self.sh_degree+1)**2, 1], dim=-1)
        
        offset = reg_dense_offsets(offset)
        scales = reg_dense_scales(scales)
        rotations = reg_dense_rotation(rotations)
        sh = reg_dense_sh(sh)
        opacities = reg_dense_opacities(opacities)
        colors = sh[..., 0] * 0.2820948

        res = {
            'scales': scales,
            'rotations': rotations,
            'sh': sh,
            'opacities': opacities,
            'colors': colors
        }
        if use_offsets:
            res['means'] = point_map.detach() + offset
        else:
            res['means'] = point_map.detach()

        return res

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features
        #调用layer1_rn方法,把所有(B, oc_i, H, W)统一映射到同一个C
        layer_1_rn = self.scratch.layer1_rn(layer_1)# oc_1 → features
        layer_2_rn = self.scratch.layer2_rn(layer_2)# oc_2 → features
        layer_3_rn = self.scratch.layer3_rn(layer_3)# oc_3 → features
        layer_4_rn = self.scratch.layer4_rn(layer_4)# oc_4 → features
        #传入的layer_4_rn是_make_scratch 时注册到 self.scratch的Conv2d的输出
        #执行 refinenet4 的融合操作,通过 _make_fusion_block(features, has_residual=False) 创建的 FeatureFusionBlock 实例。
        #流程: 先对输入做一次卷积平滑（resConfUnit2），然后上采样到与第 3 路特征相同的空间大小——由关键字参数 size=layer_3_rn.shape[2:] 指定。
        #新生成的张量 out 形状为 (B·S, features, H₃, W₃)这里所有层的token的分辨率之前已经对齐过了,都是 H x W
        #输入的变量都是(B, Features, H, W)做四层的token融合
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4 #删除变量绑定

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)#输入(B, features, H, W), 输出(B, Cout, H, W)
        return out


################################################################################
# Modules
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()#这里把scratch 定义成了一个nn.Module()容器
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:#如果做扩展
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch

# @MODIFIED
def reg_dense_offsets(xyz, shift=6.0):
    """
    Apply an activation function to the offsets so that they are small at initialization
    """
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    offsets = xyz * (torch.exp(d - shift) - torch.exp(torch.zeros_like(d) - shift))
    return offsets

# @MODIFIED
# def reg_dense_scales(scales):
#     """
#     Apply an activation function to the offsets so that they are small at initialization
#     """
#     scales = scales.exp()
#     return scales
def reg_dense_scales(scales, beta=10.0, min_sigma=0.01, max_sigma=0.1):
    scales = F.softplus(scales, beta=beta)
    return torch.clamp(scales, min_sigma, max_sigma)   # ← 非原地
# @MODIFIED
def reg_dense_rotation(rotations, eps=1e-8):
    """
    Apply PixelSplat's rotation normalization
    """
    return rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

# @MODIFIED
def reg_dense_sh(sh):
    """
    Apply PixelSplat's spherical harmonic postprocessing
    """
    sh = rearrange(sh, '... (xyz d_sh) -> ... xyz d_sh', xyz=3)
    return sh

# @MODIFIED
def reg_dense_opacities(opacities):
    return torch.clamp(opacities.sigmoid(), 1e-3, 0.98)  # ← 非原地


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)
