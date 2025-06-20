#!/usr/bin/env bash
###############################################################################
# 3D Gaussian Splatting  +  vggt  统一环境安装脚本（CUDA 12.4 / PyTorch 2.3.1）
# Author: ChatGPT (2025-06-15)
# Usage : bash setup_3dgs_vggt.sh
###############################################################################
set -euo pipefail

##################### 可选参数 ##################################################
ENV_NAME=${ENV_NAME:-vggt-3dgs}           # Conda 环境名
TORCH_VER=2.4.1+cu124                      # PyTorch / cu124 版本号
TVISION_VER=0.19.1+cu124                   # torchvision
TAUDIO_VER=2.4.1+cu124                     # torchaudio
CUDA_TOOLKIT_LABEL=nvidia/label/cuda-12.4.1
EXTRA_INDEX=https://download.pytorch.org/whl/cu124
###############################################################################

echo "=== [0/8] Conda 环境准备：$ENV_NAME ========================================"
# Conda base 环境初始化
if ! command -v conda &>/dev/null; then
  echo "[错误] 未检测到 conda，请先安装 Miniconda/Anaconda。" && exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# 若同名环境已存在，提示用户是否删除
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  read -rp "⚠️  Conda env '$ENV_NAME' 已存在，是否删除并重建？[y/N] " yn
  [[ $yn =~ ^[Yy]$ ]] && conda remove -n "$ENV_NAME" --all -y
fi

conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"

echo -e "\n=== [1/8] 安装 CUDA Toolkit 12.4 ========================================="
conda install -c "$CUDA_TOOLKIT_LABEL" cuda-toolkit -y   # nvcc 12.4.1
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo -e "\n=== [2/8] 安装 PyTorch $TORCH_VER & 依赖 ==================================="
python -m pip install -U pip
python -m pip install --upgrade --force-reinstall \
  "torch==$TORCH_VER" "torchvision==$TVISION_VER" "torchaudio==$TAUDIO_VER" \
  --extra-index-url "$EXTRA_INDEX"

python -m pip install tqdm plyfile joblib                           # 3DGS 原依赖

echo -e "\n=== [3/8] 克隆 3DGS 源码 & 打补丁 ========================================"
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
# 某些编译错误的补丁（若已修复则跳过）
sed -i '1i #include <float.h>' submodules/simple-knn/simple_knn.cu              || true
sed -i '1i #include <cstdint>' submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h || true

echo -e "\n=== [4/8] 编译 3DGS 自定义 C++/CUDA 扩展 (3/3) ==========================="
pip install -e submodules/simple-knn                 --no-build-isolation --config-settings editable_mode=compat --force-reinstall
pip install -e submodules/diff-gaussian-rasterization --no-build-isolation --config-settings editable_mode=compat --force-reinstall
pip install -e submodules/fused-ssim                 --no-build-isolation --config-settings editable_mode=compat --force-reinstall

echo -e "\n=== [5/8] 安装 Flash-Attention 2（匹配新 Torch） =========================="
python -m pip install --force-reinstall --no-build-isolation flash-attn

echo -e "\n=== [6/8] 安装 vggt 追加依赖 ============================================="
python -m pip install --upgrade \
  numpy==1.26.1 pillow einops safetensors tqdm \
  hydra-core omegaconf requests \
  opencv-python scipy matplotlib trimesh \
  huggingface_hub gradio==5.17.1 viser==0.2.23 onnxruntime

# 额外三方项目（COLMAP/LightGlue）
python -m pip install --upgrade pycolmap==3.10.0 pyceres==2.3 \
  "git+https://github.com/jytime/LightGlue.git#egg=lightglue"

echo -e "\n=== [7/8] 终极自检（3DGS + Flash-Attn） ================================"
python - <<'PY'
import torch, os, importlib
print("\n=== 核心版本信息 ===")
print("Torch:", torch.__version__, "| CUDA runtime:", torch.version.cuda)
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("nvcc path:", os.popen("which nvcc").read().strip())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA 不可用")

print("\n=== 3DGS 扩展测试 ===")
from simple_knn._C import distCUDA2
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
from fused_ssim import fused_ssim
pts = torch.randn(4096, 3, device='cuda')
print("simple_knn ✔", distCUDA2(pts).shape)
settings = GaussianRasterizationSettings(
    image_height=32, image_width=32, tanfovx=1.0, tanfovy=1.0,
    bg=torch.zeros(3,device='cuda'), scale_modifier=1.0,
    viewmatrix=torch.eye(4,device='cuda'), projmatrix=torch.eye(4,device='cuda'),
    sh_degree=0, campos=torch.zeros(3,device='cuda'),
    prefiltered=False, debug=False, antialiasing=True,
)
GaussianRasterizer(raster_settings=settings).cuda()
print("diff_gaussian_rasterization ✔")
img = torch.rand(1,3,64,64,device='cuda')
print("fused_ssim ✔", fused_ssim(img,img).item())

print("\n=== Flash-Attention 2 测试 ===")
fa = importlib.import_module("flash_attn_2_cuda")
print("Flash-Attn 2 ✔", hasattr(fa, "flash_attn_bwd"))
PY

echo -e "\n=== [8/8] 环境部署完成！ ================================================"
echo "🎉  3DGS + vggt 统一环境已就绪！"
echo "👉  记得：conda activate $ENV_NAME  后再运行你的脚本。"
