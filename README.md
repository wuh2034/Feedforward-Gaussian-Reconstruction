# Feedforward Gaussian Reconstruction

Feedforward Gaussian reconstruction based on VGGT (Visual Geometry Grounded Transformer).

## Overview

This project implements feedforward Gaussian reconstruction using VGGT, generating 3D Gaussian representations from single or multiple images with differentiable rendering.

Features:
- Fast 3D reconstruction using VGGT
- Differentiable Gaussian rendering
- Multi-loss training (MSE, LPIPS, SSIM)
- Complete training and inference pipeline

## Environment Setup

### Requirements

- Linux
- NVIDIA GPU with CUDA 12.4 support
- Python 3.10
- Conda or Miniconda

### Installation

Run the automated setup script:

```bash
bash setup_3dgs_vggt.sh
```

The script creates a Conda environment `vggt-3dgs` and installs:
- CUDA Toolkit 12.4
- PyTorch 2.4.1 (CUDA 12.4)
- 3D Gaussian Splatting extensions
- Flash Attention 2
- VGGT and dependencies

If the environment exists, the script prompts for confirmation before recreating.

Activate the environment:
```bash
conda activate vggt-3dgs
```

## Training

### Quick Start

```bash
python train_diff_gaussian.py
```

### Configuration

Modify parameters in `train_diff_gaussian.py`:

- Image directory: `images/` (default)
- Image formats: `.jpg`, `.JPG`, `.jpeg`, `.JPEG`
- Batch size: `8` (default)
- Epochs: `30000` (default)
- Learning rate: `1e-4` (default)
- Loss weights:
  - MSE: 1.0
  - LPIPS: 0.2
  - SSIM: 0.2

### Outputs

Training generates:
- Checkpoints: `runs/gauss_train_{timestamp}/`
- Rendered images: `renders/{timestamp}/`
- TensorBoard logs: `runs/gauss_train_{timestamp}/`

View training progress:
```bash
tensorboard --logdir runs/gauss_train_{timestamp}
```

## Inference

### Usage

```bash
python inference.py \
    --img_dir /path/to/images \
    --ckpt /path/to/checkpoint.pth \
    --out_dir /path/to/output
```

### Arguments

- `--img_dir`: Directory containing input images (default: 18 images)
- `--ckpt`: Path to trained Gaussian head checkpoint
- `--out_dir`: Output directory for results and metrics
- `--device`: Computation device (default: `cuda`)

### Pipeline

The inference script:
1. Loads pretrained VGGT model
2. Loads trained Gaussian head model
3. Evaluates on different input group sizes (default: 4, 6, 8, 10, 12, 14, 16, 18)
4. Renders views and compares with ground truth
5. Computes metrics (PSNR, SSIM, LPIPS)
6. Saves Gaussian point clouds as PLY files

### Output Files

- `GT_vs_Render_N{size}.png`: Ground truth vs rendered comparison
- `gaussians_N{size}.ply`: Gaussian point cloud PLY file
- `metrics_over_groups.txt`: Metrics in text format
- `metrics_over_groups.json`: Metrics in JSON format

## Project Structure

```
.
├── train_diff_gaussian.py      # Main training script
├── inference.py                 # Inference script
├── train.py                     # Large-scale training (ScanNet++ dataset)
├── train_gsplat.py             # Training with gsplat renderer
├── setup_3dgs_vggt.sh          # Environment setup script
│
├── inferdataset.py             # Inference dataset class
├── inferNewdataloader.py       # Inference data loader
├── Newdataloader.py            # Training data loader
├── dataset.py                  # Dataset definitions
│
├── vggt/                       # VGGT model code
│   ├── models/
│   ├── heads/
│   └── utils/
│
├── visual_util.py              # Visualization utilities
├── vggt_to_colmap.py          # COLMAP format conversion
├── demo_gradio.py             # Gradio web interface
└── demo_viser.py              # Viser 3D visualization
```

## Utilities

### COLMAP Export

Convert VGGT outputs to COLMAP format:

```bash
python vggt_to_colmap.py \
    --image_dir /path/to/images \
    --output_dir colmap_output \
    --conf_threshold 50.0 \
    --mask_sky
```

### Visualization

Gradio web interface:
```bash
python demo_gradio.py
```

Viser 3D viewer:
```bash
python demo_viser.py --image_folder /path/to/images
```

## Dependencies

- PyTorch 2.4.1+cu124
- torchvision 0.19.1+cu124
- torchaudio 2.4.1+cu124
- 3D Gaussian Splatting
- Flash Attention 2
- VGGT
- lpips, piq, torchmetrics
- gradio, viser, trimesh
- onnxruntime (for sky segmentation)

See `setup_3dgs_vggt.sh` for the complete dependency list.

## Citation

If you use this project, please cite:

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## License

See the original VGGT project license.

## Acknowledgments

This project is built on [VGGT](https://github.com/facebookresearch/vggt) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
