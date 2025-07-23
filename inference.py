import os
import json
from typing import List, Dict
import torch
from torchvision.utils import save_image
import lpips
from piq import ssim
from torchmetrics.image import PeakSignalNoiseRatio as PSNR

from inferNewdataloader import build_loader_raw
from vggt.models.vggt import VGGT
from vggt.heads.gaussian_head import Gaussianhead
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.renderer_gsplat import render_gaussians

# Total number of images to use for inference
TOTAL_IMAGES = 18
# Indices (0-based) of ground-truth images
GT_INDICES = [5, 11]
# Possible sizes of input groups
GROUP_SIZES = [4, 6, 8, 10, 12, 14, 16, 18]

# Utility function to reshape Gaussian output dictionary
def flatten_gdict(gdict: dict, batch_size: int):
    result = {}
    for key, value in gdict.items():
        if value.ndim >= 3:
            reshaped = value.reshape(batch_size, -1, value.shape[-1])
            result[key] = reshaped.view(1, -1, value.shape[-1])
        else:
            reshaped = value.reshape(batch_size, -1)
            result[key] = reshaped.view(1, -1)
    return result

# Return the first existing entry for a list of candidate keys
def _first_key(d: dict, keys: List[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None

# Save Gaussian points and attributes to a PLY file
def save_gaussians_to_ply(gdict: dict, path: str):
    # Extract 3D coordinates
    xyz = _first_key(gdict, ['means3D', 'means'])
    if xyz is None:
        raise ValueError('Gaussian data missing 3D means')
    # Extract color and opacity
    rgb = _first_key(gdict, ['rgb', 'colors'])
    opacity = _first_key(gdict, ['opacity', 'opacities'])
    # Extract scale values
    scale = gdict.get('scales')

    xyz = xyz[0].cpu().float()
    num_points = xyz.size(0)
    # Convert colors to uint8
    if rgb is not None:
        rgb_vals = (rgb[0].cpu() * 255).clamp(0, 255).to(torch.uint8)
    else:
        rgb_vals = torch.zeros_like(xyz, dtype=torch.uint8)
    # Handle missing opacity and scale
    opacity_vals = opacity[0].cpu().float() if opacity is not None else torch.zeros(num_points, 1)
    scale_vals = scale[0].cpu().float() if scale is not None else torch.zeros_like(xyz)
    if scale_vals.ndim == 1:
        scale_vals = scale_vals.unsqueeze(1).repeat(1, 3)

    # Prepare PLY header
    header = [
        'ply',
        'format ascii 1.0',
        f'element vertex {num_points}',
        'property float x',
        'property float y',
        'property float z',
        'property uchar red',
        'property uchar green',
        'property uchar blue',
        'property float opacity',
        'property float scale_x',
        'property float scale_y',
        'property float scale_z',
        'end_header'
    ]
    # Create PLY body lines
    body = []
    for (x, y, z), (r, g, b), a, (sx, sy, sz) in zip(
        xyz.tolist(), rgb_vals.tolist(), opacity_vals.squeeze(1).tolist(), scale_vals.tolist()
    ):
        body.append(f"{x} {y} {z} {r} {g} {b} {a} {sx} {sy} {sz}")

    # Write to file
    with open(path, 'w') as f:
        f.write('\n'.join(header + body))
    print(f"Saved {num_points} Gaussian points to {path}")

# Main inference routine
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='/usr/prakt/s0012/ADL4VC/test_set',
                        help='Directory containing 18 images for inference')
    parser.add_argument('--ckpt', default='/usr/prakt/s0012/ADL4VC/checkpoints/withoutAUX_40_6/epoch_0201.pth',
                        help='Path to the Gaussian head checkpoint')
    parser.add_argument('--out_dir', default='/usr/prakt/s0012/ADL4VC/infer_10',
                        help='Directory to save inference outputs')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Computation device')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device
    dtype = torch.bfloat16 if (device == 'cuda' and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    print(f"Device: {device}, data type: {dtype}")

    # Load images without shuffling
    loader = build_loader_raw(
        img_dir=args.img_dir,
        shuffle=False,
        num_workers=2,
        verbose=True
    )
    images, image_names = next(iter(loader))  # Shape: (TOTAL_IMAGES, 3, H, W)
    print(f"Loaded {len(image_names)} images from {args.img_dir}")

    # Load pre-trained VGGT and freeze parameters
    vggt = VGGT.from_pretrained('facebook/VGGT-1B').eval().to(device)
    for p in vggt.parameters():
        p.requires_grad_(False)

    # Initialize Gaussian head and load checkpoint
    gaussian_head = Gaussianhead(
        2 * vggt.embed_dim,
        3 + 3 + 4 + 3 * (0 + 1)**2 + 1,
        activation='exp',
        conf_activation='expp1',
        sh_degree=0
    ).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    # Support different key names for state dict
    model_state = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    gaussian_head.load_state_dict(model_state)
    gaussian_head.eval()
    print('Checkpoint loaded successfully')

    # Setup metrics
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    psnr_fn = PSNR(data_range=1.0).to(device)

    results: List[Dict] = []

    # Iterate over different group sizes
    for group_size in GROUP_SIZES:
        num_generate = group_size - len(GT_INDICES)
        print(f"Running inference with {num_generate} generated views")

        half = num_generate // 2
        generate_indices = list(range(half)) + list(range(TOTAL_IMAGES - half, TOTAL_IMAGES))

        generated_views = images[generate_indices].to(device)
        ground_truth_views = images[GT_INDICES].to(device)

        # Combine for feature extraction
        all_views = torch.cat([generated_views, ground_truth_views], dim=0)
        batch_size, _, H, W = generated_views.shape

        with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
            tokens, pose_idxs = vggt.aggregator(all_views.unsqueeze(1))
            predictions = vggt(all_views)
            point_maps = predictions['world_points'].unsqueeze(1)
            pose_encodings = predictions['pose_enc']

            # Split tokens and point maps for generated views
            gen_tokens = [t[:num_generate] for t in tokens]
            gen_point_maps = point_maps[:, :, :num_generate]

            # Decode extrinsics and intrinsics
            extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_encodings, (H, W))
            extr_gen = extrinsics[:, :num_generate]
            intr_gen = intrinsics[:, :num_generate]
            extr_gt = extrinsics[:, num_generate:]
            intr_gt = intrinsics[:, num_generate:]

            # Generate Gaussian representation
            raw_gaussians = gaussian_head(gen_tokens, generated_views.unsqueeze(1), pose_idxs, gen_point_maps)
            gaussian_dict = flatten_gdict(raw_gaussians, num_generate)
            # Ensure standard key names exist
            gaussian_dict.setdefault('means3D', gaussian_dict.get('means'))
            gaussian_dict.setdefault('rgb', gaussian_dict.get('colors'))
            gaussian_dict.setdefault('opacity', gaussian_dict.get('opacities'))
            # Convert all values to float
            gaussian_dict = {k: v.float() for k, v in gaussian_dict.items()}

        # Render ground-truth viewpoints using Gaussian model
        rendered_views = []
        for i in range(len(GT_INDICES)):
            intr = intr_gt[:, i:i+1]
            extr = extr_gt[:, i:i+1]
            extr_h = torch.cat([extr.to(device).float(), torch.tensor([0, 0, 0, 1], device=device).view(1,1,1,4)], dim=-2)
            render = render_gaussians(gaussian_dict, intr.to(device).float(), extr_h, H, W)
            rendered_views.append(render)
        rendered_views = torch.cat(rendered_views, dim=0)

        # Compute metrics per group
        psnr_vals = [psnr_fn(rendered_views[i:i+1], ground_truth_views[i:i+1]).item() for i in range(len(GT_INDICES))]
        ssim_vals = [ssim(rendered_views[i:i+1], ground_truth_views[i:i+1], data_range=1.0).mean().item() for i in range(len(GT_INDICES))]
        lpips_vals = [lpips_fn(rendered_views[i:i+1], ground_truth_views[i:i+1]).mean().item() for i in range(len(GT_INDICES))]

        results.append({
            'N': num_generate,
            'psnr': sum(psnr_vals) / len(psnr_vals),
            'ssim': sum(ssim_vals) / len(ssim_vals),
            'lpips': sum(lpips_vals) / len(lpips_vals)
        })

        # Save visualization of ground truth vs render
        grid = torch.cat([ground_truth_views.cpu(), rendered_views.cpu()], dim=0)
        save_image(grid, os.path.join(args.out_dir, f"GT_vs_Render_N{num_generate}.png"), nrow=len(GT_INDICES), normalize=True, value_range=(0,1))
        print('Saved comparison image')

        # Save Gaussian point cloud to PLY
        save_gaussians_to_ply(gaussian_dict, os.path.join(args.out_dir, f"gaussians_N{num_generate}.ply"))

    # Write metrics to text and JSON
    metrics_txt = os.path.join(args.out_dir, 'metrics_over_groups.txt')
    with open(metrics_txt, 'w') as f:
        for m in results:
            f.write(f"N={m['N']:2d}  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  LPIPS={m['lpips']:.4f}\n")
    print(f"Saved metrics to {metrics_txt}")

    metrics_json = os.path.join(args.out_dir, 'metrics_over_groups.json')
    with open(metrics_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON metrics to {metrics_json}")

if __name__ == '__main__':
    main()