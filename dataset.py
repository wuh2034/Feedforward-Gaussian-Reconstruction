"""
dataset.py
-----------
SceneDataset for ScanNet++-style scene hierarchy.

Expected directory structure:
  data_root/
    <scene_id>/
      dslr/resized_undistorted_images/*.jpg

Scene IDs are read from a split file (TXT) with one scene_id per line.
Dataset returns:
    imgs      : Tensor of shape (M, 3, H, W), preprocessed images in range [0,1]
    scene_id  : str, the scene identifier
    img_names : List[str], the filenames corresponding to imgs
"""

import os
import glob
import torch
from typing import List, Tuple
from torch.utils.data import Dataset
from vggt.utils.load_fn import load_and_preprocess_images

class SceneDataset(Dataset):
    """
    Dataset for loading a fixed number of images per scene.
    Samples img_num images per scene at a given stride, starting from a random offset.
    """
    def __init__(
        self,
        root_dir: str,
        scene_ids: List[str],
        img_num: int = 4,
        stride: int = 3,
        transform=load_and_preprocess_images,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.scene_ids = scene_ids
        self.img_num = img_num
        self.stride = stride
        self.transform = transform

    def __len__(self) -> int:
        return len(self.scene_ids)

    def _sample_indices(self, total_images: int) -> List[int]:
        """
        Sample img_num indices by cycling through images with given stride,
        starting from a random index modulo total_images.
        """
        start = torch.randint(0, total_images, ()).item()
        return [(start + i * self.stride) % total_images for i in range(self.img_num)]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, List[str]]:
        scene_id = self.scene_ids[idx]
        img_dir = os.path.join(
            self.root_dir, scene_id, "dslr", "resized_undistorted_images"
        )
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        if not img_paths:
            raise RuntimeError(f"No images found in {img_dir}")

        selected_indices = self._sample_indices(len(img_paths))
        selected_paths = [img_paths[i] for i in selected_indices]
        img_names = [os.path.basename(p) for p in selected_paths]
        imgs = self.transform(selected_paths)  # shape: (img_num, 3, H, W), float32, [0,1]

        return imgs, scene_id, img_names