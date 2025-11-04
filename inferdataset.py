import os
import glob
import torch
import numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset
from vggt.utils.load_fn import load_and_preprocess_images

class RawDataset(Dataset):
    """
    Dataset for a single folder: loads and preprocesses all images at once.
    Returns tensor imgs (N, 3, H, W) and corresponding filename list.
    """
    def __init__(self, img_dir: str,
                 transform=load_and_preprocess_images):
        super().__init__()
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        if not self.img_paths:
            raise RuntimeError(f"No images found in {img_dir}")
        self.transform = transform

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        imgs = self.transform(self.img_paths)
        names = [os.path.basename(p) for p in self.img_paths]
        return imgs, names

class SceneDataset(Dataset):
    """
    Dataset for ScanNet++-style scene hierarchy.
    Expects structure:
      root_dir/<scene_id>/dslr/resized_undistorted_images/*.jpg

    scene_ids: list of scene_id strings loaded from split file (.txt)
    __getitem__ returns (imgs, scene_id, img_names)
    where imgs are preprocessed by load_and_preprocess_images.
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
        Sample img_num indices by cycling with given stride from a random start.
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

        selected = self._sample_indices(len(img_paths))
        selected_paths = [img_paths[i] for i in selected]
        img_names = [os.path.basename(p) for p in selected_paths]
        imgs = self.transform(selected_paths)

        return imgs, scene_id, img_names