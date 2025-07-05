"""
dataset.py
──────────
SceneDataset for ScanNet++ style hierarchy:

data_root/
  <scene_id>/
    dslr/resized_undistorted_images/*.jpg

scene_id 列表由 split 文件 (.txt) 提供；每行一个 scene_id。
Dataset 返回 (imgs, scene_id, img_names)，其中 imgs 已经过 vggt.utils.load_fn.load_and_preprocess_images
"""

import os, glob, torch, numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset
from vggt.utils.load_fn import load_and_preprocess_images


class SceneDataset(Dataset):
    def __init__(
        self,
        root_dir   : str,
        scene_ids  : List[str],      # 从 split txt 读入
        img_num    : int  = 4,
        stride     : int  = 3,
        tfm              = load_and_preprocess_images,
    ):
        super().__init__()
        self.root_dir  = root_dir
        self.scene_ids = scene_ids
        self.img_num   = img_num
        self.stride    = stride
        self.tfm       = tfm

    def __len__(self) -> int:
        return len(self.scene_ids)

    # ---------- helper -------------------------------------------------------
    def _sample_indices(self, n_total: int) -> List[int]:
        """
        从随机起点开始，按照 stride 间隔循环取 img_num 张索引。
        """
        start = torch.randint(0, n_total, ()).item()
        return [(start + i * self.stride) % n_total for i in range(self.img_num)]

    # ---------- main ---------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, List[str]]:
        scene_id = self.scene_ids[idx]
        img_dir  = os.path.join(
            self.root_dir, scene_id, "dslr", "resized_undistorted_images"
        )
        img_paths = sorted(
            glob.glob(os.path.join(img_dir, "*"))
        )
        if len(img_paths) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

        sel_idx  = self._sample_indices(len(img_paths))
        sel_imgs = [img_paths[i] for i in sel_idx]
        img_names = [os.path.basename(p) for p in sel_imgs]
        imgs     = self.tfm(sel_imgs)          # (M,3,H,W), float32, 0–1

        return imgs, scene_id, img_names