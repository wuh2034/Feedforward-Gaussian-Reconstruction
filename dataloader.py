"""
dataloader_factory.py
─────────────────────
build_loader  : 构造单个 DataLoader
build_train_val : 同时返回 train_loader & val_loader
"""
import random
from typing import List, Tuple
from torch.utils.data import DataLoader, RandomSampler
from dataset import SceneDataset


def _read_split(txt_path: str) -> List[str]:
    """读取 split.txt，每行一个 scene_id"""
    with open(txt_path, "r") as f:
        return [l.strip() for l in f if l.strip()]


# ---------- 单个 loader --------------------------------------------------------
def build_loader(
    root_dir: str,
    scene_ids: List[str],
    batch_scenes: int = 4,
    img_num: int = 4,
    stride: int = 3,
    shuffle: bool = True,
    num_workers: int = 4,
):
    """
    每次迭代输出 (imgs, scene_id)
      imgs  : (img_num,3,H,W)
      scene_id : str
    """
    dataset = SceneDataset(root_dir, scene_ids, img_num, stride)

    sampler = (
        RandomSampler(
            dataset, replacement=True, num_samples=batch_scenes
        )
        if shuffle
        else None
    )

    loader = DataLoader(
        dataset,
        batch_size=1,                # 1 个 scene/批
        sampler=sampler,
        shuffle=False if sampler else shuffle,
        collate_fn=lambda x: x[0],   # 去掉 list 外壳
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


# ---------- train / val 两个 loader -------------------------------------------
def build_train_val(
    root_dir: str,
    train_split_txt: str,
    val_split_txt: str,
    train_batch_scenes: int,
    train_img_num: int,
    val_batch_scenes: int,
    val_img_num: int,
    stride: int = 3,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_ids = _read_split(train_split_txt)
    val_ids = _read_split(val_split_txt)

    train_loader = build_loader(
        root_dir,
        train_ids,
        batch_scenes=train_batch_scenes,
        img_num=train_img_num,
        stride=stride,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = build_loader(
        root_dir,
        val_ids,
        batch_scenes=val_batch_scenes,
        img_num=val_img_num,
        stride=stride,
        shuffle=True,
        num_workers=num_workers,
    )
    return train_loader, val_loader