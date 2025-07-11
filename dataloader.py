"""
dataloader_factory.py
─────────────────────
build_loader            : 构造单个 DataLoader
build_loader_with_aux   : 构造同时加载主/辅助图像的 DataLoader
build_train_val         : 同时返回 train_loader & val_loader

新增功能
────────
1. 打印每个 batch 采样到的 scene_id（可选，默认关闭）  
2. 通过 `verbose` / `train_verbose` / `val_verbose` 开关灵活控制
"""

import random
from typing import List, Tuple
from torch.utils.data import DataLoader, RandomSampler
from dataset import SceneDataset

# ─────────────────────────────────────────────────────────────────────────────
# 全局调试开关：改成 True 就会默认打印所有 loader 的 scene_id
# 也可以在 build_loader / build_loader_with_aux / build_train_val 里显式传参覆盖
# ─────────────────────────────────────────────────────────────────────────────
VERBOSE_DEFAULT = True


# -----------------------------------------------------------------------------
# 读取 split 文件
# -----------------------------------------------------------------------------
def _read_split(txt_path: str) -> List[str]:
    """读取 split.txt，每行一个 scene_id"""
    with open(txt_path, "r") as f:
        return [l.strip() for l in f if l.strip()]


# -----------------------------------------------------------------------------
# 构造单个 loader（只返回主视图）
# -----------------------------------------------------------------------------
def build_loader(
    root_dir: str,
    scene_ids: List[str],
    batch_scenes: int = 4,
    img_num: int = 4,
    stride: int = 3,
    shuffle: bool = True,
    num_workers: int = 4,
    verbose: bool = VERBOSE_DEFAULT,   # ★ 新增
) -> DataLoader:
    """
    每次迭代输出 (imgs, scene_id, img_names)
      imgs      : (img_num,3,H,W)
      scene_id  : str
      img_names : List[str]
    """
    dataset = SceneDataset(root_dir, scene_ids, img_num, stride)

    sampler = (
        RandomSampler(dataset, replacement=True, num_samples=batch_scenes)
        if shuffle
        else None
    )

    # 把原来的匿名 lambda 改成显式函数，顺便打印 scene_id
    def collate_fn(batch):
        imgs, sid, img_names = batch[0]
        if verbose:
            print(f"[DataLoader] chosen scene → {sid}", flush=True)
        return imgs, sid, img_names

    loader = DataLoader(
        dataset,
        batch_size=1,                # 1 个 scene / batch
        sampler=sampler,
        shuffle=False if sampler else shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


# -----------------------------------------------------------------------------
# 同时加载主视图 & 辅助视图
# -----------------------------------------------------------------------------
def build_loader_with_aux(
    root_dir: str,
    scene_ids: List[str],
    batch_scenes: int,
    img_num_main: int,
    img_num_aux: int,
    stride: int = 3,
    shuffle: bool = True,
    num_workers: int = 4,
    verbose: bool = VERBOSE_DEFAULT,   # ★ 新增
) -> DataLoader:
    """
    返回：
      imgs_main   : (N_main,3,H,W)
      imgs_aux    : (N_aux,3,H,W)
      scene_id    : str
      names_main  : List[str]
      names_aux   : List[str]
    N_main = img_num_main, N_aux = img_num_aux
    """
    total_imgs = img_num_main + img_num_aux
    dataset = SceneDataset(root_dir, scene_ids, total_imgs, stride)

    sampler = (
        RandomSampler(dataset, replacement=True, num_samples=batch_scenes)
        if shuffle
        else None
    )

    def collate_fn(batch):
        imgs, sid, names = batch[0]
        if verbose:
            print(f"[DataLoader] chosen scene → {sid}", flush=True)

        imgs_main  = imgs[:img_num_main]
        imgs_aux   = imgs[img_num_main:]
        names_main = names[:img_num_main]
        names_aux  = names[img_num_main:]
        return imgs_main, imgs_aux, sid, names_main, names_aux

    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False if sampler else shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


# -----------------------------------------------------------------------------
# 同时构造 train / val loader
# -----------------------------------------------------------------------------
def build_train_val(
    root_dir: str,
    train_split_txt: str,
    val_split_txt: str,
    train_batch_scenes: int,
    train_img_num: int,
    val_batch_scenes: int,
    val_img_num: int,
    train_img_aux: int = 0,
    stride: int = 3,
    num_workers: int = 4,
    # ▲ 新增两个开关，默认采用全局 VERBOSE_DEFAULT
    train_verbose: bool = VERBOSE_DEFAULT,
    val_verbose: bool = VERBOSE_DEFAULT,
) -> Tuple[DataLoader, DataLoader]:
    """
    根据 train_img_aux 选择不同的 loader 版本：
      • train_img_aux == 0 → 只主视图
      • train_img_aux  > 0 → 主 + 辅视图
    """
    train_ids = _read_split(train_split_txt)
    val_ids   = _read_split(val_split_txt)

    # ---------------- train loader ----------------
    if train_img_aux > 0:
        train_loader = build_loader_with_aux(
            root_dir,
            train_ids,
            batch_scenes=train_batch_scenes,
            img_num_main=train_img_num,
            img_num_aux=train_img_aux,
            stride=stride,
            shuffle=True,
            num_workers=num_workers,
            verbose=train_verbose,
        )
    else:
        train_loader = build_loader(
            root_dir,
            train_ids,
            batch_scenes=train_batch_scenes,
            img_num=train_img_num,
            stride=stride,
            shuffle=True,
            num_workers=num_workers,
            verbose=train_verbose,
        )

    # ---------------- val loader ------------------
    val_loader = build_loader(
        root_dir,
        val_ids,
        batch_scenes=val_batch_scenes,
        img_num=val_img_num,
        stride=stride,
        shuffle=True,
        num_workers=num_workers,
        verbose=val_verbose,
    )

    return train_loader, val_loader