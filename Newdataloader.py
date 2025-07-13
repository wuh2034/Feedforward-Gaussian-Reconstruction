"""
dataloader_factory.py
─────────────────────
• build_loader / build_loader_with_aux
• build_train_val
"""
from typing import List, Tuple, Sequence
from torch.utils.data import DataLoader
from dataset import SceneDataset

VERBOSE_DEFAULT = False   # ← 默认静默


# -------------------------------------------------------------------------
# split 读取
# -------------------------------------------------------------------------
def _read_split(txt_path: str) -> List[str]:
    with open(txt_path, "r") as f:
        return [l.strip() for l in f if l.strip()]


# -------------------------------------------------------------------------
# dataset → loader（只主视图）
# -------------------------------------------------------------------------
def build_loader(
    root_dir: str,
    scene_ids: Sequence[str],
    img_num: int,
    stride: int = 3,
    shuffle: bool = True,
    num_workers: int = 4,
    verbose: bool = VERBOSE_DEFAULT,
) -> DataLoader:

    dataset = SceneDataset(root_dir, list(scene_ids), img_num, stride)

    def collate_fn(batch):
        imgs, sid, names = batch[0]
        if verbose:
            print(f"[DataLoader] scene → {sid}", flush=True)
        return imgs, sid, names

    return DataLoader(
        dataset,
        batch_size=1,          # 一个 batch = 一个 scene
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


# -------------------------------------------------------------------------
# dataset → loader（主 + 辅视图）
# -------------------------------------------------------------------------
def build_loader_with_aux(
    root_dir: str,
    scene_ids: Sequence[str],
    img_num_main: int,
    img_num_aux: int,
    stride: int = 3,
    shuffle: bool = True,
    num_workers: int = 4,
    verbose: bool = VERBOSE_DEFAULT,
) -> DataLoader:

    total_imgs = img_num_main + img_num_aux
    dataset    = SceneDataset(root_dir, list(scene_ids), total_imgs, stride)

    def collate_fn(batch):
        imgs, sid, names = batch[0]
        if verbose:
            print(f"[DataLoader] scene → {sid}", flush=True)

        return (
            imgs[:img_num_main],          # imgs_main
            imgs[img_num_main:],          # imgs_aux
            sid,
            names[:img_num_main],
            names[img_num_main:],
        )

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


# -------------------------------------------------------------------------
# 构造 train & val loader
# -------------------------------------------------------------------------
def build_train_val(
    root_dir: str,
    train_split_txt: str,
    val_split_txt: str,
    train_scene_subset: Sequence[str] | None,
    val_scene_subset: Sequence[str] | None,
    train_img_num: int,
    val_img_num: int,
    train_img_aux: int = 0,
    val_img_aux: int   = 0,
    stride: int = 3,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    # ---------------- train ----------------
    train_ids_all = _read_split(train_split_txt)
    train_ids = train_scene_subset if train_scene_subset is not None else train_ids_all

    if train_img_aux > 0:
        train_loader = build_loader_with_aux(
            root_dir, train_ids,
            img_num_main=train_img_num,
            img_num_aux=train_img_aux,
            stride=stride, shuffle=True,
            num_workers=num_workers,
        )
    else:
        train_loader = build_loader(
            root_dir, train_ids,
            img_num=train_img_num,
            stride=stride, shuffle=True,
            num_workers=num_workers,
        )

    # ---------------- val ------------------
    val_ids_all = _read_split(val_split_txt)
    val_ids = val_scene_subset if val_scene_subset is not None else val_ids_all

    if val_img_aux > 0:
        val_loader = build_loader_with_aux(
            root_dir, val_ids,
            img_num_main=val_img_num,
            img_num_aux=val_img_aux,
            stride=stride, shuffle=False,
            num_workers=num_workers,
        )
    else:
        val_loader = build_loader(
            root_dir, val_ids,
            img_num=val_img_num,
            stride=stride, shuffle=False,
            num_workers=num_workers,
        )

    return train_loader, val_loader