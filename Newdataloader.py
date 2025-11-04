from typing import List, Tuple, Sequence
from torch.utils.data import DataLoader
from dataset import SceneDataset

VERBOSE_DEFAULT = False

def _read_split(txt_path: str) -> List[str]:
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

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
        images, scene_id, names = batch[0]
        if verbose:
            print(f"[DataLoader] Scene: {scene_id}", flush=True)
        return images, scene_id, names

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

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
    total_images = img_num_main + img_num_aux
    dataset = SceneDataset(root_dir, list(scene_ids), total_images, stride)

    def collate_fn(batch):
        images, scene_id, names = batch[0]
        if verbose:
            print(f"[DataLoader] Scene: {scene_id}", flush=True)

        main_images = images[:img_num_main]
        aux_images = images[img_num_main:]
        main_names = names[:img_num_main]
        aux_names = names[img_num_main:]

        return main_images, aux_images, scene_id, main_names, aux_names

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

def build_train_val(
    root_dir: str,
    train_split_txt: str,
    val_split_txt: str,
    train_scene_subset: Sequence[str] | None,
    val_scene_subset: Sequence[str] | None,
    train_img_num: int,
    val_img_num: int,
    train_img_aux: int = 0,
    val_img_aux: int = 0,
    stride: int = 3,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    all_train_ids = _read_split(train_split_txt)
    train_ids = train_scene_subset or all_train_ids

    if train_img_aux > 0:
        train_loader = build_loader_with_aux(
            root_dir,
            train_ids,
            img_num_main=train_img_num,
            img_num_aux=train_img_aux,
            stride=stride,
            shuffle=True,
            num_workers=num_workers,
        )
    else:
        train_loader = build_loader(
            root_dir,
            train_ids,
            img_num=train_img_num,
            stride=stride,
            shuffle=True,
            num_workers=num_workers,
        )

    all_val_ids = _read_split(val_split_txt)
    val_ids = val_scene_subset or all_val_ids

    if val_img_aux > 0:
        val_loader = build_loader_with_aux(
            root_dir,
            val_ids,
            img_num_main=val_img_num,
            img_num_aux=val_img_aux,
            stride=stride,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        val_loader = build_loader(
            root_dir,
            val_ids,
            img_num=val_img_num,
            stride=stride,
            shuffle=False,
            num_workers=num_workers,
        )

    return train_loader, val_loader