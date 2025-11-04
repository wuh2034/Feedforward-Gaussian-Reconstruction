import random
from typing import List, Tuple
from torch.utils.data import DataLoader, RandomSampler
from dataset import SceneDataset

# Global default verbosity flag (print scene_id each batch if True)
VERBOSE_DEFAULT = True

# Read scene split file and return a list of scene IDs
def _read_split(txt_path: str) -> List[str]:
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# Build DataLoader for main views only
def build_loader(
    root_dir: str,
    scene_ids: List[str],
    batch_scenes: int = 4,
    img_num: int = 4,
    stride: int = 3,
    shuffle: bool = True,
    num_workers: int = 4,
    verbose: bool = VERBOSE_DEFAULT,
) -> DataLoader:
    """
    Returns:
      imgs      : Tensor of shape (img_num, 3, H, W)
      scene_id  : str
      img_names : List[str]
    """
    dataset = SceneDataset(root_dir, scene_ids, img_num, stride)
    sampler = (
        RandomSampler(dataset, replacement=True, num_samples=batch_scenes)
        if shuffle else None
    )

    def collate_fn(batch):
        imgs, sid, img_names = batch[0]
        if verbose:
            print(f"[DataLoader] selected scene -> {sid}", flush=True)
        return imgs, sid, img_names

    return DataLoader(
        dataset,
        batch_size=1,                   # one scene per batch
        sampler=sampler,
        shuffle=False if sampler else shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

# Build DataLoader for main and auxiliary views
def build_loader_with_aux(
    root_dir: str,
    scene_ids: List[str],
    batch_scenes: int,
    img_num_main: int,
    img_num_aux: int,
    stride: int = 3,
    shuffle: bool = True,
    num_workers: int = 4,
    verbose: bool = VERBOSE_DEFAULT,
) -> DataLoader:
    """
    Returns:
      imgs_main   : (img_num_main, 3, H, W)
      imgs_aux    : (img_num_aux,  3, H, W)
      scene_id    : str
      names_main  : List[str]
      names_aux   : List[str]
    """
    total_imgs = img_num_main + img_num_aux
    dataset = SceneDataset(root_dir, scene_ids, total_imgs, stride)
    sampler = (
        RandomSampler(dataset, replacement=True, num_samples=batch_scenes)
        if shuffle else None
    )

    def collate_fn(batch):
        imgs, sid, names = batch[0]
        if verbose:
            print(f"[DataLoader] selected scene -> {sid}", flush=True)
        imgs_main = imgs[:img_num_main]
        imgs_aux = imgs[img_num_main:]
        names_main = names[:img_num_main]
        names_aux = names[img_num_main:]
        return imgs_main, imgs_aux, sid, names_main, names_aux

    return DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False if sampler else shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

# Build both training and validation DataLoaders
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
    train_verbose: bool = VERBOSE_DEFAULT,
    val_verbose: bool = VERBOSE_DEFAULT,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates loaders based on whether auxiliary images are used:
      • train_img_aux == 0 : main views only
      • train_img_aux  > 0 : main + auxiliary views

    Returns (train_loader, val_loader)
    """
    train_ids = _read_split(train_split_txt)
    val_ids = _read_split(val_split_txt)

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

    val_loader = build_loader(
        root_dir,
        val_ids,
        batch_scenes=val_batch_scenes,
        img_num=val_img_num,
        stride=stride,
        shuffle=False,
        num_workers=num_workers,
        verbose=val_verbose,
    )

    return train_loader, val_loader