"""
torch-side dataset, augmentation and DataLoader construction (Approach 1, 3, 4).

Kept separate from data.py so that Approach 2, which needs no deep-learning
stack, can import the split logic without importing torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

import config
import data as data_module


class LeafDataset(Dataset):
    """
    Reads images listed in one split of data/splits.csv.

    The label -> index mapping comes from data/classes.json, not from the
    directory order, so index 7 means the same disease in every script and in
    every rerun.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        classes: list[str],
        root: Path = config.RAW_DIR,
        transform=None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.root = Path(root)
        self.transform = transform
        self.targets = np.array(
            [self.class_to_idx[lbl] for lbl in self.df["label"]], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        relpath = self.df.loc[idx, "relpath"]
        image = Image.open(self.root / relpath).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[idx])

    def relpath(self, idx: int) -> str:
        """Path of one sample -- needed when saving Grad-CAM overlays."""
        return str(self.df.loc[idx, "relpath"])


def build_transforms(img_size: int = config.IMG_SIZE, train: bool = False):
    """
    Augmentation policy.

    Flips and rotations are safe here because a leaf photograph has no
    canonical orientation. Colour jitter is deliberately mild: hue and
    saturation carry the disease signal (chlorosis, necrotic browning), so
    distorting them heavily would erase the very evidence the model needs.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])

    # Evaluation: resize slightly larger then centre-crop, the standard
    # deterministic protocol for ImageNet-pretrained backbones.
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def make_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    img_size: int = config.IMG_SIZE,
    num_workers: int = config.NUM_WORKERS,
    max_per_class: int | None = None,
    balanced_sampler: bool = False,
) -> tuple[dict[str, DataLoader], list[str], dict[str, LeafDataset]]:
    """
    Build train/val/test loaders from the frozen split.

    balanced_sampler oversamples rare classes during training. It is off by
    default: with macro-F1 as the reported metric plus class-weighted loss the
    imbalance is already accounted for, and oversampling makes the epoch
    length harder to interpret.
    """
    classes = config.load_classes()
    loaders: dict[str, DataLoader] = {}
    datasets: dict[str, LeafDataset] = {}
    # pin_memory only helps when copying to a GPU; on CPU it just warns.
    pin_memory = torch.cuda.is_available()

    for split in ("train", "val", "test"):
        df = data_module.get_split(split)
        if max_per_class:
            df = data_module.subsample_per_class(df, max_per_class)

        is_train = split == "train"
        dataset = LeafDataset(df, classes, transform=build_transforms(img_size, is_train))
        datasets[split] = dataset

        if is_train and balanced_sampler:
            counts = np.bincount(dataset.targets, minlength=len(classes))
            weights = 1.0 / np.maximum(counts[dataset.targets], 1)
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(dataset),
                replacement=True,
            )
            loaders[split] = DataLoader(
                dataset, batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
            )
        else:
            loaders[split] = DataLoader(
                dataset, batch_size=batch_size, shuffle=is_train,
                num_workers=num_workers, pin_memory=pin_memory, drop_last=is_train,
            )

    return loaders, classes, datasets


def class_weights_from_dataset(dataset: LeafDataset) -> torch.Tensor:
    """
    Inverse-frequency weights, normalised to mean 1 so the loss scale stays
    comparable to the unweighted case.
    """
    counts = np.bincount(dataset.targets, minlength=len(dataset.classes)).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Undo the ImageNet normalisation and return an HxWx3 array in [0, 1],
    so a Grad-CAM heatmap can be drawn over the original-looking photo.
    """
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    img = (tensor.detach().cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()
