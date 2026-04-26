"""
dataset.py
----------
PyTorch Dataset and DataLoader factory for the disaster triage project.
Supports both single-task (type only) and hierarchical (type + severity) modes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224, augment: bool = False, aug_config: dict = None):
    aug_config = aug_config or {}

    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if not augment:
        return transforms.Compose(base)

    aug_transforms = [transforms.Resize((image_size, image_size))]

    if aug_config.get("random_crop"):
        aug_transforms += [transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0))]
    if aug_config.get("horizontal_flip"):
        aug_transforms.append(transforms.RandomHorizontalFlip())
    if aug_config.get("rotation", 0):
        aug_transforms.append(transforms.RandomRotation(aug_config["rotation"]))
    if aug_config.get("color_jitter"):
        aug_transforms.append(transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
        ))

    aug_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return transforms.Compose(aug_transforms)


class DisasterDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, label_mode: str = "combined"):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.label_mode = label_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.label_mode == "type":
            return image, int(row["disaster_idx"])
        elif self.label_mode == "severity":
            return image, int(row["severity_idx"])
        elif self.label_mode == "combined":
            return image, int(row["combined_idx"])
        elif self.label_mode == "both":
            return image, int(row["disaster_idx"]), int(row["severity_idx"])
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")


def build_dataloaders(
    labels_csv: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.70,
    val_split: float = 0.15,
    label_mode: str = "combined",
    aug_config: dict = None,
    seed: int = 42,
):
    df = pd.read_csv(labels_csv)

    stratify_col = "combined_label"
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_split),
        stratify=df[stratify_col], random_state=seed
    )
    relative_val = val_split / (1 - train_split)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val),
        stratify=temp_df[stratify_col], random_state=seed
    )

    train_tf = get_transforms(image_size, augment=True,  aug_config=aug_config)
    val_tf   = get_transforms(image_size, augment=False)
    test_tf  = get_transforms(image_size, augment=False)

    train_ds = DisasterDataset(train_df, transform=train_tf, label_mode=label_mode)
    val_ds   = DisasterDataset(val_df,   transform=val_tf,   label_mode=label_mode)
    test_ds  = DisasterDataset(test_df,  transform=test_tf,  label_mode=label_mode)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    class_info = {
        "disaster_types": ["flood", "fire", "earthquake", "traffic_incident", "non_disaster"],
        "severity_levels": ["mild", "moderate", "severe"],
        "combined_classes": sorted(df["combined_label"].unique().tolist()),
        "num_classes": {
            "type": 5,
            "severity": 3,
            "combined": df["combined_idx"].nunique(),
        },
        "split_sizes": {
            "train": len(train_df),
            "val":   len(val_df),
            "test":  len(test_df),
        }
    }

    return train_loader, val_loader, test_loader, class_info