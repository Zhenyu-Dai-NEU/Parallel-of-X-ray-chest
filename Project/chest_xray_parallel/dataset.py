"""
PyTorch Dataset for NIH ChestX-ray14.
Supports multi-label classification with 14 thoracic disease labels.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# 14 disease labels in the dataset
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]


def get_transforms(split="train", img_size=224):
    """Get image transforms for train/val/test splits."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


class ChestXrayDataset(Dataset):
    """NIH ChestX-ray14 dataset for multi-label classification."""

    def __init__(self, csv_path, image_dir, split="train",
                 split_file=None, transform=None, img_size=224, use_processed=False):
        """
        Args:
            csv_path: Path to Data_Entry_2017_v2020.csv
            image_dir: Root directory containing images
            split: 'train', 'val', or 'test'
            split_file: Path to train_val_list.txt or test_list.txt
            transform: Optional custom transforms
            img_size: Target image size
            use_processed: Whether to use preprocessed images
        """
        self.image_dir = image_dir
        self.use_processed = use_processed
        self.transform = transform or get_transforms(split, img_size)

        # Load labels
        df = pd.read_csv(csv_path)

        # Filter by split if split file provided
        if split_file and os.path.exists(split_file):
            with open(split_file, "r") as f:
                split_images = set(f.read().strip().split("\n"))
            df = df[df["Image Index"].isin(split_images)]

        # If doing train/val split from train_val_list, split 80/20
        if split in ("train", "val") and split_file and "train_val" in split_file:
            np.random.seed(42)
            indices = np.random.permutation(len(df))
            split_idx = int(0.8 * len(df))
            if split == "train":
                df = df.iloc[indices[:split_idx]]
            else:
                df = df.iloc[indices[split_idx:]]

        self.image_names = df["Image Index"].tolist()
        self.labels_raw = df["Finding Labels"].tolist()

        # Encode labels as multi-hot vectors
        self.labels = []
        for label_str in self.labels_raw:
            multi_hot = np.zeros(len(DISEASE_LABELS), dtype=np.float32)
            for disease in label_str.split("|"):
                disease = disease.strip()
                if disease in DISEASE_LABELS:
                    idx = DISEASE_LABELS.index(disease)
                    multi_hot[idx] = 1.0
            self.labels.append(multi_hot)

        print(f"[{split.upper()}] Loaded {len(self)} images")

    def __len__(self):
        return len(self.image_names)

    def _find_image(self, img_name):
        """Locate image file across subdirectories."""
        if self.use_processed:
            path = os.path.join(self.image_dir, img_name)
            if os.path.exists(path):
                return path

        # Direct path
        direct = os.path.join(self.image_dir, img_name)
        if os.path.exists(direct):
            return direct

        # Search subdirectories
        for subdir in sorted(Path(self.image_dir).iterdir()):
            if subdir.is_dir() and subdir.name.startswith("images_"):
                for candidate in [subdir / "images" / img_name, subdir / img_name]:
                    if candidate.exists():
                        return str(candidate)
        return None

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        img_path = self._find_image(img_name)
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_name}")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label


def get_dataloaders(data_dir, csv_path, batch_size=32, num_workers=4,
                    img_size=224, use_processed=False, distributed=False):
    """Create train/val/test DataLoaders."""
    train_val_list = os.path.join(data_dir, "train_val_list.txt")
    test_list = os.path.join(data_dir, "test_list.txt")

    image_dir = os.path.join(data_dir, "processed") if use_processed else data_dir

    train_dataset = ChestXrayDataset(
        csv_path, image_dir, split="train",
        split_file=train_val_list, img_size=img_size, use_processed=use_processed,
    )
    val_dataset = ChestXrayDataset(
        csv_path, image_dir, split="val",
        split_file=train_val_list, img_size=img_size, use_processed=use_processed,
    )
    test_dataset = ChestXrayDataset(
        csv_path, image_dir, split="test",
        split_file=test_list, img_size=img_size, use_processed=use_processed,
    )

    # For distributed training, use DistributedSampler
    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_sampler
