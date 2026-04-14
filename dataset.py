# dataset.py — Dataset class and image transforms for Kvasir Capsule

import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class KvasirCapsuleDataset(Dataset):
    """
    PyTorch Dataset for the Kvasir-Capsule dataset.

    Expects a CSV with at least two columns:
        - ``filename`` : image file name (e.g. ``abc123.jpg``)
        - ``label``    : class string (e.g. ``Polyp``)

    The constructor probes several candidate image directories so it works
    regardless of whether images live in ``data_dir/train/``, ``data_dir/images/``,
    or directly in ``data_dir``.
    """

    _CANDIDATE_SUBDIRS = ['train', 'Train', 'images', 'Images', '']

    def __init__(
        self,
        data_dir: str,
        csv_file: str,
        transform=None,
        split: str = 'train',
        split_ratio: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.transform = transform

        # Load CSV 
        csv_path = csv_file if os.path.isabs(csv_file) else os.path.join(data_dir, csv_file)
        if not os.path.exists(csv_path):
            self._print_dir(data_dir)
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        print(f"CSV loaded: {len(self.df)} rows | columns: {list(self.df.columns)}")

        # Class mapping 
        self.classes      = sorted(self.df['label'].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Locate image directory 
        img_dir = self._find_image_dir(data_dir)

        # Build path / label lists 
        missing = []
        paths, labels = [], []

        for _, row in self.df.iterrows():
            img_path = os.path.join(img_dir, row['filename'])
            if os.path.exists(img_path):
                paths.append(img_path)
                labels.append(self.class_to_idx[row['label']])
            else:
                missing.append(row['filename'])

        if missing:
            print(f"Warning: {len(missing)} images not found. First few: {missing[:5]}")

        if not paths:
            self._debug_missing(img_dir)
            raise ValueError("No images could be loaded from the dataset.")

        # Train / val split 
        data = list(zip(paths, labels))
        random.seed(seed)
        random.shuffle(data)
        split_idx = int(len(data) * split_ratio)
        data = data[:split_idx] if split == 'train' else data[split_idx:]

        self.image_paths, self.labels = zip(*data) if data else ([], [])

        print(f"{split.capitalize()} split: {len(self.image_paths)} images")
        print(f"Classes: {self.classes}")
        if self.labels:
            uniq, counts = np.unique(self.labels, return_counts=True)
            print(f"Class distribution: {dict(zip(uniq, counts))}")

    # Internal helpers 

    def _find_image_dir(self, data_dir: str) -> str:
        for sub in self._CANDIDATE_SUBDIRS:
            candidate = os.path.join(data_dir, sub) if sub else data_dir
            if os.path.isdir(candidate):
                imgs = [f for f in os.listdir(candidate)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if imgs:
                    print(f"Image directory: {candidate}  (sample: {imgs[:3]})")
                    return candidate
        raise FileNotFoundError(
            f"Could not find an image directory under '{data_dir}'. "
            f"Tried: {self._CANDIDATE_SUBDIRS}"
        )

    @staticmethod
    def _print_dir(path: str) -> None:
        if os.path.isdir(path):
            print(f"Contents of {path}: {os.listdir(path)}")

    def _debug_missing(self, img_dir: str) -> None:
        print("First filenames in CSV:", self.df['filename'].head().tolist())
        print("First files in image dir:",
              [f for f in os.listdir(img_dir)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:10])

    # Dataset interface 

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label    = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as exc:
            print(f"Error loading {img_path}: {exc}")
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)
        return image, label


# Transforms 

def get_transforms(img_size: int = 224):
    """Return (train_transform, val_transform) for the given image size."""
    _mean = [0.485, 0.456, 0.406]
    _std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1,
                               saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
    ])

    return train_transform, val_transform