from __future__ import annotations
import os
from typing import Dict, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

OPTIMIZER_TYPES = ["Adam", "SGD", "AdamW", "RMSprop", "Adagrad"]
LOSS_TYPES = ["CrossEntropyLoss", "NLLLoss", "BCEWithLogitsLoss"]
VAL_METRIC_CHOICES = [
    "Best Validation Accuracy",
    "Best Validation F1",
    "Best Validation Loss",
    "Best Train Accuracy",
    "Best Train F1",
    "Best Train Loss",
    "Every Epoch",
]
WEIGHT_MODES = ["Auto-balance (Inverse Frequency)", "Manual (comma-separated)"]

IMAGEFOLDER_WARNING = (
    "FORMAT REQUIREMENT FOR ImageFolder:\n"
    "The dataset directory must contain a subfolder for each class:\n\n"
    "  root_dir/\n"
    "  ├── class_a/\n"
    "  │   ├── img01.jpg\n"
    "  │   └── img02.png\n"
    "  └── class_b/\n"
    "      ├── img03.jpg\n"
    "      └── img04.png\n\n"
    "Subfolder names will automatically be used as class labels."
)

CSV_FOLDER_WARNING = (
    "FORMAT REQUIREMENT FOR Folder + CSV:\n"
    "1. An images directory containing all raw image files.\n"
    "2. A CSV file containing AT LEAST these EXACT column names:\n"
    "   id, img_name, label\n\n"
    "   • 'id': sample identifier (e.g. 1, 2, 3...)\n"
    "   • 'img_name': image filename (e.g. 'cat_01.jpg') relative to images folder\n"
    "   • 'label': class name string or class integer\n\n"
    "Example CSV header and rows:\n"
    "   id,img_name,label\n"
    "   1,img001.jpg,cat\n"
    "   2,img002.jpg,dog"
)

class CSVImageDataset(Dataset):

    #CSV Must have'id', 'img_name', 'label'

    def __init__(
        self,
        images_dir: str,
        csv_path: str,
        transform=None,
        class_to_idx: Optional[Dict[str, int]] = None,
        in_channels: int = 3,
    ):
        self.images_dir = str(images_dir).strip()
        self.csv_path = str(csv_path).strip()
        self.transform = transform
        self.in_channels = in_channels

        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Images directory not found:\n{self.images_dir}")
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV file not found:\n{self.csv_path}")

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as exc:
            raise ValueError(f"Failed to read CSV file ({self.csv_path}):\n{exc}") from exc

        # Strict validation of required column names: 'id', 'img_name', 'label'
        required_cols = {"id", "img_name", "label"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV file '{os.path.basename(self.csv_path)}' is missing required column(s): "
                f"{', '.join(sorted(missing))}.\n\n"
                f"The CSV MUST have columns: 'id', 'img_name', 'label'.\n"
                f"Found columns in file: {list(df.columns)}"
            )

        df = df.dropna(subset=["img_name", "label"]).copy()
        if len(df) == 0:
            raise ValueError("CSV file contains no valid rows with non-empty 'img_name' and 'label'.")

        df["img_name"] = df["img_name"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip()

        if class_to_idx is None:
            unique_classes = sorted(df["label"].unique().tolist())
            self.classes = unique_classes
            self.class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = sorted(list(class_to_idx.keys()), key=lambda k: class_to_idx[k])

        self.df = df
        self.targets = [self.class_to_idx[lbl] for lbl in df["label"]]
        self.img_names = df["img_name"].tolist()

    def __len__(self) -> int:
        return len(self.img_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_name = self.img_names[idx]
        img_path = os.path.join(self.images_dir, img_name)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(
                f"Image file '{img_name}' listed in CSV row {idx+1} not found in:\n{self.images_dir}"
            )

        img = Image.open(img_path)
        if self.in_channels == 1:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        target = self.targets[idx]
        return img, target


def create_image_folder_dataset(root_dir: str, in_channels: int = 3, transform=None) -> datasets.ImageFolder:
    root_dir = str(root_dir).strip()
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"ImageFolder directory does not exist:\n{root_dir}")

    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if not subdirs:
        raise ValueError(
            f"No class subfolders found inside directory:\n{root_dir}\n\n"
            f"ImageFolder requires a directory containing subfolders for each class.\n"
            f"Example:\n"
            f"  {root_dir}/class_a/\n"
            f"  {root_dir}/class_b/"
        )

    def _loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("L") if in_channels == 1 else img.convert("RGB")

    return datasets.ImageFolder(root_dir, transform=transform, loader=_loader)


def get_dataset_transform(in_channels: int, input_height: int, input_width: int,
                           aug_hflip: bool = False, aug_crop: bool = False,
                           aug_color_jitter: bool = False):
    t_list = [transforms.Resize((max(1, input_height), max(1, input_width)))]
    #Data Augmentation is Optional
    # Data augmentation (training only)
    if aug_hflip:
        t_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if aug_crop:
        pad = max(4, min(input_height, input_width) // 8)
        t_list.append(transforms.RandomCrop((input_height, input_width), padding=pad))
    if aug_color_jitter:
        t_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))

    t_list.append(transforms.ToTensor())
    if in_channels == 1:
        t_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        t_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(t_list)
