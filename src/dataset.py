"""
PyTorch Dataset and DataLoader for 3-Class Ovarian Ultrasound AI.
Supports 3 Clinical Classes:
  - Class 0: Normal Ovary (NORMAL / Non-PCOS)
  - Class 1: PCOS (Polycystic Ovary Syndrome)
  - Class 2: Dominant Follicle (DF / Ovulatory Follicle)
Supports loading from structured folders ('real time data' and 'data') and split CSVs.
"""

from pathlib import Path
from typing import Tuple, Dict, Optional, List
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import letterbox_pad_to_square, get_train_transform, get_eval_transform

CLASS_MAP_3 = {
    # Class 0: Normal Ovary
    "normal": 0,
    "non-pcos": 0,
    "normal ovary": 0,
    "normal_preservation": 0,
    "eps": 0,
    # Class 1: PCOS
    "pcos": 1,
    "pco": 1,
    # Class 2: Dominant Follicle
    "df": 2,
    "dominant_follicle": 2,
    "dominant follicle": 2
}

CLASS_NAMES = ["Normal Ovary", "PCOS", "Dominant Follicle"]

class PCOSUltrasoundDataset(Dataset):
    def __init__(self, data_split_dir: Path, csv_file: Optional[Path] = None, transform=None):
        self.data_dir = Path(data_split_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int, int]] = []

        if csv_file and Path(csv_file).exists():
            df = pd.read_csv(csv_file)
            for _, r in df.iterrows():
                # Check relative_path or construct from folder
                p = Path(r["relative_path"]) if "relative_path" in r else self.data_dir / r["label"] / r["imagePath"]
                raw_lbl = str(r["label"]).strip().lower()
                label_idx = CLASS_MAP_3.get(raw_lbl, 0)
                self.samples.append((p, label_idx, r.get("group", -1)))
        else:
            # Load directly from folder structure
            if self.data_dir.exists():
                for subfolder in self.data_dir.iterdir():
                    if subfolder.is_dir():
                        folder_name_lower = subfolder.name.strip().lower()
                        if folder_name_lower in CLASS_MAP_3:
                            label_idx = CLASS_MAP_3[folder_name_lower]
                            for img_p in subfolder.glob("*.*"):
                                if img_p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                                    self.samples.append((img_p, label_idx, -1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, group = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = letterbox_pad_to_square(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), group

def create_dataloaders(base_data_dir: Path, batch_size: int = 16, num_workers: int = 0):
    base = Path(base_data_dir)
    
    # Check if 'real time data' or 'data' subdirectories exist
    train_dir = base / "train"
    val_dir = base / "val" if (base / "val").exists() else base / "validation"
    test_dir = base / "test"

    train_ds = PCOSUltrasoundDataset(train_dir, base / "train.csv" if (base / "train.csv").exists() else None, transform=get_train_transform())
    val_ds = PCOSUltrasoundDataset(val_dir, base / "val.csv" if (base / "val.csv").exists() else None, transform=get_eval_transform())
    test_ds = PCOSUltrasoundDataset(test_dir, base / "test.csv" if (base / "test.csv").exists() else None, transform=get_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

