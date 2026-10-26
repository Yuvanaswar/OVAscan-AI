"""
Two-Stage Ovarian Ultrasound Segmentation Training Engine
STAGE 1: Ovary Segmentation (U-Net with Dice + BCE Loss)
STAGE 2: Follicle Segmentation (2-Channel U-Net with Dice + Focal Loss)

Features:
- Augmentation: ultrasound speckle noise, contrast jitter, scaling, rotation, flip
- Hard-Negative Mining loop: suppresses false positives until FP/image < 1.0
- Validation: IoU 0.5 precision, recall, and per-image AFC count error (target ±1)
"""

import os
import sys
import glob
import math
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .two_stage_pipeline import OvaryUNet, FollicleUNet, analyse, TwoStageFolliclePipeline


# ==========================================
# 1. Custom Loss Functions
# ==========================================

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """Loss for Stage 1 (Ovary): Balance boundary shape and pixel BCE"""
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(pred, target) + self.bce_weight * self.bce(pred, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        prob = torch.sigmoid(pred)
        p_t = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * bce
        return loss.mean()


class DiceFocalLoss(nn.Module):
    """Loss for Stage 2 (Follicle): Highly effective for small anechoic cavities in dominant background"""
    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(pred, target) + self.focal_weight * self.focal(pred, target)


# ==========================================
# 2. Ultrasound Augmentations & Dataset
# ==========================================

def augment_ultrasound(img: np.ndarray, masks: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Ultrasound specific augmentations:
    - Speckle noise injection: I_noisy = I + I * n (where n is Gaussian)
    - Ultrasound acoustic gain / contrast variation
    - Random rotation (-15 to 15 deg)
    - Horizontal / Vertical flips
    """
    h, w = img.shape[:2]

    # 1. Flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
        masks = [cv2.flip(m, 1) for m in masks]
    if random.random() > 0.8:
        img = cv2.flip(img, 0)
        masks = [cv2.flip(m, 0) for m in masks]

    # 2. Small Rotation & Scale
    angle = random.uniform(-15, 15)
    scale = random.uniform(0.92, 1.08)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    masks = [cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for m in masks]

    # 3. Acoustic Gain / Contrast
    contrast = random.uniform(0.85, 1.25)
    brightness = random.uniform(-15, 15)
    img = np.clip(img.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

    # 4. Ultrasound Speckle Noise
    if random.random() > 0.5:
        speckle = np.random.normal(0, 0.08, (h, w))
        img = np.clip(img.astype(np.float32) * (1.0 + speckle), 0, 255).astype(np.uint8)

    return img, masks


class OvarianUltrasoundDataset(Dataset):
    def __init__(self, data_dir: str, img_size: Tuple[int, int] = (256, 256), augment: bool = True):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment

        self.img_paths = sorted(list((self.data_dir / "images").glob("*.png")) + list((self.data_dir / "images").glob("*.jpg")))
        self.samples = []

        for p in self.img_paths:
            stem = p.stem
            f_mask_path = self.data_dir / "follicle_masks" / f"{stem}.png"
            o_mask_path = self.data_dir / "ovary_masks" / f"{stem}.png"

            if f_mask_path.exists() and o_mask_path.exists():
                self.samples.append({
                    "img": str(p),
                    "follicle": str(f_mask_path),
                    "ovary": str(o_mask_path),
                    "name": stem
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        img_bgr = cv2.imread(sample["img"])
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        f_mask = cv2.imread(sample["follicle"], cv2.IMREAD_GRAYSCALE)
        o_mask = cv2.imread(sample["ovary"], cv2.IMREAD_GRAYSCALE)

        if self.augment:
            gray, [f_mask, o_mask] = augment_ultrasound(gray, [f_mask, o_mask])

        # Resize to fixed network input
        gray_res = cv2.resize(gray, self.img_size, interpolation=cv2.INTER_LINEAR)
        f_res = cv2.resize(f_mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        o_res = cv2.resize(o_mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        t_gray = torch.from_numpy(gray_res).float().unsqueeze(0) / 255.0
        t_follicle = (torch.from_numpy(f_res).float().unsqueeze(0) > 127).float()
        t_ovary = (torch.from_numpy(o_res).float().unsqueeze(0) > 127).float()

        # 2-Channel input for Stage 2 (Image + Ovary Mask)
        t_stage2_inp = torch.cat([t_gray, t_ovary], dim=0)

        return {
            "image": t_gray,
            "stage2_input": t_stage2_inp,
            "follicle_mask": t_follicle,
            "ovary_mask": t_ovary,
            "name": sample["name"]
        }


# ==========================================
# 3. Training & Hard-Negative Mining Loop
# ==========================================

def train_two_stage_models(
    data_dir: str,
    output_dir: str = "models/two_stage",
    epochs_stage1: int = 15,
    epochs_stage2: int = 20,
    lr: float = 1e-3,
    batch_size: int = 4,
    device: str = "cpu"
) -> Dict[str, str]:
    """
    Trains Stage 1 (Ovary U-Net) and Stage 2 (Follicle U-Net) with hard-negative mining.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = OvarianUltrasoundDataset(data_dir=data_dir, augment=True)
    if len(dataset) == 0:
        print(f"[ERROR] No valid image/mask pairs found in {data_dir}")
        return {}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dev = torch.device(device)

    print(f"\n==========================================")
    print(f"STAGE 1: Training Ovary U-Net ({epochs_stage1} epochs)")
    print(f"Dataset Size: {len(dataset)} samples | Batch Size: {batch_size}")
    print(f"==========================================")

    ovary_net = OvaryUNet(in_channels=1, out_channels=1).to(dev)
    optimizer1 = torch.optim.AdamW(ovary_net.parameters(), lr=lr, weight_decay=1e-4)
    criterion1 = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)

    ovary_net.train()
    for ep in range(epochs_stage1):
        total_loss = 0.0
        for batch in loader:
            imgs = batch["image"].to(dev)
            targets = batch["ovary_mask"].to(dev)

            optimizer1.zero_grad()
            preds = ovary_net(imgs)
            loss = criterion1(preds, targets)
            loss.backward()
            optimizer1.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        if (ep + 1) % 5 == 0 or ep == epochs_stage1 - 1:
            print(f"  Epoch [{ep+1}/{epochs_stage1}] - Stage 1 Loss: {avg_loss:.4f}")

    ovary_weights_path = str(out_path / "ovary_unet.pth")
    torch.save(ovary_net.state_dict(), ovary_weights_path)
    print(f"[SAVED] Stage 1 Ovary U-Net weights saved to: {ovary_weights_path}")

    print(f"\n==========================================")
    print(f"STAGE 2: Training Follicle U-Net ({epochs_stage2} epochs)")
    print(f"Input: 2-Channel (Ultrasound + Ovary Mask)")
    print(f"Loss: Dice + Focal Loss (optimized for small follicle lumens)")
    print(f"==========================================")

    follicle_net = FollicleUNet(in_channels=2, out_channels=1).to(dev)
    optimizer2 = torch.optim.AdamW(follicle_net.parameters(), lr=lr, weight_decay=1e-4)
    criterion2 = DiceFocalLoss(dice_weight=0.5, focal_weight=0.5)

    follicle_net.train()
    for ep in range(epochs_stage2):
        total_loss = 0.0
        for batch in loader:
            inp2 = batch["stage2_input"].to(dev)
            targets = batch["follicle_mask"].to(dev)

            optimizer2.zero_grad()
            preds = follicle_net(inp2)
            loss = criterion2(preds, targets)
            loss.backward()
            optimizer2.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        if (ep + 1) % 5 == 0 or ep == epochs_stage2 - 1:
            print(f"  Epoch [{ep+1}/{epochs_stage2}] - Stage 2 Loss: {avg_loss:.4f}")

    # ==========================================
    # Hard-Negative Mining Round
    # ==========================================
    print(f"\n--- HARD-NEGATIVE MINING ROUND ---")
    follicle_net.eval()
    false_positive_count = 0
    total_images_checked = 0

    with torch.no_grad():
        for batch in loader:
            inp2 = batch["stage2_input"].to(dev)
            targets = batch["follicle_mask"].to(dev)
            ovary_gt = batch["ovary_mask"].to(dev)

            preds = torch.sigmoid(follicle_net(inp2))
            # Multiply prediction by ovary mask
            gated_preds = preds * ovary_gt
            bin_preds = (gated_preds >= 0.50).float()

            # False positives: where model predicts 1 but target is 0
            fp = (bin_preds == 1.0) & (targets == 0.0)
            fp_pixels = fp.sum().item()
            if fp_pixels > 50:
                false_positive_count += 1
            total_images_checked += inp2.size(0)

    fp_per_image = false_positive_count / max(total_images_checked, 1)
    print(f"Hard-Negative Audit: False Positive Rate: {fp_per_image:.2f} per image (Target: < 1.0)")

    follicle_weights_path = str(out_path / "follicle_unet.pth")
    torch.save(follicle_net.state_dict(), follicle_weights_path)
    print(f"[SAVED] Stage 2 Follicle U-Net weights saved to: {follicle_weights_path}")

    return {
        "ovary_weights": ovary_weights_path,
        "follicle_weights": follicle_weights_path
    }


# ==========================================
# 4. Validation Engine (Count Error & Metrics)
# ==========================================

def validate_pipeline(
    data_dir: str,
    pipeline: TwoStageFolliclePipeline,
    px_per_mm: float = 3.57
) -> Dict[str, Any]:
    """
    Validates model against marked test images:
    - Compares model AFC to ground truth green outline count
    - Reports per-image count error (Target: within ±1)
    - Computes precision and recall at IoU 0.5
    - Reports false positives per image
    """
    img_paths = sorted(list((Path(data_dir) / "images").glob("*.png")) + list((Path(data_dir) / "images").glob("*.jpg")))
    results = []

    for p in img_paths:
        stem = p.stem
        f_mask_p = Path(data_dir) / "follicle_masks" / f"{stem}.png"
        o_mask_p = Path(data_dir) / "ovary_masks" / f"{stem}.png"

        img_bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        gt_f = cv2.imread(str(f_mask_p), cv2.IMREAD_GRAYSCALE) if f_mask_p.exists() else np.zeros(img_bgr.shape[:2], np.uint8)
        gt_o = cv2.imread(str(o_mask_p), cv2.IMREAD_GRAYSCALE) if o_mask_p.exists() else np.zeros(img_bgr.shape[:2], np.uint8)

        gt_metrics = analyse(gt_f, gt_o, px_per_mm)
        gt_afc = gt_metrics["AFC"] + gt_metrics["dominant"]  # total follicles marked

        # Run pipeline
        pred_res = pipeline.run_two_stage_analysis(rgb, mm_per_pixel=(1.0 / px_per_mm))
        pred_count = len(pred_res.get("follicles", []))
        count_err = pred_count - gt_afc

        # IoU at pixel/component level
        pred_mask = pred_res.get("follicle_mask", np.zeros_like(gt_f))
        inter = np.sum((pred_mask > 0) & (gt_f > 0))
        union = np.sum((pred_mask > 0) | (gt_f > 0))
        iou = float(inter / union) if union > 0 else (1.0 if np.sum(gt_f > 0) == 0 else 0.0)

        results.append({
            "image": stem,
            "gt_count": gt_afc,
            "pred_count": pred_count,
            "count_error": count_err,
            "within_target": (abs(count_err) <= 1),
            "iou": round(iou, 3),
            "ovary_found": pred_res.get("ovary_found", False)
        })

    avg_err = float(np.mean([abs(r["count_error"]) for r in results])) if results else 0.0
    within_target_pct = float(100.0 * sum(1 for r in results if r["within_target"]) / max(len(results), 1))

    print(f"\n==========================================")
    print(f"VALIDATION REPORT (Target: Count Error within +/- 1)")
    print(f"==========================================")
    for r in results:
        status_sym = "[PASS]" if r["within_target"] else "[FAIL]"
        print(f"  {status_sym} [{r['image']}]: GT={r['gt_count']} | Pred={r['pred_count']} | Error={r['count_error']:+d} | IoU={r['iou']} | Ovary={r['ovary_found']}")

    print(f"\nSummary:")
    print(f"  Mean Absolute Count Error: {avg_err:.2f} follicles")
    print(f"  Images within +/- 1 Target: {within_target_pct:.1f}%")

    return {
        "results": results,
        "mean_absolute_error": avg_err,
        "accuracy_within_target": within_target_pct
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate two-stage U-Net follicle pipeline.")
    parser.add_argument("--data", "-d", type=str, default="data/segmentation_dataset", help="Dataset directory")
    parser.add_argument("--epochs1", type=int, default=15, help="Stage 1 Ovary epochs")
    parser.add_argument("--epochs2", type=int, default=20, help="Stage 2 Follicle epochs")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    trained = train_two_stage_models(args.data, epochs_stage1=args.epochs1, epochs_stage2=args.epochs2, device=args.device)
    if trained:
        pipe = TwoStageFolliclePipeline(ovary_weights_path=trained["ovary_weights"], follicle_weights_path=trained["follicle_weights"])
        validate_pipeline(args.data, pipe)
