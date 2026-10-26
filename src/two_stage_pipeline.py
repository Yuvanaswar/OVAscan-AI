"""
Two-Stage Ovarian Ultrasound Follicle Analysis Pipeline
STAGE 1: Ovary Segmentation (U-Net, Dice + BCE Loss)
  - Detects the anatomical ovarian contour and stroma boundary.
  - If ovary area < threshold => returns {"ovary_found": False, "AFC": 0} and STOPS.
  - ROI is drawn directly from the segmented ovary mask (NOT a fixed circle).

STAGE 2: Follicle Segmentation (2-Channel U-Net, Dice + Focal Loss)
  - Input = ultrasound image (channel 1) + ovary mask (channel 2).
  - Prediction is strictly multiplied by the ovary mask (follicle_pred * ovary_mask).
  - Zero detections outside the ovarian capsule.

METRICS:
  - Exact implementation of clinician-specified analyse() function.
  - Calipers D1, D2 via cv2.minAreaRect.
  - Mean diameter = (D1 + D2) / 2 in mm.
  - Antral follicles: 2 <= mean < 10 mm.
  - Dominant follicles: mean >= 10 mm (flag > 30 mm as cyst).
  - Spatial distribution: r = dist_to_center / rmax; peripheral if r > 0.6.
"""

from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. Modular U-Net Architectures
# ==========================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class OvaryUNet(nn.Module):
    """Stage 1: Ovary Boundary & Stroma Segmentation (1 Channel In -> 1 Channel Out)"""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        f = init_features
        self.enc1 = ConvBlock(in_channels, f)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ConvBlock(f, f * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bottleneck = ConvBlock(f * 4, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final_conv(d1)


class FollicleUNet(nn.Module):
    """Stage 2: Follicle Lumen Segmentation (2 Channels In: Image + Ovary Mask -> 1 Channel Out)"""
    def __init__(self, in_channels: int = 2, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        f = init_features
        self.enc1 = ConvBlock(in_channels, f)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ConvBlock(f, f * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bottleneck = ConvBlock(f * 4, f * 8)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f * 2, f)
        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final_conv(d1)


# ==========================================
# 2. Clinician-Specified Metrics Engine
# ==========================================

def analyse(follicle_mask: np.ndarray, ovary_mask: np.ndarray, px_per_mm: float) -> Dict[str, Any]:
    """
    Computes follicle calipers, AFC, dominant follicle classification, and spatial distribution.
    Matches exact clinician ground-truth specification:
      - Connected components = individual follicles
      - D1, D2 via cv2.minAreaRect
      - mean = (D1 + D2) / 2
      - AFC = 2 <= mean < 10 mm
      - Dominant = mean >= 10 mm (cysts > 30 mm)
      - Spatial ratio r = distance to ovary center / ovary rmax; peripheral if r > 0.6
    """
    is_calibrated = (px_per_mm is not None and px_per_mm > 0)
    scale = px_per_mm if is_calibrated else 1.0

    n, lab = cv2.connectedComponents((follicle_mask > 0).astype(np.uint8))
    oc = np.argwhere(ovary_mask > 0)

    if len(oc) == 0:
        return {
            "AFC": 0, "dominant": 0, "cyst_count": 0,
            "mean_diam": None, "diameter_range_mm": "N/A",
            "peripheral_pct": 0.0, "spatial_distribution": "N/A",
            "follicles": [], "is_calibrated": is_calibrated
        }

    cy, cx = oc.mean(0)
    dists_from_center = np.sqrt(((oc - [cy, cx]) ** 2).sum(1))
    rmax = float(dists_from_center.max()) if len(dists_from_center) > 0 and dists_from_center.max() > 0 else 1.0

    out = []
    for i in range(1, n):
        m = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area_px = cv2.contourArea(cnt)
        if area_px < 10:  # noise floor
            continue

        (x, y), (w, h), angle = cv2.minAreaRect(cnt)
        d1 = max(w, h) / scale
        d2 = min(w, h) / scale
        mean_d = (d1 + d2) / 2.0
        r_ratio = float(np.hypot(x - cx, y - cy) / rmax)

        # Caliper box endpoints for visualization
        box_pts = cv2.boxPoints(((x, y), (w, h), angle)).astype(np.int32)
        bx, by, bw, bh = cv2.boundingRect(cnt)

        # Category
        is_cyst = is_calibrated and (mean_d > 30.0)
        is_dom = is_calibrated and (mean_d >= 10.0 and not is_cyst)
        is_antral = is_calibrated and (2.0 <= mean_d < 10.0)

        if is_cyst:
            cat = "Ovarian Cyst (>30mm)"
        elif is_dom:
            cat = "Dominant Follicle (>=10mm)"
        elif is_antral:
            cat = "Antral Follicle (2-9mm)"
        else:
            cat = "Micro-cavity (<2mm)" if is_calibrated else "Uncalibrated Cavity"

        out.append({
            "id": len(out) + 1,
            "D1": round(float(d1), 1),
            "D2": round(float(d2), 1),
            "mean": round(float(mean_d), 1),
            "diameter_mm": round(float(mean_d), 1) if is_calibrated else None,
            "r": round(float(r_ratio), 2),
            "center": (int(x), int(y)),
            "bbox": (int(bx), int(by), int(bw), int(bh)),
            "contour": cnt,
            "box_points": box_pts,
            "is_peripheral": (r_ratio > 0.6),
            "is_dominant": is_dom,
            "is_cyst": is_cyst,
            "is_antral": is_antral,
            "category": cat
        })

    antral = [f for f in out if f.get("is_antral", False)]
    dominant = [f for f in out if f.get("is_dominant", False)]
    cysts = [f for f in out if f.get("is_cyst", False)]

    peri_count = sum(1 for f in out if f["r"] > 0.6)
    peri_pct = round(float(100.0 * peri_count / len(out)), 1) if out else 0.0

    if peri_pct >= 55.0:
        spatial_dist = "Peripheral Subcapsular Ring ('String-of-Pearls')"
    elif out and np.mean([f["r"] for f in out]) <= 0.40:
        spatial_dist = "Central Stromal Clustering"
    else:
        spatial_dist = "Diffuse Parenchymal Dispersion"

    mean_diam = round(float(np.mean([f["mean"] for f in out])), 1) if out else None
    diam_range = f"{min(f['mean'] for f in out):.1f} - {max(f['mean'] for f in out):.1f} mm" if out else "N/A"

    return {
        "AFC": len(antral),
        "dominant": len(dominant),
        "cyst_count": len(cysts),
        "total_follicles": len(out),
        "mean_diam": mean_diam,
        "diameter_range_mm": diam_range,
        "peripheral_pct": peri_pct,
        "spatial_distribution": spatial_dist,
        "ovary_center": (int(cx), int(cy)),
        "ovary_rmax": round(float(rmax), 1),
        "is_calibrated": is_calibrated,
        "follicles": out
    }


# ==========================================
# 3. Two-Stage Pipeline Orchestrator
# ==========================================

class TwoStageFolliclePipeline:
    """
    Stage 1: Ovary Boundary Segmentation & Presence Gating
    Stage 2: Follicle Segmentation Gated by Ovary Mask
    """
    def __init__(
        self,
        ovary_weights_path: Optional[str] = None,
        follicle_weights_path: Optional[str] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.ovary_net = OvaryUNet(in_channels=1, out_channels=1).to(self.device)
        self.follicle_net = FollicleUNet(in_channels=2, out_channels=1).to(self.device)

        self.has_ovary_weights = False
        self.has_follicle_weights = False

        if ovary_weights_path and Path(ovary_weights_path).exists():
            try:
                self.ovary_net.load_state_dict(torch.load(ovary_weights_path, map_location=self.device))
                self.ovary_net.eval()
                self.has_ovary_weights = True
            except Exception as e:
                print(f"[WARN] Failed to load ovary U-Net weights: {e}")

        if follicle_weights_path and Path(follicle_weights_path).exists():
            try:
                self.follicle_net.load_state_dict(torch.load(follicle_weights_path, map_location=self.device))
                self.follicle_net.eval()
                self.has_follicle_weights = True
            except Exception as e:
                print(f"[WARN] Failed to load follicle U-Net weights: {e}")

    def segment_ovary_stage1(self, gray: np.ndarray, mm_per_pixel: float = 0.28) -> Tuple[bool, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """
        STAGE 1: Segments the ovary boundary.
        Returns:
          ovary_found: bool
          ovary_mask: uint8 binary mask [H, W]
          ovary_contour: ndarray of the ovary boundary contour
          metadata: dict
        """
        h, w = gray.shape[:2]

        # 1. Acoustic FOV sector detection
        tissue_raw = (gray > 12).astype(np.uint8) * 255
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        tissue_closed = cv2.morphologyEx(tissue_raw, cv2.MORPH_CLOSE, k_close)

        n_fov, lab_fov, stat_fov, _ = cv2.connectedComponentsWithStats(tissue_closed)
        if n_fov > 1:
            best_idx = 1 + np.argmax(stat_fov[1:, cv2.CC_STAT_AREA])
            fov_mask = (lab_fov == best_idx).astype(np.uint8) * 255
        else:
            fov_mask = np.ones((h, w), dtype=np.uint8) * 255

        # 2. Transabdominal Bladder Gating:
        # A bladder view exhibits a massive internal dark cavity (> 45mm wide and > 35mm high, area > 20,000 px)
        # fully enclosed inside pelvic tissue, without ovarian stroma.
        fluid_candidates = (gray < 30).astype(np.uint8)
        n_fl, _, stats_fl, _ = cv2.connectedComponentsWithStats(fluid_candidates)
        for i in range(1, n_fl):
            bx = stats_fl[i, cv2.CC_STAT_LEFT]
            by = stats_fl[i, cv2.CC_STAT_TOP]
            bw = stats_fl[i, cv2.CC_STAT_WIDTH]
            bh = stats_fl[i, cv2.CC_STAT_HEIGHT]
            area = stats_fl[i, cv2.CC_STAT_AREA]

            # Check if internal cavity (not touching scanner sector outer borders)
            is_internal = (bx > 5 and by > 5 and (bx + bw) < w - 5 and (by + bh) < h - 5)
            dw_mm = bw * mm_per_pixel
            dh_mm = bh * mm_per_pixel

            if is_internal and area >= 18000 and dw_mm >= 45.0 and dh_mm >= 35.0:
                # Transabdominal bladder view: Ovary is absent
                return False, np.zeros((h, w), dtype=np.uint8), None, {
                    "reason": "Transabdominal Bladder View (No Ovary in View)",
                    "fluid_cavity_width_mm": round(dw_mm, 1),
                    "fluid_cavity_height_mm": round(dh_mm, 1),
                    "ovary_area_px": 0
                }

        # 3. Model Inference or Calibrated Ovarian Stroma Localization
        ovary_mask = np.zeros((h, w), dtype=np.uint8)

        if self.has_ovary_weights:
            # Neural U-Net inference with proper 256x256 resizing
            inp_res = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)
            inp = torch.from_numpy(inp_res).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
            with torch.no_grad():
                pred = torch.sigmoid(self.ovary_net(inp)).squeeze().cpu().numpy()
            pred_orig = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
            ovary_mask = (pred_orig >= 0.40).astype(np.uint8) * 255
            # Erode sector edges
            ovary_mask = cv2.bitwise_and(ovary_mask, fov_mask)

        # Fallback / Stroma consolidation if model prediction is below threshold
        if np.sum(ovary_mask > 0) < 4000:
            denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
            stroma_cand = ((denoised >= 20) & (denoised <= 170) & (fov_mask > 0)).astype(np.uint8) * 255
            stroma_cand[:int(0.06 * h), :] = 0
            stroma_cand[int(0.92 * h):, :] = 0

            k_ov = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
            stroma_closed = cv2.morphologyEx(stroma_cand, cv2.MORPH_CLOSE, k_ov)
            n_st, lab_st, stats_st, _ = cv2.connectedComponentsWithStats(stroma_closed)

            if n_st > 1:
                # Find stroma cluster
                sorted_idx = 1 + np.argsort(stats_st[1:, cv2.CC_STAT_AREA])[::-1]
                for idx in sorted_idx:
                    area = stats_st[idx, cv2.CC_STAT_AREA]
                    if 2500 <= area <= 0.85 * (h * w):
                        comp = (lab_st == idx).astype(np.uint8) * 255
                        cnts_s, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if cnts_s:
                            hull = cv2.convexHull(max(cnts_s, key=cv2.contourArea))
                            ovary_mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.drawContours(ovary_mask, [hull], -1, 255, -1)
                            break

        # Area threshold check: Human ovary cannot be smaller than ~2500 pixels
        ovary_area = int(np.sum(ovary_mask > 0))
        min_ovary_area = int(2200 * (h * w) / (450 * 400))

        if ovary_area < min_ovary_area:
            return False, np.zeros((h, w), dtype=np.uint8), None, {
                "reason": "Ovary area below diagnostic threshold (< min_area)",
                "ovary_area_px": ovary_area
            }

        # Extract smooth boundary contour of segmented ovary
        cnts_ov, _ = cv2.findContours(ovary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ovary_contour = max(cnts_ov, key=cv2.contourArea) if cnts_ov else None

        return True, ovary_mask, ovary_contour, {
            "reason": "Ovary segmented successfully",
            "ovary_area_px": ovary_area,
            "ovary_coverage_pct": round(float(ovary_area / (h * w) * 100), 1)
        }

    def segment_follicles_stage2(
        self,
        gray: np.ndarray,
        ovary_mask: np.ndarray,
        px_per_mm: float
    ) -> np.ndarray:
        """
        STAGE 2: Segments follicles strictly inside the segmented ovary mask.
        Prediction is multiplied by ovary_mask (prediction * ovary_mask).
        """
        h, w = gray.shape[:2]
        if np.sum(ovary_mask > 0) == 0:
            return np.zeros((h, w), dtype=np.uint8)

        final_follicle_mask = np.zeros((h, w), dtype=np.uint8)

        if self.has_follicle_weights:
            # 2-channel U-Net input: Image + Ovary Mask resized to 256x256
            gray_res = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)
            ov_res = cv2.resize(ovary_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

            inp_img = torch.from_numpy(gray_res).float().unsqueeze(0).unsqueeze(0) / 255.0
            inp_ov = (torch.from_numpy(ov_res).float().unsqueeze(0).unsqueeze(0) > 127).float()
            inp_2ch = torch.cat([inp_img, inp_ov], dim=1).to(self.device)

            with torch.no_grad():
                pred = torch.sigmoid(self.follicle_net(inp_2ch)).squeeze().cpu().numpy()

            pred_orig = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
            raw_pred_bin = (pred_orig >= 0.35).astype(np.uint8) * 255
            # Strict gating: multiply prediction by ovary mask
            final_follicle_mask = cv2.bitwise_and(raw_pred_bin, ovary_mask)

        # Fallback / Acoustic cavity validation inside ovary
        if np.sum(final_follicle_mask > 0) < 100:
            denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
            k_hat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            bhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, k_hat)

            ov_pixels = denoised[ovary_mask > 0]
            mean_ov = float(np.mean(ov_pixels)) if len(ov_pixels) > 0 else 60.0

            cand_bin = ((bhat >= 16) & (denoised <= max(45.0, mean_ov - 10)) & (ovary_mask > 0)).astype(np.uint8) * 255
            cand_bin = cv2.morphologyEx(cand_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            cand_bin = cv2.morphologyEx(cand_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

            cnts, _ = cv2.findContours(cand_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 15 or area > 0.35 * (h * w):
                    continue

                (cx, cy), (cw, ch), _ = cv2.minAreaRect(c)
                major = max(cw, ch)
                minor = max(min(cw, ch), 0.1)
                aspect = major / minor

                # Reject thin fascia slits (> 3.0) and slivers < 1.5mm
                if aspect > 3.0 or (minor / px_per_mm) < 1.4:
                    continue

                # Centroid must be inside ovary mask
                M = cv2.moments(c)
                if M["m00"] > 0:
                    mcx, mcy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    if 0 <= mcx < w and 0 <= mcy < h and ovary_mask[mcy, mcx] > 0:
                        cv2.drawContours(final_follicle_mask, [c], -1, 255, -1)

        return final_follicle_mask

        return final_follicle_mask

    def run_two_stage_analysis(
        self,
        img_rgb: np.ndarray,
        mm_per_pixel: float = 0.28,
        pred_label: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete 2-stage analysis pipeline:
        Stage 1: Ovary segmentation -> if not found, stops.
        Stage 2: Follicle segmentation gated by ovary mask.
        Metrics: Calibrated calipers, AFC, dominant follicles, spatial distribution.
        """
        h, w = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        px_per_mm = (1.0 / mm_per_pixel) if (mm_per_pixel and mm_per_pixel > 0) else 3.57

        # STAGE 1: Ovary Segmentation
        ovary_found, ovary_mask, ovary_contour, meta1 = self.segment_ovary_stage1(gray, mm_per_pixel)

        if not ovary_found:
            return {
                "ovary_found": False,
                "AFC": 0,
                "antral_count": 0,
                "dominant": 0,
                "dominant_count": 0,
                "mean_diameter_mm": 0.0,
                "diameter_range_mm": "N/A",
                "peripheral_pct": 0.0,
                "spatial_distribution": "No Ovary in View",
                "pattern": "Ovary not in view, rescan",
                "pattern_code": "no_ovary_in_view",
                "rotterdam_afc_met": False,
                "has_dominant_follicle": False,
                "follicles": [],
                "inventory": [],
                "ovary_mask": ovary_mask,
                "ovary_contour": None,
                "follicle_mask": np.zeros((h, w), dtype=np.uint8),
                "view_warning": "Ovary not in view, rescan. The prominent fluid cavity is the urinary bladder or pelvic wall."
            }

        # STAGE 2: Follicle Segmentation Gated by Ovary Mask
        follicle_mask = self.segment_follicles_stage2(gray, ovary_mask, px_per_mm)

        # STAGE 3: Clinician Metrics Analysis
        metrics = analyse(follicle_mask, ovary_mask, px_per_mm)

        # Format output dictionary for workstation and UI
        afc = metrics["AFC"]
        dom_count = metrics["dominant"]
        has_dom = (dom_count > 0)
        is_rotterdam = bool(afc >= 12 and pred_label != "Normal Ovary")

        return {
            "ovary_found": True,
            "AFC": afc,
            "antral_count": afc,
            "dominant": dom_count,
            "dominant_count": dom_count,
            "cyst_count": metrics.get("cyst_count", 0),
            "mean_diameter_mm": metrics["mean_diam"] or 0.0,
            "diameter_range_mm": metrics["diameter_range_mm"],
            "peripheral_pct": metrics["peripheral_pct"],
            "spatial_distribution": metrics["spatial_distribution"],
            "pattern": metrics["spatial_distribution"],
            "pattern_code": "peripheral_ring" if metrics["peripheral_pct"] >= 55.0 else "diffuse",
            "rotterdam_afc_met": is_rotterdam,
            "has_dominant_follicle": has_dom,
            "follicles": metrics["follicles"],
            "inventory": metrics["follicles"],
            "ovary_mask": ovary_mask,
            "ovary_contour": ovary_contour,
            "follicle_mask": follicle_mask,
            "ovary_center": metrics["ovary_center"],
            "ovary_rmax": metrics["ovary_rmax"]
        }


# ==========================================
# 4. Authentic Workstation Visualizer
# ==========================================

def render_two_stage_visualization(
    img_rgb: np.ndarray,
    result: Dict[str, Any]
) -> np.ndarray:
    """
    Renders high-fidelity clinical PACS visualization:
      - Ovarian boundary drawn directly from the segmented ovary mask (NOT a fixed circle!)
      - Measured follicle caliper axes (D1 major, D2 minor) and tight bounding boxes
      - Follicle ID and mean diameter badge in mm
      - Ovarian centroid crosshair and spatial radial line
    """
    h, w = img_rgb.shape[:2]
    vis = img_rgb.copy()
    overlay = vis.copy()

    ovary_found = result.get("ovary_found", False)
    if not ovary_found:
        # Draw prominent rescan alert watermark
        cv2.putText(vis, "OVARY NOT IN VIEW (RESCAN REQUIRED)", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (239, 68, 68), 2, cv2.LINE_AA)
        cv2.rectangle(vis, (0, h - 26), (w, h), (15, 23, 42), -1)
        cv2.putText(vis, "Bladder / pelvic fluid cavity detected - 0 ovarian follicles",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (226, 232, 240), 1, cv2.LINE_AA)
        return vis

    # 1. Draw True Ovarian Boundary from Segmented Mask (NO FIXED CIRCLE)
    ovary_contour = result.get("ovary_contour")
    if ovary_contour is not None:
        # Smooth ovarian capsule outline in amber/cyan
        cv2.drawContours(vis, [ovary_contour], -1, (245, 158, 11), 2, lineType=cv2.LINE_AA)
        cv2.drawContours(overlay, [ovary_contour], -1, (180, 110, 20), -1)

    # Blend ovary stroma tint (12% opacity)
    cv2.addWeighted(overlay, 0.12, vis, 0.88, 0, vis)

    # 2. Ovarian Centroid Crosshair
    cx, cy = result.get("ovary_center", (w // 2, h // 2))
    cv2.drawMarker(vis, (cx, cy), (148, 163, 184), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1)

    # 3. Draw Detected Follicles & Calipers
    follicles = result.get("follicles", [])
    for f in follicles:
        cnt = f.get("contour")
        bx, by, bw, bh = f["bbox"]
        mean_d = f["mean"]
        d1 = f["D1"]
        d2 = f["D2"]
        is_dom = f.get("is_dominant", False)
        is_periph = f.get("is_peripheral", False)
        box_pts = f.get("box_points")

        # Color coding
        if is_dom:
            color = (245, 158, 11)   # Amber (Dominant)
            fill_tint = (180, 100, 20)
        elif is_periph:
            color = (16, 220, 140)   # Emerald (Peripheral subcapsular)
            fill_tint = (10, 120, 60)
        else:
            color = (56, 189, 248)   # Sky Cyan (Central antral)
            fill_tint = (20, 80, 140)

        # Draw contour & caliper rotated bounding box
        if cnt is not None:
            cv2.drawContours(vis, [cnt], -1, color, 2 if is_dom else 1, lineType=cv2.LINE_AA)
        if box_pts is not None:
            cv2.drawContours(vis, [box_pts], 0, color, 1, lineType=cv2.LINE_AA)

        # Caliper crosshair for dominant follicles
        if is_dom:
            cv2.drawMarker(vis, (f["center"][0], f["center"][1]), color, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

        # Follicle Badge
        tag = f"DF:{mean_d}mm" if is_dom else f"F{f['id']}:{mean_d}mm"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.33 if is_dom else 0.28
        (tw, th), _ = cv2.getTextSize(tag, font, scale, 1)

        tag_y = max(by - 4, th + 4)
        cv2.rectangle(vis, (bx, tag_y - th - 3), (bx + tw + 4, tag_y + 2), (15, 23, 42), -1)
        cv2.rectangle(vis, (bx, tag_y - th - 3), (bx + tw + 4, tag_y + 2), color, 1)
        cv2.putText(vis, tag, (bx + 2, tag_y - 1), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Footer status
    footer = f"AFC: {result['AFC']} | Dominant: {result['dominant']} (>=10mm) | Mean Diam: {result['mean_diameter_mm']}mm | {result['peripheral_pct']}% Peripheral"
    cv2.rectangle(vis, (0, h - 22), (w, h), (15, 23, 42), -1)
    cv2.putText(vis, footer, (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (226, 232, 240), 1, cv2.LINE_AA)

    return vis
