"""
Two-Stage Segmentation Dataset Pre-processing Tool
Converts hand-drawn green outlines (follicles) and yellow outlines (ovary boundary)
into ground-truth binary training masks, with Telea inpainting of markers.

Outputs three directories:
  - images/         : clean ultrasound scans (markers inpainted out)
  - follicle_masks/ : filled binary masks from green marker outlines (positive only)
  - ovary_masks/    : filled binary masks of ovary stroma (empty if no ovary in view)

Usage:
  python tools/prepare_segmentation_masks.py --input <path_to_images_or_dir> --output data/segmentation_dataset
"""

import os
import glob
import argparse
from pathlib import Path
import cv2
import numpy as np


def extract_filled_mask_from_marker(hsv: np.ndarray, lower_thresh: tuple, upper_thresh: tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a binary filled mask from a colored marker boundary ring.
    Returns:
      filled_mask: uint8 binary mask with 255 inside the marker loops
      marker_ring: uint8 mask of the actual marker strokes (for inpainting)
    """
    ring = cv2.inRange(hsv, lower_thresh, upper_thresh)
    ring_closed = cv2.morphologyEx(ring, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, hier = cv2.findContours(ring_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros(ring.shape, dtype=np.uint8)
    if hier is not None and len(cnts) > 0:
        for i, c in enumerate(cnts):
            # Inner contour = region inside the marker ring
            if hier[0][i][3] != -1:
                cv2.drawContours(mask, [c], -1, 255, -1)
            else:
                # If marker stroke was solid/filled rather than hollow ring
                cv2.drawContours(mask, [c], -1, 255, -1)

    return mask, ring


def process_ultrasound_annotations(
    img_path: str,
    out_dir: str,
    is_negative_scan: bool = False
) -> bool:
    """
    Processes an annotated ultrasound image into clean image, follicle mask, and ovary mask.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Unable to load image: {img_path}")
        return False

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    out_base = Path(out_dir)
    img_dir = out_base / "images"
    follicle_dir = out_base / "follicle_masks"
    ovary_dir = out_base / "ovary_masks"

    for d in [img_dir, follicle_dir, ovary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    name = Path(img_path).stem

    # Negative scan (e.g. bladder view or no ovary in frame)
    if is_negative_scan:
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        # Inpaint any green/yellow notes if present
        green_ring = cv2.inRange(hsv, (35, 100, 100), (90, 255, 255))
        yellow_ring = cv2.inRange(hsv, (18, 100, 100), (34, 255, 255))
        all_markers = cv2.bitwise_or(green_ring, yellow_ring)
        clean = img.copy()
        if np.any(all_markers > 0):
            clean[all_markers > 0] = cv2.inpaint(img, all_markers, 3, cv2.INPAINT_TELEA)[all_markers > 0]
        
        cv2.imwrite(str(img_dir / f"{name}.png"), clean)
        cv2.imwrite(str(follicle_dir / f"{name}.png"), empty_mask)
        cv2.imwrite(str(ovary_dir / f"{name}.png"), empty_mask)
        return True

    # 1. Extract Follicles from Green Marker (HSV 35-90)
    # Bright green markers: H: 35-90, S: 100-255, V: 100-255
    follicle_mask, green_ring = extract_filled_mask_from_marker(
        hsv, lower_thresh=(35, 100, 100), upper_thresh=(90, 255, 255)
    )

    # 2. Extract Ovary Outline from Yellow Marker (HSV 18-34)
    # Yellow markers: H: 18-34, S: 100-255, V: 100-255
    ovary_mask, yellow_ring = extract_filled_mask_from_marker(
        hsv, lower_thresh=(18, 100, 100), upper_thresh=(34, 255, 255)
    )

    # If no explicit yellow ovary outline was drawn, but follicles exist:
    # Compute the ovarian boundary as the convex hull of follicles dilated with an anatomical stromal margin
    if np.sum(ovary_mask > 0) == 0 and np.sum(follicle_mask > 0) > 0:
        cnts_f, _ = cv2.findContours(follicle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_f:
            all_pts = np.vstack([c.reshape(-1, 2) for c in cnts_f])
            ovary_hull = cv2.convexHull(all_pts)
            temp_ovary = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(temp_ovary, [ovary_hull], -1, 255, -1)
            # Dilate by 35-45 pixels (~10-12 mm stroma margin)
            k_stroma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
            ovary_mask = cv2.dilate(temp_ovary, k_stroma)

    # 3. Telea inpainting: Inpaint ALL marker strokes out of raw ultrasound
    # Training MUST be performed exclusively on clean ultrasound without markers!
    all_markers = cv2.bitwise_or(green_ring, yellow_ring)
    clean = img.copy()
    if np.any(all_markers > 0):
        # Dilate slightly so marker edges don't leave faint borders
        marker_dil = cv2.dilate(all_markers, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        clean[marker_dil > 0] = cv2.inpaint(img, marker_dil, 3, cv2.INPAINT_TELEA)[marker_dil > 0]

    # 4. Save to respective dataset folders
    cv2.imwrite(str(img_dir / f"{name}.png"), clean)
    cv2.imwrite(str(follicle_dir / f"{name}.png"), follicle_mask)
    cv2.imwrite(str(ovary_dir / f"{name}.png"), ovary_mask)
    return True


def batch_process(input_path: str, output_path: str, negative_keywords: list = None, force_negative: bool = False):
    negative_keywords = negative_keywords or ["bladder", "negative", "empty", "no_ovary"]
    p = Path(input_path)
    if p.is_file():
        files = [str(p)]
    else:
        files = glob.glob(str(p / "*.png")) + glob.glob(str(p / "*.jpg")) + glob.glob(str(p / "*.jpeg"))

    print(f"Found {len(files)} image(s) to process for 2-stage segmentation...")
    success = 0
    for f in files:
        fname_lower = Path(f).name.lower()
        is_neg = force_negative or any(k in fname_lower for k in negative_keywords)
        if process_ultrasound_annotations(f, output_path, is_negative_scan=is_neg):
            success += 1

    print(f"[DONE] Successfully generated {success} samples into:")
    print(f"  - {output_path}/images/")
    print(f"  - {output_path}/follicle_masks/")
    print(f"  - {output_path}/ovary_masks/")


if __name__ == "__main__":
    from typing import Tuple
    parser = argparse.ArgumentParser(description="Convert hand-drawn green/yellow annotations to 2-stage binary masks.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", "-o", type=str, default="data/segmentation_dataset", help="Output directory")
    parser.add_argument("--negative", action="store_true", help="Force treat inputs as negative (no ovary) scans")
    args = parser.parse_args()

    batch_process(args.input, args.output, force_negative=args.negative)
