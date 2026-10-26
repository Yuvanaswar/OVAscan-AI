"""
Dataset Pre-processing Tool: Green Annotation to Binary Training Mask Converter
Converts hand-drawn or workstation green outline annotations into ground-truth binary masks
and generates Telea-inpainted clean ultrasound training images.

Usage:
    python tools/prepare_segmentation_masks.py --input /path/to/annotated_images --output /path/to/dataset
"""

import os
import glob
import argparse
from pathlib import Path
import cv2
import numpy as np


def green_to_mask(img_path: str, out_dir: str, hsv_lower=(35, 100, 100), hsv_upper=(90, 255, 255)):
    """
    Extracts binary follicle masks from green hand-drawn rings and inpaints clean images.
    
    Parameters:
      img_path: Path to the annotated ultrasound image containing green marker rings.
      out_dir: Base directory where images/ and masks/ subdirectories will be saved.
      hsv_lower: Lower HSV threshold for bright green marker.
      hsv_upper: Upper HSV threshold for bright green marker.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Unable to load image: {img_path}")
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. Isolate bright green marker
    ring = cv2.inRange(hsv, hsv_lower, hsv_upper)
    ring = cv2.morphologyEx(ring, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # 2. Extract inner contour to fill lumen
    cnts, hier = cv2.findContours(ring, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(ring.shape, dtype=np.uint8)

    if hier is not None and len(cnts) > 0:
        for i, c in enumerate(cnts):
            # Inner contour = region inside the green ring marker
            if hier[0][i][3] != -1:
                cv2.drawContours(mask, [c], -1, 255, -1)
            else:
                # Fallback if ring has no inner hole (filled green circle)
                cv2.drawContours(mask, [c], -1, 255, -1)

    # 3. Telea inpainting: Inpaint the marker out of the raw ultrasound
    # Training MUST be done on the inpainted image to prevent the model from learning marker shortcuts
    clean = img.copy()
    if np.any(ring > 0):
        clean[ring > 0] = cv2.inpaint(img, ring, 3, cv2.INPAINT_TELEA)[ring > 0]

    # 4. Save to dataset directories
    name = Path(img_path).stem
    img_out_dir = Path(out_dir) / "images"
    mask_out_dir = Path(out_dir) / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(img_out_dir / f"{name}.png"), clean)
    cv2.imwrite(str(mask_out_dir / f"{name}.png"), mask)
    return True


def batch_process(input_path: str, output_path: str):
    p = Path(input_path)
    if p.is_file():
        files = [str(p)]
    else:
        files = glob.glob(str(p / "*.png")) + glob.glob(str(p / "*.jpg")) + glob.glob(str(p / "*.jpeg"))

    print(f"Found {len(files)} image(s) to process...")
    success = 0
    for f in files:
        if green_to_mask(f, output_path):
            success += 1

    print(f"[DONE] Successfully generated {success} clean image/mask pairs in '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert green annotated scans to binary masks.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", "-o", type=str, default="data/segmentation_dataset", help="Output directory")
    args = parser.parse_args()

    batch_process(args.input, args.output)
