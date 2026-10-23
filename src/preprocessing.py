"""
Ultrasound Preprocessing, Letterbox Padding, and Out-of-Distribution (OOD) Quality Gate.
"""

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def letterbox_pad_to_square(image: Image.Image, fill_color=(0, 0, 0)) -> Image.Image:
    """
    Pads an image to a 1:1 square aspect ratio without stretching.
    Preserves true circular follicular geometry.
    """
    w, h = image.size
    if w == h:
        return image
    max_side = max(w, h)
    padded = Image.new("RGB", (max_side, max_side), fill_color)
    offset_x = (max_side - w) // 2
    offset_y = (max_side - h) // 2
    padded.paste(image, (offset_x, offset_y))
    return padded

def get_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def validate_ultrasound_input(pil_image: Image.Image):
    """
    Acoustic Domain & Quality Gate:
    Pre-screens images to reject non-ultrasound photos, UI graphics, and corrupt files.
    """
    img_np = np.array(pil_image)
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    # 1. Grayscale Consistency Check
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    rg_diff = np.abs(r.astype(np.float32) - g.astype(np.float32))
    rb_diff = np.abs(r.astype(np.float32) - b.astype(np.float32))
    color_std = float(np.mean(rg_diff + rb_diff))

    if color_std > 14.0:
        return {
            "is_valid": False,
            "status": "unsupported_input",
            "message": "Non-Ultrasound Image Detected. Color photographs and UI graphics are not supported. Please upload a monochromatic B-mode ultrasound scan.",
            "details": {"reason": "Color image detected", "color_std": round(color_std, 1)}
        }

    # 2. Dynamic Range & Intensity Extremes Check
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    if mean_val < 15.0 or std_val < 12.0:
        return {
            "is_valid": False,
            "status": "unsupported_input",
            "message": "Low Signal / Blank Dark Image. Please provide a clear ovarian ultrasound scan with visible tissue parenchyma.",
            "details": {"reason": "Insufficient tissue signal / near-black image", "mean_brightness": round(mean_val, 1), "contrast_std": round(std_val, 1)}
        }

    if mean_val > 228.0 and std_val < 35.0:
        return {
            "is_valid": False,
            "status": "unsupported_input",
            "message": "Overexposed / Blank White Screen Detected. Please upload an authentic ultrasound scan.",
            "details": {"reason": "Overexposed white screen", "mean_brightness": round(mean_val, 1)}
        }

    # 3. Acoustic Speckle Edge Density Check
    edges = cv2.Canny(gray, 40, 110)
    edge_density = float(np.mean(edges > 0))
    if edge_density < 0.008:
        return {
            "is_valid": False,
            "status": "unsupported_input",
            "message": "Acoustic Texture Absent. Image lacks the characteristic acoustic speckle pattern of B-mode ovarian ultrasound.",
            "details": {"reason": "Flat synthetic graphic / no acoustic speckle", "edge_density": round(edge_density, 4)}
        }

    return {"is_valid": True, "status": "valid_ultrasound", "message": "Input passed acoustic ultrasound quality gate."}

def unpad_cam_to_original(cam_224: np.ndarray, original_size: tuple) -> np.ndarray:
    """
    Maps 224x224 attribution heatmap back to original image aspect ratio without spatial offset.
    """
    w_orig, h_orig = original_size
    max_side = max(w_orig, h_orig)

    cam_square = cv2.resize(cam_224, (max_side, max_side), interpolation=cv2.INTER_LINEAR)
    pad_x = (max_side - w_orig) // 2
    pad_y = (max_side - h_orig) // 2

    cam_cropped = cam_square[pad_y : pad_y + h_orig, pad_x : pad_x + w_orig]
    cam_final = cv2.resize(cam_cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    if np.max(cam_final) > np.min(cam_final):
        cam_final = (cam_final - np.min(cam_final)) / (np.max(cam_final) - np.min(cam_final))
    return cam_final

def apply_acoustic_sector_gating(cam: np.ndarray, orig_np: np.ndarray):
    """
    Suppresses typographic machine headers (hospital text at top margin) and background black void.
    """
    gray = cv2.cvtColor((orig_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    tissue_mask = (gray > 16).astype(np.float32)
    # Zero out top 11% where hospital name and scanner metadata are printed
    tissue_mask[:int(0.11 * cam.shape[0]), :] = 0.0

    gated_cam = cam * tissue_mask
    if np.max(gated_cam) > np.min(gated_cam):
        gated_cam = (gated_cam - np.min(gated_cam)) / (np.max(gated_cam) - np.min(gated_cam))
    return gated_cam, tissue_mask
