"""
CLI Prediction Script for Clinical Ovarian Ultrasound AI:
Supports 3 Classes:
  - Normal Ovary
  - PCOS (Polycystic Ovary Syndrome)
  - Dominant Follicle (DF)

Usage:
  python -m src.predict --image path/to/scan.png [--model convnext|resnet]
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from .models import build_model
from .preprocessing import letterbox_pad_to_square, get_eval_transform, validate_ultrasound_input, IMAGE_SIZE
from .dataset import CLASS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_single_image(image_path: Path, model_choice: str = "convnext"):
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"

    if model_choice.lower() in ["resnet", "resnet50"]:
        ckpt_path = models_dir / "ResNet50" / "best_resnet50_3class.pth"
        arch = "resnet50"
        name = "ResNet-50 (Deep Residual Top-2)"
    else:
        ckpt_path = models_dir / "V6_ConvNeXt-Tiny" / "best_convnext_tiny_3class.pth"
        arch = "convnext_tiny"
        name = "ConvNeXt-Tiny V6 (Primary Champion)"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    model = build_model(arch, num_classes=3, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cleaned_state = {k.replace("backbone.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned_state)
    model.to(DEVICE).eval()

    img = Image.open(image_path).convert("RGB")
    val_res = validate_ultrasound_input(img)
    if not val_res["is_valid"]:
        print(f"\n[Quality Gate Rejection]: {val_res['message']}")
        return

    padded = letterbox_pad_to_square(img)
    eval_tf = get_eval_transform(IMAGE_SIZE)
    tensor = eval_tf(padded).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]
    top_prob = float(probs[pred_idx])

    print(f"\n==================== Clinical AI Diagnostic Result ====================")
    print(f"Active Model       : {name}")
    print(f"Image File         : {image_path.name}")
    print(f"Primary Verdict    : {pred_label}")
    print(f"Confidence         : {top_prob * 100:.2f}%")
    print(f"-----------------------------------------------------------------------")
    print(f"Normal Ovary Prob  : {probs[0] * 100:.2f}%")
    print(f"PCOS Probability   : {probs[1] * 100:.2f}%")
    print(f"Dominant Follicle  : {probs[2] * 100:.2f}%")
    print("=======================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Ovarian Ultrasound AI Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to ultrasound image file")
    parser.add_argument("--model", type=str, default="convnext", choices=["convnext", "resnet"], help="Model (convnext or resnet)")
    args = parser.parse_args()

    predict_single_image(Path(args.image), model_choice=args.model)
