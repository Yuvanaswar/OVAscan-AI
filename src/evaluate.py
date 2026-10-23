"""
Evaluation Script for Top-2 Clinical Ovarian Ultrasound Models on 3-Class Test Set.
Supports:
  - ConvNeXt-Tiny V6 (Primary Champion)
  - ResNet-50 (Deep Residual Top-2 Benchmark)
Evaluates on 'real time data/test' across:
  - Class 0: Normal Ovary
  - Class 1: PCOS (Polycystic Ovary Syndrome)
  - Class 2: Dominant Follicle (DF)
"""

from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .models import build_model
from .preprocessing import letterbox_pad_to_square, get_eval_transform, IMAGE_SIZE
from .dataset import CLASS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model(model_name: str, ckpt_path: Path, test_dir: Path):
    arch = "convnext_tiny" if "convnext" in model_name.lower() else "resnet50"
    model = build_model(arch, num_classes=3, pretrained=False)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cleaned_state = {k.replace("backbone.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned_state)
    model.to(DEVICE).eval()

    eval_tf = get_eval_transform(IMAGE_SIZE)

    y_true = []
    y_pred = []
    y_probs = []

    class_subfolders = [("NORMAL", 0), ("PCOS", 1), ("DF", 2)]

    for folder_name, label_idx in class_subfolders:
        dir_p = test_dir / folder_name
        if not dir_p.exists():
            continue
        for img_path in sorted(dir_p.glob("*.*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                img_padded = letterbox_pad_to_square(img)
                tensor = eval_tf(img_padded).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = model(tensor)
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

                pred_class = int(np.argmax(probs))
                y_true.append(label_idx)
                y_pred.append(pred_class)
                y_probs.append(probs)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)

    print(f"\n==================== {model_name} (3-Class Test Evaluation) ====================")
    print(f"Total Test Samples: {len(y_true)}")
    print(f"Overall Accuracy  : {acc * 100:.2f}%")
    print(f"Macro F1-Score    : {macro_f1:.4f}")
    print("\nConfusion Matrix (Rows: True, Cols: Pred):")
    print("                 Pred Normal   Pred PCOS   Pred DF")
    for idx, c_name in enumerate(CLASS_NAMES):
        row_str = " ".join([f"{cm[idx, j]:11d}" for j in range(3)])
        print(f"True {c_name:10s} : {row_str}")
    print("\nDetailed Classification Report:")
    print(report)
    print("=================================================================================\n")

    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm}

def main():
    root = Path(__file__).resolve().parent.parent
    test_dir = root / "real time data" / "test"
    models_dir = root / "models"

    convnext_ckpt = models_dir / "V6_ConvNeXt-Tiny" / "best_convnext_tiny_3class.pth"
    resnet_ckpt = models_dir / "ResNet50" / "best_resnet50_3class.pth"

    print("Evaluating Top-2 Clinical Models on 3-Class Test Set...\n")
    if convnext_ckpt.exists():
        evaluate_model("ConvNeXt-Tiny V6 (Primary Champion)", convnext_ckpt, test_dir)
    if resnet_ckpt.exists():
        evaluate_model("ResNet-50 (Deep Residual Benchmark)", resnet_ckpt, test_dir)

if __name__ == "__main__":
    main()
