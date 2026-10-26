import matplotlib
matplotlib.use('Agg')
"""
Clinical Ovarian Ultrasound AI Diagnostic Platform:
3-Class Screening System:
  - Class 0: Normal Ovary
  - Class 1: PCOS (Polycystic Ovary Syndrome)
  - Class 2: Dominant Follicle (DF / Ovulatory Follicle)
Supported Models:
  1. ConvNeXt-Tiny V6 (Primary Champion: 27.8M params, 84.9% 3-Class Test Acc, 100% PCOS Sensitivity)
  2. ResNet-50 (Deep Residual Baseline: 23.5M params)
  3. MobileNet-V2 (Ultra-Fast Lightweight: 2.2M params, 30ms latency)
Explainability:
  - Multi-Scale LayerCAM & Grad-CAM++
  - Real Acoustic Follicle Delineation & Bounding
  - AI & Follicle Concordance Analysis
  - Real-Time Clinical Rotterdam Diagnostic Metrics
"""

import os
import uuid
import datetime
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

from src.models import build_model
from src.preprocessing import letterbox_pad_to_square, get_eval_transform, validate_ultrasound_input, IMAGE_SIZE
from src.explainability import V6ExplainabilityEngine, generate_all_explanations

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
GRADCAM_OUT_DIR = STATIC_DIR / "gradcam_output"
UPLOADS_DIR = BASE_DIR / "uploads"
REAL_DATA_DIR = BASE_DIR / "real time data"

GRADCAM_OUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(STATIC_DIR))
CORS(app)

# 3-Class Definition
CLASS_NAMES = ["Normal Ovary", "PCOS", "Dominant Follicle"]
NUM_CLASSES = 3

# Available Top-2 Production Models
AVAILABLE_MODELS = {
    "convnext_v6_3class": {
        "name": "ConvNeXt-Tiny V6 (Primary Champion)",
        "arch": "convnext_tiny",
        "ckpt": MODELS_DIR / "V6_ConvNeXt-Tiny" / "best_convnext_tiny_3class.pth",
        "num_classes": 3,
        "default_threshold": 0.50,
        "latency_ms": 41.7,
        "parameters": "27.8M",
        "accuracy": "84.9% Test Accuracy (0.8506 Macro-F1)",
        "description": "Top-1 Champion: 7x7 depthwise convolutions with 100% PCOS sensitivity and 86.2% DF sensitivity on real clinical dataset."
    },
    "resnet50_3class": {
        "name": "ResNet-50 (Deep Residual Top-2)",
        "arch": "resnet50",
        "ckpt": MODELS_DIR / "ResNet50" / "best_resnet50_3class.pth",
        "num_classes": 3,
        "default_threshold": 0.50,
        "latency_ms": 153.2,
        "parameters": "23.5M",
        "accuracy": "81.4% Test Accuracy (96.6% DF Sensitivity, 100% PCOS Precision)",
        "description": "Top-2 Benchmark: Deep residual bottleneck blocks with 96.6% dominant follicle recall and 100% precision on PCOS."
    }
}

ACTIVE_MODEL_KEY = "convnext_v6_3class"
ACTIVE_THRESHOLD = 0.50
loaded_model = None
explainability_engine = None

def load_ai_model(model_key: str):
    global loaded_model, explainability_engine, ACTIVE_MODEL_KEY, ACTIVE_THRESHOLD
    if model_key not in AVAILABLE_MODELS:
        model_key = "convnext_v6_3class"

    cfg = AVAILABLE_MODELS[model_key]
    arch = cfg["arch"]
    num_classes = cfg.get("num_classes", 3)
    model = build_model(arch, num_classes=num_classes, pretrained=False)

    ckpt_path = cfg["ckpt"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cleaned_state = {k.replace("backbone.", ""): v for k, v in state.items()}

    model.load_state_dict(cleaned_state)
    model.to(DEVICE).eval()

    if explainability_engine:
        explainability_engine.remove_hooks()

    explainability_engine = V6ExplainabilityEngine(model, arch=arch)
    loaded_model = model
    ACTIVE_MODEL_KEY = model_key
    ACTIVE_THRESHOLD = cfg["default_threshold"]

    print(f"[+] Loaded {cfg['name']} on {DEVICE}")

# Initialize default model
load_ai_model("convnext_v6_3class")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/models", methods=["GET"])
def get_models():
    models_info = {}
    for k, v in AVAILABLE_MODELS.items():
        models_info[k] = {
            "name": v["name"],
            "description": v["description"],
            "threshold": v["default_threshold"],
            "parameters": v.get("parameters", "N/A"),
            "latency_ms": v.get("latency_ms", "N/A"),
            "accuracy": v.get("accuracy", "N/A"),
            "is_active": (k == ACTIVE_MODEL_KEY)
        }
    return jsonify({
        "status": "success",
        "active_model": ACTIVE_MODEL_KEY,
        "operating_threshold": ACTIVE_THRESHOLD,
        "classes": CLASS_NAMES,
        "models": models_info
    })

@app.route("/api/set-model", methods=["POST"])
def set_model():
    data = request.get_json() or {}
    key = data.get("model_key")
    if key not in AVAILABLE_MODELS:
        return jsonify({"status": "error", "message": f"Unknown model: {key}"}), 400
    load_ai_model(key)
    return jsonify({
        "status": "success",
        "active_model": ACTIVE_MODEL_KEY,
        "model_name": AVAILABLE_MODELS[ACTIVE_MODEL_KEY]["name"],
        "operating_threshold": ACTIVE_THRESHOLD
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    pil_image = None
    orig_filename = "ultrasound.png"

    if request.is_json:
        data = request.get_json() or {}
        ex_name = data.get("example_filename")
        if ex_name:
            ex_path = STATIC_DIR / "images" / "examples" / ex_name
            if ex_path.exists():
                pil_image = Image.open(ex_path).convert("RGB")
                orig_filename = ex_name
            else:
                return jsonify({"is_valid_input": False, "status": "error", "message": f"Example image not found: {ex_name}"}), 404
        else:
            return jsonify({"is_valid_input": False, "status": "error", "message": "No example_filename provided."}), 400
    elif "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"is_valid_input": False, "status": "error", "message": "No file selected."}), 400
        try:
            pil_image = Image.open(file.stream).convert("RGB")
            orig_filename = Path(file.filename).name
        except Exception as e:
            return jsonify({"is_valid_input": False, "status": "error", "message": "Corrupt image format."}), 400
    else:
        return jsonify({"is_valid_input": False, "status": "error", "message": "No image file or example provided."}), 400

    # 1. Quality Gate Pre-Screening (Acoustic sector and ultrasound brightness check)
    val_res = validate_ultrasound_input(pil_image)
    if not val_res["is_valid"]:
        return jsonify({
            "is_valid_input": False,
            "status": val_res["status"],
            "message": val_res["message"],
            "validation_details": val_res.get("details", {})
        }), 200

    # 2. Save Upload / Processed Image
    analysis_id = str(uuid.uuid4())[:8]
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_filename = f"{date_str}_{analysis_id}_{orig_filename}"
    upload_path = UPLOADS_DIR / save_filename
    pil_image.save(upload_path)

    # 3. Model Inference (3-Class Softmax)
    img_padded = letterbox_pad_to_square(pil_image)
    eval_tf = get_eval_transform(IMAGE_SIZE)
    input_tensor = eval_tf(img_padded).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = loaded_model(input_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    p_normal = float(probs[0])
    p_pcos = float(probs[1])
    p_df = float(probs[2])

    pred_idx = int(np.argmax(probs))
    prediction_label = CLASS_NAMES[pred_idx]
    top_prob = float(probs[pred_idx])

    # Status color and clinical summary
    if pred_idx == 1:
        status_color = "#e11d48"  # Rose / Crimson
        status_code = "pcos"
        prediction_summary = "High probability of Polycystic Ovary Syndrome (PCOS). Ultrasound shows characteristic polyfollicular patterns with stromal thickening."
    elif pred_idx == 2:
        status_color = "#d97706"  # Amber / Gold
        status_code = "dominant_follicle"
        prediction_summary = "Dominant Follicle (DF) detected. Indicates physiological pre-ovulatory / Graafian follicle development (>10mm) in normal ovulatory cycle."
    else:
        status_color = "#059669"  # Emerald Green
        status_code = "normal"
        prediction_summary = "Normal ovarian parenchymal architecture. Physiological follicular reserve without polyfollicular crowding or dominant follicle expansion."

    class_probabilities = {
        "Normal Ovary": round(p_normal * 100, 2),
        "PCOS": round(p_pcos * 100, 2),
        "Dominant Follicle": round(p_df * 100, 2)
    }

    # 4. Generate Explainability Suite (LayerCAM for the predicted class)
    layercam, gradcampp, _ = explainability_engine.compute_attributions(input_tensor, target_class=pred_idx)
    explanations, attention_metrics = generate_all_explanations(
        pil_image=pil_image,
        layercam_raw=layercam,
        gradcam_raw=gradcampp,
        analysis_id=analysis_id,
        base_filename=orig_filename,
        pred_class_idx=pred_idx,
        pred_label=prediction_label,
        class_probs=class_probabilities,
        threshold=ACTIVE_THRESHOLD,
        output_dir=GRADCAM_OUT_DIR
    )

    conf_level = "High" if top_prob >= 0.70 else ("Moderate" if top_prob >= 0.50 else "Borderline Decision")

    # View Quality Check: If no ovary in frame (e.g. transabdominal bladder/wall view)
    if attention_metrics.get("ovary_in_view") is False or attention_metrics.get("ovary_found") is False:
        prediction_label = "Ovary Not in View (Rescan Required)"
        status_code = "ovary_absent"
        status_color = "#64748b"
        prediction_summary = attention_metrics.get(
            "view_warning",
            "Ovary not in view, rescan. The prominent fluid cavity is the urinary bladder, and surrounding hypoechoic bands represent abdominal wall muscle/fascia."
        )

    return jsonify({
        "is_valid_input": True,
        "analysis_id": analysis_id,
        "analysis_date": datetime.datetime.now().strftime("%B %d, %Y, %I:%M %p"),
        "image_name": orig_filename,
        "image_url": f"/uploads/{save_filename}",
        "gradcam_url": explanations["standard"]["url"],
        "explanations": explanations,
        "attention_metrics": attention_metrics,
        "follicle_metrics": {
            "afc": attention_metrics.get("afc", 0),
            "rotterdam_afc_met": attention_metrics.get("rotterdam_afc_met", False),
            "pattern": attention_metrics.get("follicle_pattern", "N/A"),
            "mean_diameter_mm": attention_metrics.get("mean_diameter_mm", 0.0),
            "diameter_range_mm": attention_metrics.get("diameter_range_mm", "N/A"),
            "subcapsular_pct": attention_metrics.get("subcapsular_pct", 0.0),
            "peripheral_pct": attention_metrics.get("peripheral_pct", 0.0),
            "spatial_distribution": attention_metrics.get("spatial_distribution", "N/A"),
            "subcapsular_count": attention_metrics.get("subcapsular_count", 0),
            "has_dominant_follicle": attention_metrics.get("has_dominant_follicle", False),
            "dominant_count": attention_metrics.get("dominant_count", 0),
            "cyst_count": attention_metrics.get("cyst_count", 0),
            "antral_count": attention_metrics.get("antral_count", 0),
            "ovary_found": attention_metrics.get("ovary_found", True),
            "stromal_follicle_ratio": attention_metrics.get("stromal_follicle_ratio", 1.0),
            "ai_concordance_pct": attention_metrics.get("ai_concordance_pct", 0.0),
            "inventory": attention_metrics.get("follicle_inventory", [])
        },
        "prediction": prediction_label,
        "pred_class_idx": pred_idx,
        "status_code": status_code,
        "status_color": status_color,
        "prediction_summary": prediction_summary,
        "is_pcos": (pred_idx == 1),
        "is_dominant_follicle": (pred_idx == 2),
        "is_normal": (pred_idx == 0),
        "class_probabilities": class_probabilities,
        "pcos_probability": round(p_pcos * 100, 2),
        "normal_probability": round(p_normal * 100, 2),
        "df_probability": round(p_df * 100, 2),
        "top_probability": round(top_prob * 100, 2),
        "operating_threshold": round(ACTIVE_THRESHOLD, 2),
        "confidence": conf_level,
        "confidence_level": conf_level,
        "confidence_bars": 3 if conf_level == "High" else (2 if conf_level == "Moderate" else 1),
        "confidence_color": status_color,
        "model_name": AVAILABLE_MODELS[ACTIVE_MODEL_KEY]["name"]
    })

@app.route("/api/examples")
def get_examples():
    ex_dir = STATIC_DIR / "images" / "examples"
    examples = []
    if ex_dir.exists():
        for p in sorted(ex_dir.glob("*.*")):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                name_l = p.name.lower()
                if "df" in name_l or "dominant" in name_l:
                    tag = "Dominant Follicle (DF)"
                    badge_cls = "df"
                elif "pcos" in name_l or "pco" in name_l or "105" in name_l or "102" in name_l:
                    tag = "PCOS Case"
                    badge_cls = "pcos"
                else:
                    tag = "Normal Ovary"
                    badge_cls = "normal"

                examples.append({
                    "filename": p.name,
                    "url": f"/static/images/examples/{p.name}",
                    "label": tag,
                    "badge_class": badge_cls
                })
    return jsonify({"examples": examples})

@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOADS_DIR, filename)

if __name__ == "__main__":
    print(f"\n==================================================")
    print(f"  PCOS & Dominant Follicle AI Clinical Platform")
    print(f"  3 Classes: Normal Ovary | PCOS | Dominant Follicle")
    print(f"  Access UI at: http://localhost:5050")
    print(f"==================================================\n")
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
