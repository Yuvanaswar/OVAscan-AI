# OvaScan AI: Clinical Ultrasound PACS Workstation (3-Class Diagnostic System)

## 1. Project Purpose & Summary
**OvaScan AI** is an advanced clinical-grade artificial intelligence workstation designed for transvaginal and transabdominal ovarian ultrasound diagnostic assistance.

Addressing the critical clinical distinction between physiological ovulation and endocrine pathology, the platform implements **3-Class Ovarian Ultrasound Classification**:
1. **Normal Ovary (`NORMAL`)**: Physiological resting follicular reserve ($<12$ follicles, balanced stroma).
2. **Polycystic Ovary Syndrome (`PCOS`)**: Polyfollicular morphology meeting international Rotterdam criteria ($\ge 12$ follicles in peripheral subcapsular distribution around thickened hyperechoic stroma).
3. **Dominant Follicle (`DF`)**: Pre-ovulatory Graafian follicle ($\ge 10\text{ mm}$ expansion, reaching 18–25 mm prior to ovulation), preventing ovulating patients from receiving false-positive PCOS flags.

---

## 2. Top-2 Production Model Checkpoints

This clean repository includes **only the Top-2 performing production architectures**:

1. **ConvNeXt-Tiny V6 — Primary Champion Model**:
   - **Path**: `models/V6_ConvNeXt-Tiny/best_convnext_tiny_3class.pth` (111.36 MB, 27.82M parameters)
   - **Architecture**: Modern pure convolutional network with 7×7 depthwise convolutions and inverted bottleneck stages.
   - **Performance**: **84.88% Test Accuracy, 0.8506 Macro-F1, 100.00% PCOS Sensitivity (29/29 detected — Zero False Negatives on PCOS), 86.2% DF Sensitivity**.
   - **Latency**: 41.7 ms on GPU / CPU.

2. **ResNet-50 — Deep Residual Top-2 Benchmark**:
   - **Path**: `models/ResNet50/best_resnet50_3class.pth` (94.38 MB, 23.51M parameters)
   - **Architecture**: Deep residual network with bottleneck residual blocks.
   - **Performance**: **81.40% Test Accuracy, 0.8122 Macro-F1, 96.6% Dominant Follicle Recall (28/29 detected), 100.00% PCOS Precision**.
   - **Latency**: 153.2 ms.

---

## 3. Dataset Architecture (`real time data/`)

The repository operates on authentic clinical ovarian ultrasound scans organized into three splits:

```
real time data/
├── train/
│   ├── DF/          (128 clinical ultrasound scans)
│   ├── NORMAL/      (134 clinical ultrasound scans)
│   └── PCOS/        (135 clinical ultrasound scans)
├── val/
│   ├── DF/          (29 clinical ultrasound scans)
│   ├── NORMAL/      (29 clinical ultrasound scans)
│   └── PCOS/        (29 clinical ultrasound scans)
└── test/
    ├── DF/          (29 clinical ultrasound scans)
    ├── NORMAL/      (28 clinical ultrasound scans)
    └── PCOS/        (29 clinical ultrasound scans)
```

- **Total Dataset**: 570 authentic clinical ultrasound scans across training (397), validation (87), and locked test (86).

---

## 4. Native-Resolution Follicle Detection & Explainability

- **Acoustic Sector Masking**: Convex hull filtering isolating ovarian parenchyma while zeroing transducer boundaries and scan text.
- **Multi-Scale Black-Hat Morphometry**: Detects both antral follicles (2–9 mm) and dominant Graafian follicles ($\ge 10\text{ mm}$).
- **Multi-Mode LayerCAM Visualizations**:
  1. `LayerCAM Thermal`: Gradient-weighted feature attributions mapped over acoustic anatomy.
  2. `Follicle Calipers & Bounds`: True-to-scale follicle calipers (amber for dominant $\ge 10$mm, cyan for antral).
  3. `AI Concordance`: Deep learning attention overlap with segmented follicle contours.
  4. `Tri-Panel Inspection`: Side-by-side comparative inspection (B-mode | Follicle Contours | LayerCAM).
  5. `Edge Saliency`: High-frequency acoustic margin enhancement.

---

## 5. Quickstart & Launch Instructions

### Run the Web PACS Application:
From workspace root:
```bash
python app.py
```
Or from the project directory:
```bash
python -m flask --app app run --port 5050
```
Open your browser at **`http://localhost:5050`**.

### CLI Single-Image Prediction:
```bash
python -m src.predict --image "real time data/test/DF/DF_Case_007.png" --model convnext
```

### Full Benchmark Evaluation on Locked Test Set:
```bash
python -m src.evaluate
```

---

## 6. Project Directory Clean Structure

```
├── app.py                      # Root launcher
├── PCOS_Project_PS-main/
│   ├── app.py                  # Main Flask PACS API backend
│   ├── requirements.txt        # Python dependencies
│   ├── models/                 # Top-2 Checkpoints Only
│   │   ├── ResNet50/           # best_resnet50_3class.pth
│   │   └── V6_ConvNeXt-Tiny/   # best_convnext_tiny_3class.pth
│   ├── real time data/         # Authentic Clinical 3-Class Dataset
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── src/                    # Clean modular backend
│   │   ├── dataset.py          # 3-class PyTorch dataset & loader
│   │   ├── evaluate.py         # 3-class benchmark evaluation
│   │   ├── explainability.py   # LayerCAM & concordance engine
│   │   ├── follicle_detector.py# Native resolution follicle caliper detector
│   │   ├── models.py           # ConvNeXt-Tiny & ResNet-50 builders
│   │   ├── predict.py          # CLI inference tool
│   │   └── preprocessing.py    # Acoustic gating & letterboxing
│   ├── static/
│   │   ├── css/style.css       # PACS workstation styling
│   │   ├── js/app.js           # Interactive UI logic
│   │   └── images/examples/    # Clinical demo scans
│   └── templates/
│       └── index.html          # PACS workstation web application
```
