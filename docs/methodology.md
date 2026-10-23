# Research Methodology: PCOS Ultrasound AI Classification System

## 1. Clinical Context & Problem Formulation
Polycystic Ovary Syndrome (PCOS) is an endocrine disorder affecting 8–13% of reproductive-aged women globally. Diagnostic criteria (Rotterdam 2003) emphasize polycystic ovarian morphology on transvaginal ultrasonography (TVUS), characterized by:
* Enlarged ovarian volume (>10 mL).
* Increased stromal echogenicity and area.
* Excessive small subcapsular follicular distribution (typically >= 12–20 follicles measuring 2–9 mm in diameter per ovary).

Automating morphological assessment from raw B-mode ultrasound frames presents severe technical challenges:
* Acoustic shadowing, speckle noise, and reverberation artifacts.
* Significant intra-observer and inter-observer variability in manual follicle counting.
* Data leakage risks when multiple frames from the same examination or patient are split across train and test sets.

---

## 2. Dataset Partitioning Protocol & Data Leakage Prevention
To ensure peer-reviewed scientific validity:
1. **Patient-Level Group Partitioning**: All scans are grouped by unique patient identity (`group`). No patient contributes scans to more than one split.
2. **Locked Test Partition**: A dedicated 48-image cohort (32 PCOS, 16 Non-PCOS) is permanently frozen as an independent, untouched benchmark.
3. **Validation Strategy**: All model selection, temperature calibration, learning rate schedules, and routing thresholds ($T = 0.39$) were derived exclusively from the 164-image validation set.

---

## 3. Deep Learning Architecture Evolution
* **V5 Baseline (EfficientNet-B0)**:
  * Neural architecture search baseline with depthwise separable convolutions.
  * 4.01M parameters, 35.2 ms CPU inference latency.
  * Achieved 93.75% accuracy and 96.88% sensitivity on the locked test set.
* **V6 Final Model (ConvNeXt-Tiny 224x224)**:
  * Modernized pure convolutional architecture adopting Vision Transformer design principles.
  * Large 7x7 depthwise convolution kernels, inverted bottleneck design, GELU activations, and LayerNorm in place of BatchNorm.
  * 27.82M parameters, 85.3 ms CPU latency.
  * Achieved 93.75% accuracy and **100.00% sensitivity (Zero False Negatives)** on the locked test set.

---

## 4. Training Protocol & Regularization
* **Optimization**: AdamW optimizer ($\text{lr} = 5 \times 10^{-5}$, $\text{weight decay} = 10^{-2}$).
* **Learning Rate Schedule**: Cosine Annealing learning rate decay over 15 epochs.
* **Loss Function**: Cross-Entropy Loss with label smoothing ($\epsilon = 0.05$) to prevent overconfident output distributions.
* **Aspect Ratio Preservation**: Circular follicular geometries are preserved via 1:1 square letterbox padding prior to resizing.

---

## 5. Dual-Opinion Selective Inference Routing
In clinical screening, false negatives carry high risk. The web application deploys an intelligent routing strategy:
* **Confident Range ($P < 0.25$ or $P \ge 0.75$)**: Direct decision by primary V6 ConvNeXt-Tiny.
* **Borderline Range ($0.25 \le P < 0.75$)**: Automated secondary routing through frozen V5 EfficientNet-B0 for multi-model consensus verification.
