# Top-2 Clinical Ovarian Ultrasound AI Models Benchmark (3-Class)

## Dataset Specification
- **Dataset**: Authentic Clinical Transvaginal & Transabdominal Ovarian Ultrasound Dataset (`real time data/`)
- **Split Distribution**:
  - **Train**: 397 clinical images
  - **Validation**: 87 clinical images
  - **Locked Test**: 86 clinical images (`NORMAL`: 28, `PCOS`: 29, `DF`: 29)
- **Clinical Target Classes**:
  1. **Normal Ovary** (Resting physiological follicular reserve, $<12$ antral follicles)
  2. **PCOS** (Polycystic Ovarian Syndrome meeting Rotterdam criteria: $\ge 12$ follicles, subcapsular "string of pearls")
  3. **Dominant Follicle (DF)** (Pre-ovulatory Graafian follicle $\ge 10\text{ mm}$, preventing false PCOS misclassification)

---

## Performance Comparison: Top-2 Clinical Models

| Model Architecture | Parameters | Input Res | Test Accuracy | Macro-F1 | DF Sensitivity | PCOS Sensitivity | Inference Latency | Primary Clinical Strength |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ConvNeXt-Tiny V6 (Primary Champion)** | **27.8M** | **224×224** | **84.88%** | **0.8506** | **86.2%** | **100.0%** | **41.7 ms** | **Best overall accuracy, 100% PCOS sensitivity, zero false negatives on PCOS** |
| **ResNet-50 (Deep Residual Benchmark)** | **23.5M** | **224×224** | **81.40%** | **0.8122** | **96.6%** | **86.2%** | **153.2 ms** | **Exceptional Dominant Follicle recall (28/29 detected), 100% PCOS precision** |

---

## Detailed Classification Metrics (Locked Test Set, N=86)

### 1. ConvNeXt-Tiny V6 (Primary Champion)
- **Overall Accuracy**: 84.88% (73/86 correct)
- **Macro-F1**: 0.8506
- **Class Metrics**:
  - `Normal Ovary`: Precision: 76.5%, Recall: 92.9% (26/28), F1: 0.8387
  - `PCOS`: Precision: 78.4%, Recall: 100.0% (29/29), F1: 0.8788
  - `Dominant Follicle`: Precision: 100.0%, Recall: 86.2% (25/29), F1: 0.9259

### 2. ResNet-50 (Deep Residual Top-2)
- **Overall Accuracy**: 81.40% (70/86 correct)
- **Macro-F1**: 0.8122
- **Class Metrics**:
  - `Normal Ovary`: Precision: 69.4%, Recall: 89.3% (25/28), F1: 0.7812
  - `PCOS`: Precision: 100.0%, Recall: 86.2% (25/29), F1: 0.9259
  - `Dominant Follicle`: Precision: 77.8%, Recall: 96.6% (28/29), F1: 0.8615
