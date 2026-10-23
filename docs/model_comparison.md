# Model Comparison: V5 Baseline vs. V6 ConvNeXt-Tiny

## 1. Executive Summary
This document details the comparative performance of the baseline architecture (**V5 EfficientNet-B0**) and the final selected architecture (**V6 ConvNeXt-Tiny**) on the frozen 48-image locked test set.

---

## 2. Quantitative Performance Metrics

### V6 ConvNeXt-Tiny 224x224 (Primary Selected Model)
* **Architecture**: ConvNeXt-Tiny (Modernized ConvNet, 7x7 depthwise convolutions)
* **Parameters**: 27.82 Million
* **Inference Latency**: 85.3 ms (CPU)
* **Operating Threshold**: $T = 0.39$
* **Accuracy**: **93.75%** (45 / 48 correct)
* **Sensitivity (Recall)**: **100.00%** (32 / 32 detected — **ZERO False Negatives**)
* **Specificity**: **81.25%** (13 / 16)
* **Precision**: **91.43%** (32 / 35)
* **F1-Score**: **0.9552** (Highest across all evaluated architectures)
* **ROC-AUC**: **0.9609**
* **Confusion Matrix**: True Positives (TP) = 32, True Negatives (TN) = 13, False Positives (FP) = 3, False Negatives (FN) = 0.

### V5 Baseline EfficientNet-B0 (Frozen Comparison Model)
* **Architecture**: EfficientNet-B0 (Compound scaling baseline)
* **Parameters**: 4.01 Million
* **Inference Latency**: 35.2 ms (CPU)
* **Operating Threshold**: $T = 0.39$
* **Accuracy**: **93.75%** (45 / 48 correct)
* **Sensitivity (Recall)**: **96.88%** (31 / 32 detected — 1 False Negative)
* **Specificity**: **87.50%** (14 / 16)
* **Precision**: **93.94%** (31 / 33)
* **F1-Score**: **0.9538**
* **ROC-AUC**: **0.9863**
* **Confusion Matrix**: True Positives (TP) = 31, True Negatives (TN) = 14, False Positives (FP) = 2, False Negatives (FN) = 1.

---

## 3. Rationale for Selecting V6 ConvNeXt-Tiny as Primary

0. **Primary Model Selection Rationale**:
   * V6 ConvNeXt-Tiny is **not** characterized merely as "the most accurate model" (both V5 and V6 achieved identical 93.75% test accuracy).
   * Rather, it was selected as the primary model because of its **100% sensitivity (32/32 detected — Zero False Negatives) and highest locked-test F1-score (0.9552)**, while V5 EfficientNet-B0 is retained as the lightweight baseline/comparison model.

1. **Clinical Screening Priority (Zero False Negatives)**:
   * In a medical screening setting, missing a positive diagnosis (Type II error / False Negative) causes diagnostic delay.
   * V6 ConvNeXt-Tiny detected every single PCOS positive scan in the test partition (Sensitivity = 100.00%), outperforming all other candidates.

2. **Structural Receptive Field Advantage**:
   * The 7x7 depthwise convolutions in ConvNeXt provide a large effective receptive field that captures macroscopic stromal morphology and multi-follicular spatial dispersion.

3. **High F1-Score**:
   * With an F1-Score of 0.9552, ConvNeXt-Tiny delivers an optimal balance of precision and recall.

4. **Complementary Deployment in Selective Routing**:
   * V5 EfficientNet-B0 remains an integral part of the production pipeline, providing fast secondary opinion arbitration for borderline predictions ($0.25 \le P < 0.75$).
