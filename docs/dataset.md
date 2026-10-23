# Dataset Specification & Partitioning Protocol

## 1. Overview & Source
The dataset consists of authentic B-mode transvaginal ultrasound examinations acquired from gynecological clinics. Scans depict authentic ovarian parenchyma, follicular rings, and pelvic background structures.

---

## 2. Partition Breakdown

* **Training Set (`data/train/`)**:
  * Total Images: **946**
  * PCOS Positive: 231 images
  * Non-PCOS (Normal/Other): 715 images
  * Purpose: Model parameter optimization using data augmentation (rotation, horizontal flips, mild color jitter).

* **Validation Set (`data/validation/`)**:
  * Total Images: **164**
  * PCOS Positive: 42 images
  * Non-PCOS: 122 images
  * Purpose: Hyperparameter tuning, early stopping checkpoint selection, temperature scaling, and decision threshold selection ($T=0.39$).

* **Locked Test Set (`data/test/`)**:
  * Total Images: **48**
  * PCOS Positive: 32 images
  * Non-PCOS: 16 images
  * Purpose: Strictly untouched, independent benchmark evaluation. Never loaded during model training or hyperparameter sweeps.

Total authentic images in project: **1,158 ultrasound scans**.

---

## 3. Patient Group Integrity (Zero Data Leakage)
* Transvaginal ultrasound exams frequently include multiple frames per subject.
* Images are tagged with patient identity keys (`group`).
* Group split partitioning guarantees that 100% of images belonging to any given patient identity reside strictly within a single partition.
* This eliminates cross-split identity memorization and ensures real clinical generalization.

---

## 4. Quality Control & Pre-Screening
All images are processed through an automated acoustic domain gate (`src/preprocessing.py`):
1. **Grayscale Consistency**: Rejects non-ultrasound color photographs and graphics ($\Delta \text{RGB} > 14.0$).
2. **Dynamic Range & Signal**: Flags blank dark frames ($\mu < 15.0$, $\sigma < 12.0$) or washed-out white monitors ($\mu > 228.0$).
3. **Speckle Edge Density**: Requires characteristic ultrasound acoustic speckle (Canny density $\ge 0.008$).
