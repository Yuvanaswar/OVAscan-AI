# Limitations & Regulatory Disclaimer

## 1. Technical & Methodological Limitations

1. **Single-Slice 2D Ultrasound Limitation**:
   * Transvaginal B-mode ultrasound is intrinsically 3D. A single 2D cross-section cannot capture total ovarian volume or complete 360-degree antral follicle counts.
   * Volumetric 3D sweeps or cine video clips represent the next frontier for clinical evaluation.

2. **Scanner & Probe Variability**:
   * Ultrasound machines vary across acoustic frequencies (5–9 MHz), dynamic range settings, and post-processing filters.
   * While the acoustic quality gate suppresses out-of-distribution inputs, performance on unseen hardware requires broader multicenter external cohorts.

3. **Nature of Gradient Attribution**:
   * Saliency maps (LayerCAM and Grad-CAM++) illustrate regions that numerically influenced the CNN's logits.
   * These regions do not constitute definitive radiological segmentation and must not be used as an automated follicle counter.

---

## 2. Clinical Scope & Research Disclaimers

### Research Prototype Notice
This software is an AI research prototype developed strictly for educational and scientific research purposes. It is **not** a certified medical device under FDA (510k), CE-IVD, or MDR regulations.

### Non-Diagnostic Declaration
* This system must **not** be used as a standalone diagnostic tool, and its outputs must not replace professional clinical judgment.
* Diagnostic decisions for Polycystic Ovary Syndrome must incorporate complete clinical criteria (Rotterdam 2003 / NIH 1990), including clinical and biochemical hyperandrogenism, ovulatory dysfunction, and full pelvic examination.
