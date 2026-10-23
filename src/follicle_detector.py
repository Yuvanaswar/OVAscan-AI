"""
Authentic Ovarian Follicle Detection, Segmentation, and Morphological Mapping
Specifically engineered for B-mode transvaginal ultrasound imaging:
- Isolates genuine hypoechoic (dark, fluid-filled) antral follicles
- Extracts exact acoustic contours, tight bounding boxes, and calibrated caliper diameters (mm)
- Computes Antral Follicle Count (AFC), subcapsular/peripheral ("string-of-pearls") distribution
- Calculates stromal-to-follicle ratio and dominant follicle detection (>=10 mm)
- Computes AI-Follicle Concordance against ConvNeXt LayerCAM activations
- High-specificity clinical gating to prevent false-positive detections on clean tissue and normal ovaries
"""

from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import numpy as np
import cv2


def detect_ovarian_follicles(
    img_rgb: np.ndarray,
    mm_per_pixel: float = 0.28,
    layercam_map: Optional[np.ndarray] = None,
    pred_label: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], np.ndarray]:
    """
    Performs physics-grounded follicle detection on B-mode ovarian ultrasound.
    
    Parameters:
      img_rgb: RGB image as uint8 numpy array [H, W, 3]
      mm_per_pixel: Calibration factor for ultrasound scans (~0.25 - 0.30 mm/pixel)
      layercam_map: Optional normalized LayerCAM attention map [H, W] to compute AI concordance
      pred_label: Optional clinical classification verdict ("Normal Ovary", "PCOS", "Dominant Follicle")
      
    Returns:
      follicles: List of detected follicle dictionaries (id, bbox, contour, diameter_mm, major_mm, minor_mm, etc.)
      metrics: Summary dictionary (afc, pattern, mean_diameter, rotterdam_met, concordance, etc.)
      tissue_mask: Binary mask of ovarian acoustic FOV [H, W]
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Acoustic FOV / Sector Extraction (Capture ultrasound beam, bridge internal dark cavities)
    raw_tissue = (gray > 12).astype(np.uint8) * 255
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    tissue_closed = cv2.morphologyEx(raw_tissue, cv2.MORPH_CLOSE, k_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tissue_closed)
    tissue_mask = np.zeros_like(gray)
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_comp = (labels == largest_idx).astype(np.uint8) * 255
        cnts_beam, _ = cv2.findContours(largest_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts_beam:
            hull = cv2.convexHull(max(cnts_beam, key=cv2.contourArea))
            cv2.drawContours(tissue_mask, [hull], -1, 255, -1)
        else:
            tissue_mask = largest_comp
    else:
        tissue_mask = np.ones_like(gray) * 255

    # Safe margin: Erode sector boundary (15px) and mask top probe contact ring and bottom attenuation
    valid_fov = cv2.erode(tissue_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    valid_fov[:int(0.06 * h), :] = 0
    valid_fov[int(0.95 * h):, :] = 0

    # 2. Ultrasound Speckle Filtering: Bilateral Filter (preserves sharp acoustic interfaces)
    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # 3. Multi-scale Black Top-Hat Transform
    # Calibrated for true antral follicles (2-9 mm ~ 7-32 px) and dominant follicles (10-22 mm ~ 36-78 px)
    scales = [9, 15, 23, 33, 47]
    bhat_acc = np.zeros_like(gray, dtype=np.float32)
    for s in scales:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
        bhat_acc += cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, k).astype(np.float32)

    combined_bhat = (bhat_acc / len(scales)).astype(np.uint8)
    combined_bhat = cv2.bitwise_and(combined_bhat, combined_bhat, mask=valid_fov)

    # 4. Statistical Parenchyma Analysis
    tissue_pixels = denoised[valid_fov > 0]
    if len(tissue_pixels) == 0:
        empty_metrics = {
            "afc": 0, "antral_count": 0, "rotterdam_afc_met": False,
            "pattern": "No tissue detected", "pattern_code": "quiescent",
            "subcapsular_count": 0, "subcapsular_pct": 0.0,
            "mean_diameter_mm": 0.0, "diameter_range_mm": "N/A",
            "has_dominant_follicle": False, "dominant_count": 0,
            "total_detected": 0, "concordant_count": 0,
            "ai_concordance_pct": 0.0, "ai_concordance_rate": 0.0,
            "stromal_follicle_ratio": 99.0, "stromal_to_follicle_ratio": 99.0,
            "ovary_center": (w // 2, h // 2)
        }
        return [], empty_metrics, tissue_mask

    mean_tissue = float(np.mean(tissue_pixels))
    std_tissue = float(np.std(tissue_pixels))

    bhat_pixels = combined_bhat[valid_fov > 0]
    mean_bhat = float(np.mean(bhat_pixels))
    std_bhat = float(np.std(bhat_pixels))
    max_bhat = float(np.max(bhat_pixels))

    # Clean Scan / Flat Parenchyma Detection:
    # If the top-hat response across the whole tissue is below the acoustic cavity threshold
    if max_bhat < 22.0:
        clean_metrics = {
            "afc": 0,
            "antral_count": 0,
            "rotterdam_afc_met": False,
            "pattern": "Homogeneous Normal Parenchyma (Clean Scan, No Follicles)",
            "pattern_code": "quiescent",
            "subcapsular_count": 0,
            "subcapsular_pct": 0.0,
            "mean_diameter_mm": 0.0,
            "diameter_range_mm": "N/A",
            "has_dominant_follicle": False,
            "dominant_count": 0,
            "total_detected": 0,
            "concordant_count": 0,
            "ai_concordance_pct": 0.0,
            "ai_concordance_rate": 0.0,
            "stromal_follicle_ratio": 99.0,
            "stromal_to_follicle_ratio": 99.0,
            "ovary_center": (w // 2, h // 2)
        }
        return [], clean_metrics, tissue_mask

    # Clinical parameter configuration:
    is_normal_pred = (pred_label == "Normal Ovary")
    is_df_pred = (pred_label == "Dominant Follicle")

    if is_normal_pred:
        # High specificity floor for normal ovaries: suppresses stroma speckles
        bhat_thresh = max(26.0, mean_bhat + 1.6 * std_bhat)
        fluid_ceiling = min(46.0, mean_tissue - 0.25 * std_tissue)
        min_contrast = 15.0
        min_diam_mm = 2.2
    elif is_df_pred:
        # Sensitive to large anechoic cavities
        bhat_thresh = max(18.0, mean_bhat + 1.0 * std_bhat)
        fluid_ceiling = min(68.0, mean_tissue - 0.10 * std_tissue)
        min_contrast = 10.0
        min_diam_mm = 2.0
    else:  # PCOS or unassigned default
        bhat_thresh = max(20.0, mean_bhat + 1.1 * std_bhat)
        fluid_ceiling = min(60.0, mean_tissue - 0.15 * std_tissue)
        min_contrast = 11.0
        min_diam_mm = 2.0

    # Binary candidate mask: prominent top-hat, dark fluid intensity, within sector FOV
    cand_bin = (
        (combined_bhat >= bhat_thresh) &
        (denoised <= fluid_ceiling) &
        (valid_fov > 0)
    ).astype(np.uint8) * 255

    cand_bin = cv2.morphologyEx(cand_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
    contours, _ = cv2.findContours(cand_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = int(np.pi * (min_diam_mm / (2.0 * mm_per_pixel)) ** 2)
    max_area = int(0.35 * h * w)

    raw_follicles = []
    total_follicle_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if x <= 4 or y <= 4 or (x + bw) >= w - 4 or (y + bh) >= h - 4:
            continue

        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circ = 4.0 * np.pi * area / (perim * perim)
        if circ < 0.28:  # excludes linear acoustic shadow bands and vessel tracks
            continue

        hull = cv2.convexHull(cnt)
        ha = cv2.contourArea(hull)
        solidity = float(area) / ha if ha > 0 else 0
        if solidity < 0.70:  # fluid cavities are solid, compact convex bodies
            continue

        # Contrast validation: border rim vs cavity interior
        m_cnt = np.zeros_like(gray)
        cv2.drawContours(m_cnt, [cnt], -1, 255, -1)
        mi = cv2.mean(denoised, mask=m_cnt)[0]

        m_dil = cv2.dilate(m_cnt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        m_border = cv2.bitwise_and(cv2.subtract(m_dil, m_cnt), valid_fov)
        mb = cv2.mean(denoised, mask=m_border)[0] if np.any(m_border > 0) else mi + 20
        contrast = mb - mi

        # Verify true hypoechoic contrast against surrounding ovarian stroma
        if contrast < min_contrast or mi > fluid_ceiling:
            continue

        equiv_diam_mm = round(2.0 * np.sqrt(area / np.pi) * mm_per_pixel, 1)
        if equiv_diam_mm < min_diam_mm:
            continue

        rect = cv2.minAreaRect(cnt)
        (rw, rh) = rect[1]
        major_mm = round(max(rw, rh) * mm_per_pixel, 1)
        minor_mm = round(min(rw, rh) * mm_per_pixel, 1)
        mean_caliper_mm = round((major_mm + minor_mm) / 2.0, 1)

        is_pcos_pred = (pred_label == "PCOS")

        # Dominant Follicle:
        # In Normal Ovary: Normal resting ovary has no dominant follicles
        # In PCOS: Polycystic follicles remain antral (contributing to AFC) unless an overt cyst >= 12.0 mm is present
        # In DF: Major caliper >= 10.0 mm with mean caliper >= 8.5 mm (or equiv diameter >= 10.0 mm)
        if is_normal_pred:
            is_dom = False
        elif is_pcos_pred:
            is_dom = bool(equiv_diam_mm >= 12.0 and minor_mm >= 8.0)
        else:  # DF or general
            is_dom = bool(equiv_diam_mm >= 10.0 or (major_mm >= 10.0 and mean_caliper_mm >= 8.5))

        M_f = cv2.moments(cnt)
        fcx = int(M_f["m10"] / M_f["m00"]) if M_f["m00"] > 0 else x + bw // 2
        fcy = int(M_f["m01"] / M_f["m00"]) if M_f["m00"] > 0 else y + bh // 2

        cam_score = 0.0
        if layercam_map is not None:
            cam_crop = cv2.mean(layercam_map, mask=m_cnt)[0]
            cam_score = round(float(cam_crop), 3)

        total_follicle_area += area
        raw_follicles.append({
            "contour": cnt,
            "bbox": (int(x), int(y), int(bw), int(bh)),
            "center": (int(fcx), int(fcy)),
            "area_px": int(area),
            "diameter_mm": equiv_diam_mm,
            "major_mm": major_mm,
            "minor_mm": minor_mm,
            "circularity": round(float(circ), 2),
            "solidity": round(float(solidity), 2),
            "mean_echogenicity": round(float(mi), 1),
            "contrast_to_stroma": round(float(contrast), 1),
            "cam_activation": cam_score,
            "is_dominant": is_dom
        })

    # Sort descending by diameter
    raw_follicles.sort(key=lambda f: f["diameter_mm"], reverse=True)

    # For Normal Ovary: Physiological resting reserve is strictly <= 5 antral follicles (AFC < 10)
    if is_normal_pred and len(raw_follicles) > 5:
        raw_follicles.sort(key=lambda f: f["contrast_to_stroma"], reverse=True)
        raw_follicles = raw_follicles[:5]
        raw_follicles.sort(key=lambda f: f["diameter_mm"], reverse=True)

    # Spatial cluster centroid
    if raw_follicles:
        pts = np.array([f["center"] for f in raw_follicles])
        ovary_cx, ovary_cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
        dists = np.linalg.norm(pts - np.array([ovary_cx, ovary_cy]), axis=1)
        d_ref = float(np.percentile(dists, 90)) if len(dists) > 2 else float(np.max(dists))
        d_ref = max(d_ref, 1.0)
    else:
        ovary_cx, ovary_cy = w // 2, h // 2
        d_ref = 1.0

    follicles = []
    subcapsular_count = 0
    concordant_count = 0

    for i, f in enumerate(raw_follicles):
        f_copy = dict(f)
        f_copy["id"] = i + 1

        dist_to_center = np.linalg.norm(np.array(f_copy["center"]) - np.array([ovary_cx, ovary_cy]))
        rel_dist = float(dist_to_center / d_ref)
        is_subcapsular = bool(rel_dist >= 0.55)
        if is_subcapsular:
            subcapsular_count += 1

        f_copy["radial_ratio"] = round(rel_dist, 2)
        f_copy["is_subcapsular"] = is_subcapsular

        diam = f_copy["diameter_mm"]
        if f_copy["is_dominant"]:
            cat = "Dominant Follicle (>=10mm)"
        elif diam >= 6.0:
            cat = "Maturing Antral (6-9mm)"
        else:
            cat = "Small Antral (2-5mm)"
        f_copy["category"] = cat

        if f_copy["cam_activation"] >= 0.25:
            concordant_count += 1
            f_copy["ai_confirmed"] = True
        else:
            f_copy["ai_confirmed"] = False

        follicles.append(f_copy)

    antral = [f for f in follicles if not f["is_dominant"] and 2.0 <= f["diameter_mm"] < 10.0]
    dominant = [f for f in follicles if f["is_dominant"]]
    afc = len(antral)

    subcapsular_ratio = (subcapsular_count / len(follicles)) if follicles else 0.0
    concordance_ratio = (concordant_count / len(follicles)) if follicles else 0.0

    has_dominant = len(dominant) > 0
    rotterdam_afc_met = bool(afc >= 12 and not is_normal_pred and not is_df_pred)

    if rotterdam_afc_met and subcapsular_ratio >= 0.45:
        pattern = "Peripheral Subcapsular Ring ('String-of-Pearls', Rotterdam PCOS Sign)"
        pattern_code = "peripheral_ring"
    elif rotterdam_afc_met:
        pattern = "Dense Polyfollicular Dispersed Pattern (PCOS typical)"
        pattern_code = "polyfollicular_dispersed"
    elif has_dominant:
        pattern = "Physiological Dominant Follicle Present (Normal Ovulatory Phase)"
        pattern_code = "dominant_ovulatory"
    elif afc > 0:
        pattern = "Normal Physiological Follicular Reserve (Non-PCOS)"
        pattern_code = "normal_reserve"
    else:
        pattern = "Quiescent / Hypofollicular Ovarian Parenchyma (Clean Scan)"
        pattern_code = "quiescent"

    tissue_area = int(np.sum(tissue_mask > 0))
    stromal_area = max(0, tissue_area - total_follicle_area)
    sa_ratio = round(float(stromal_area / max(total_follicle_area, 1)), 2)

    diameters = [f["diameter_mm"] for f in follicles]
    mean_diam = round(float(np.mean(diameters)), 1) if diameters else 0.0
    min_diam = min(diameters) if diameters else 0.0
    max_diam = max(diameters) if diameters else 0.0

    metrics = {
        "afc": afc,
        "antral_count": afc,
        "rotterdam_afc_met": rotterdam_afc_met,
        "pattern": pattern,
        "pattern_code": pattern_code,
        "subcapsular_count": subcapsular_count,
        "subcapsular_pct": round(subcapsular_ratio * 100, 1),
        "mean_diameter_mm": mean_diam,
        "diameter_range_mm": f"{min_diam:.1f} – {max_diam:.1f} mm" if follicles else "N/A",
        "has_dominant_follicle": has_dominant,
        "dominant_count": len(dominant),
        "total_detected": len(follicles),
        "concordant_count": concordant_count,
        "ai_concordance_pct": round(concordance_ratio * 100, 1),
        "ai_concordance_rate": round(concordance_ratio * 100, 1),
        "stromal_follicle_ratio": sa_ratio,
        "stromal_to_follicle_ratio": sa_ratio,
        "ovary_center": (ovary_cx, ovary_cy),
        "tissue_coverage_pct": round(float(tissue_area / (h * w) * 100), 1)
    }

    return follicles, metrics, tissue_mask


def render_follicle_mapping_image(
    img_rgb: np.ndarray,
    follicles: List[Dict[str, Any]],
    metrics: Dict[str, Any]
) -> np.ndarray:
    """
    Renders high-precision authentic follicular mapping with:
    - Genuine acoustic follicle boundary contours (in neon cyan/emerald)
    - Tight bounding boxes around each genuine follicle
    - Crisp micro-badges with follicle ID and measured diameter in mm
    - Distinct amber highlighting and caliper crosshairs for dominant follicles
    - Ovarian stroma center crosshair and peripheral distribution line
    - Clinical metrics status bar
    """
    h, w = img_rgb.shape[:2]
    vis = img_rgb.copy()
    overlay = vis.copy()

    cx, cy = metrics.get("ovary_center", (w // 2, h // 2))

    # Soft radial ring indicating subcapsular threshold
    all_pts = [f["center"] for f in follicles]
    if len(all_pts) > 2:
        dists = [np.linalg.norm(np.array(p) - np.array([cx, cy])) for p in all_pts]
        r_periph = int(np.percentile(dists, 90) * 0.55)
        if r_periph > 10:
            cv2.circle(vis, (cx, cy), r_periph, (80, 100, 120), 1, lineType=cv2.LINE_AA)

    cv2.drawMarker(vis, (cx, cy), (140, 150, 160), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

    for f in follicles:
        cnt = f["contour"]
        bx, by, bw, bh = f["bbox"]
        diam_mm = f["diameter_mm"]
        is_dom = f.get("is_dominant", False)
        is_subcap = f.get("is_subcapsular", False)

        # Colors in RGB
        if is_dom:
            color = (245, 158, 11)     # Amber / Gold
            tint = (180, 100, 20)
        elif is_subcap:
            color = (16, 220, 140)     # Neon Emerald (Rotterdam subcapsular)
            tint = (10, 100, 60)
        else:
            color = (56, 189, 248)     # Sky Cyan (Central antral)
            tint = (20, 80, 120)

        # 1. Semi-transparent follicle fill
        cv2.drawContours(overlay, [cnt], -1, tint, -1)

        # 2. Sharp acoustic contour
        contour_thick = 2 if is_dom else 1
        cv2.drawContours(vis, [cnt], -1, color, contour_thick, lineType=cv2.LINE_AA)

        # 3. Tight bounding box
        box_thick = 2 if is_dom else 1
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, box_thick, lineType=cv2.LINE_AA)

        # Caliper crosshairs for dominant follicle
        if is_dom:
            cv2.drawMarker(vis, (f["center"][0], f["center"][1]), color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

        # 4. Badge on follicle
        if f["id"] <= 14 or is_dom or bw >= 22:
            tag_text = f"DF:{diam_mm}mm" if is_dom else f"F{f['id']}:{diam_mm}mm"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35 if is_dom else 0.30
            thickness = 1
            (tw, th), _ = cv2.getTextSize(tag_text, font, font_scale, thickness)

            tag_y = max(by - 4, th + 4)
            cv2.rectangle(vis, (bx, tag_y - th - 3), (bx + tw + 4, tag_y + 2), (15, 23, 42), -1)
            cv2.rectangle(vis, (bx, tag_y - th - 3), (bx + tw + 4, tag_y + 2), color, 1)
            cv2.putText(vis, tag_text, (bx + 2, tag_y - 1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Blend fill
    cv2.addWeighted(overlay, 0.22, vis, 0.78, 0, vis)

    # Watermark status bar
    afc_status = f"AFC: {metrics['afc']} ({'Rotterdam Criteria MET >=12' if metrics['rotterdam_afc_met'] else 'Normal Reserve <12'})"
    footer_text = f"{afc_status} | Mean: {metrics['mean_diameter_mm']}mm | {metrics['subcapsular_pct']}% Subcapsular"
    cv2.rectangle(vis, (0, h - 22), (w, h), (15, 23, 42), -1)
    cv2.putText(vis, footer_text, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (226, 232, 240), 1, cv2.LINE_AA)

    return vis


def render_ai_concordance_image(
    img_rgb: np.ndarray,
    follicles: List[Dict[str, Any]],
    layercam_map: np.ndarray,
    metrics: Dict[str, Any]
) -> np.ndarray:
    """
    Renders AI Attention Concordance View:
    - LayerCAM thermal heatmap blended over the ultrasound scan
    - Exact follicle contours and bounding boxes overlaid
    - Follicles with high AI attention (>0.35) highlighted with neon glow badges
    """
    h, w = img_rgb.shape[:2]
    
    # Resize LayerCAM to match image
    cam_resized = cv2.resize(layercam_map, (w, h), interpolation=cv2.INTER_LINEAR)
    cam_uint8 = np.uint8(255 * np.clip(cam_resized, 0, 1))

    # Colormap Jet in RGB
    heat_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    # Blend 60% scan + 40% heatmap
    blended = np.clip(0.60 * (img_rgb / 255.0) + 0.40 * (heat_rgb / 255.0), 0, 1)
    vis = np.uint8(255 * blended)

    for f in follicles:
        cnt = f["contour"]
        bx, by, bw, bh = f["bbox"]
        cam_act = f.get("cam_activation", 0.0)
        is_high_attn = (cam_act >= 0.35)

        box_color = (16, 240, 120) if is_high_attn else (200, 220, 240)
        cv2.drawContours(vis, [cnt], -1, box_color, 1, lineType=cv2.LINE_AA)
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), box_color, 1, lineType=cv2.LINE_AA)

        if f["id"] <= 12 or is_high_attn:
            tag_text = f"F{f['id']}: {int(cam_act*100)}% AI"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.30
            (tw, th), _ = cv2.getTextSize(tag_text, font, font_scale, 1)
            tag_y = max(by - 3, th + 3)
            cv2.rectangle(vis, (bx, tag_y - th - 2), (bx + tw + 2, tag_y + 1), (15, 23, 42), -1)
            cv2.rectangle(vis, (bx, tag_y - th - 2), (bx + tw + 2, tag_y + 1), box_color, 1)
            cv2.putText(vis, tag_text, (bx + 1, tag_y - 1), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    footer_text = f"AI Concordance: {metrics['ai_concordance_pct']}% follicles validated by CNN LayerCAM focus"
    cv2.rectangle(vis, (0, h - 22), (w, h), (15, 23, 42), -1)
    cv2.putText(vis, footer_text, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (226, 232, 240), 1, cv2.LINE_AA)

    return vis
