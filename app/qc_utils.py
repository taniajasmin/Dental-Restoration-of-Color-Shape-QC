import io
import math
from typing import Dict, Tuple
import numpy as np
import cv2
from skimage import color as skcolor

# -------------------------------
# Helpers
# -------------------------------

def read_image_from_bytes(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

def norm_to_px(img: np.ndarray, bbox_norm: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    """
    bbox_norm = (x, y, w, h) in [0,1] relative to image width/height
    returns integer pixel bbox (x, y, w, h) clipped to image bounds
    """
    H, W = img.shape[:2]
    x = max(0, min(W - 1, int(round(bbox_norm[0] * W))))
    y = max(0, min(H - 1, int(round(bbox_norm[1] * H))))
    w = max(1, int(round(bbox_norm[2] * W)))
    h = max(1, int(round(bbox_norm[3] * H)))
    if x + w > W: w = W - x
    if y + h > H: h = H - y
    return x, y, w, h

def crop_by_norm(img: np.ndarray, bbox_norm: Tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = norm_to_px(img, bbox_norm)
    return img[y:y+h, x:x+w].copy()

def to_lab(img_bgr: np.ndarray) -> np.ndarray:
    # convert BGR [0..255] to LAB (L* a* b*) floats
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lab = skcolor.rgb2lab(rgb)
    return lab

def glare_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Specular highlight mask: bright + low saturation areas."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    # thresholds are mild; tune as needed
    bright = V > 220
    low_sat = S < 40
    glare = np.logical_and(bright, low_sat)
    # dilate a touch
    k = np.ones((3,3), np.uint8)
    glare = cv2.dilate(glare.astype(np.uint8)*255, k, iterations=1) > 0
    return glare

def trimmed_mean(arr: np.ndarray, p: float = 10.0) -> float:
    lo, hi = np.percentile(arr, [p, 100 - p])
    arr2 = arr[(arr >= lo) & (arr <= hi)]
    if arr2.size == 0:
        return float(np.mean(arr))
    return float(np.mean(arr2))

def robust_lab_mean(img_bgr: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    lab = to_lab(img_bgr)
    if mask is None:
        mask = np.ones(lab.shape[:2], dtype=bool)
    # exclude glare pixels
    gmask = ~glare_mask(img_bgr)
    mask = np.logical_and(mask, gmask)
    L = lab[...,0][mask]
    a = lab[...,1][mask]
    b = lab[...,2][mask]
    if L.size < 10:  # fallback
        L = lab[...,0].ravel()
        a = lab[...,1].ravel()
        b = lab[...,2].ravel()
    return np.array([
        trimmed_mean(L, 10),
        trimmed_mean(a, 10),
        trimmed_mean(b, 10)
    ], dtype=np.float32)

def deltaE00(lab1: np.ndarray, lab2: np.ndarray) -> float:
    # skimage expects arrays, we'll wrap them
    x = skcolor.deltaE_ciede2000(lab1.reshape(1,1,3), lab2.reshape(1,1,3))
    return float(x[0,0])

def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def largest_contour_mask(gray: np.ndarray) -> np.ndarray:
    """Get a filled mask of the largest contour (good proxy for tooth region inside ROI)."""
    # blur + Canny
    g = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(g, 50, 150)
    # close gaps
    k = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, k, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return mask

def hu_distance(bin1: np.ndarray, bin2: np.ndarray) -> float:
    # inputs are binary masks 0/255
    m1 = cv2.moments(bin1)
    m2 = cv2.moments(bin2)
    if m1['m00'] == 0 or m2['m00'] == 0:
        return 10.0  # large distance if nothing detected
    h1 = cv2.HuMoments(m1).flatten()
    h2 = cv2.HuMoments(m2).flatten()
    # log transform improves scale
    h1 = -np.sign(h1) * np.log10(np.abs(h1)+1e-12)
    h2 = -np.sign(h2) * np.log10(np.abs(h2)+1e-12)
    return float(np.sum(np.abs(h1 - h2)))

def norm_to_score_dist(d: float, scale: float, cap: float) -> float:
    """Map distance -> score in [0,1]; lower d => higher score."""
    d = min(d, cap)
    return 1.0 - (d / cap) ** 0.7 if cap > 0 else 0.0

def compute_metrics(
    img_clinical: np.ndarray,
    img_lab: np.ndarray,
    roi: Dict
) -> Dict:
    """
    roi schema:
    {
      "clinical": {"tooth":[x,y,w,h], "shade":[x,y,w,h]},
      "lab":      {"tooth":[x,y,w,h], "shade":[x,y,w,h]}
    }
    values normalized in [0,1]
    """
    # --- Crop ROIs
    c_tooth = crop_by_norm(img_clinical, tuple(roi["clinical"]["tooth"]))
    c_shade = crop_by_norm(img_clinical, tuple(roi["clinical"]["shade"]))
    l_tooth = crop_by_norm(img_lab,       tuple(roi["lab"]["tooth"]))
    l_shade = crop_by_norm(img_lab,       tuple(roi["lab"]["shade"]))

    # --- Gray for sharpness
    c_tooth_gray = cv2.cvtColor(c_tooth, cv2.COLOR_BGR2GRAY)
    l_tooth_gray = cv2.cvtColor(l_tooth, cv2.COLOR_BGR2GRAY)

    sharp_c = laplacian_var(c_tooth_gray)
    sharp_l = laplacian_var(l_tooth_gray)
    # normalize sharpness (rough heuristic)
    sharp_norm = lambda s: max(0.0, min(1.0, (s - 80.0) / 400.0))
    sharpness_score = 0.5 * (sharp_norm(sharp_c) + sharp_norm(sharp_l))

    # --- LAB robust means (exclude glare)
    lab_c_tooth = robust_lab_mean(c_tooth)
    lab_c_shade = robust_lab_mean(c_shade)
    lab_l_tooth = robust_lab_mean(l_tooth)
    lab_l_shade = robust_lab_mean(l_shade)

    # --- ΔE00
    dE_c = deltaE00(lab_c_tooth, lab_c_shade)
    dE_l = deltaE00(lab_l_tooth, lab_l_shade)
    # lighting mismatch via shade ΔE across photos
    dE_shade_between = deltaE00(lab_c_shade, lab_l_shade)

    # score from ΔE (5 ~ poor threshold)
    color_score_c = max(0.0, 1.0 - dE_c / 5.0)
    color_score_l = max(0.0, 1.0 - dE_l / 5.0)
    color_score   = 0.5 * (color_score_c + color_score_l)

    # --- Shape masks
    c_mask = largest_contour_mask(c_tooth_gray)
    l_mask = largest_contour_mask(l_tooth_gray)
    # same size for comparison
    H = min(c_mask.shape[0], l_mask.shape[0])
    W = min(c_mask.shape[1], l_mask.shape[1])
    c_mask_r = cv2.resize(c_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    l_mask_r = cv2.resize(l_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    d_hu = hu_distance(c_mask_r, l_mask_r)
    shape_score = norm_to_score_dist(d_hu, scale=1.5, cap=3.0)  # tune cap/scale as needed

    # --- Glare % (penalty)
    def glare_ratio(img):
        g = glare_mask(img)
        return float(g.sum()) / (g.size + 1e-6)
    glare_c = glare_ratio(c_tooth)
    glare_l = glare_ratio(l_tooth)
    glare_penalty = min(0.2, 0.5 * (glare_c + glare_l))  # cap

    # --- Lighting mismatch penalty from shade ROIs
    lighting_mismatch_penalty = min(0.2, dE_shade_between / 20.0)  # ΔE 4 => 0.2 penalty

    # --- Quality & success
    quality = (
        0.45 * color_score +
        0.35 * shape_score +
        0.10 * sharpness_score -
        0.05 * glare_penalty -
        0.05 * lighting_mismatch_penalty
    )
    # sigmoid mapping
    success = 1.0 / (1.0 + math.exp(-3.0 * (quality - 0.6)))

    def bucket(v: float) -> str:
        if v >= 0.8: return "High"
        if v >= 0.6: return "Warning"
        return "Low"

    results = {
        "deltaE": {
            "clinical_tooth_vs_shade": round(dE_c, 2),
            "lab_tooth_vs_shade": round(dE_l, 2),
            "shade_between_images": round(dE_shade_between, 2)
        },
        "scores": {
            "color_score": round(color_score, 3),
            "shape_score": round(shape_score, 3),
            "sharpness_score": round(sharpness_score, 3),
            "glare_penalty": round(glare_penalty, 3),
            "lighting_mismatch_penalty": round(lighting_mismatch_penalty, 3),
            "quality": round(quality, 3),
            "success": round(success, 3),
            "success_bucket": bucket(success),
        },
        "debug": {
            "sharpness_raw": {"clinical": round(sharp_c, 1), "lab": round(sharp_l, 1)},
            "shape_hu_distance": round(d_hu, 3),
            "glare_ratio": {"clinical": round(glare_c, 3), "lab": round(glare_l, 3)}
        }
    }
    return results
