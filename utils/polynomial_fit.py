"""
Robust polynomial helpers for lane-detection visualisation.
-----------------------------------------------------------
Functions
    extract_lane_points(mask, min_pts=30)
    fit_polynomial(x, y, degree=2)      -> coeff, y_min, y_max
    draw_polynomial_lane(img, coeff, y0, y1, ...)
    compute_center_path(coefL, coefR, height, y_min, y_max, ...)
    draw_center_path(img, pts, ...)
"""

import numpy as np
import cv2


# ───────────────── lane-point extraction ────────────────────────────
def extract_lane_points(binary_mask: np.ndarray, min_pts: int = 30):
    ys, xs = np.where(binary_mask > 0)
    if xs.size < min_pts:
        return None, None
    return xs.astype(float), ys.astype(float)


# ───────────────── robust poly-fit (simple RANSAC) ──────────────────
def _ransac_polyfit(x, y, deg=2, iters=15, thresh=3.0):
    n = len(x)
    if n < deg + 1:
        return None
    best_coef, best_inl = None, 0
    for _ in range(iters):
        idx = np.random.choice(n, deg + 1, replace=False)
        coef = np.polyfit(y[idx], x[idx], deg)
        err  = np.abs(np.polyval(coef, y) - x)
        inl  = (err < thresh).sum()
        if inl > best_inl:
            best_inl, best_coef = inl, coef
    return best_coef


def fit_polynomial(x, y, degree: int = 2):
    """Return (coeff, y_min, y_max) or (None, None, None)."""
    if x is None or len(x) < degree + 1:
        return None, None, None
    try:
        coef = np.polyfit(y, x, degree)
    except np.linalg.LinAlgError:
        coef = None
    if coef is None or np.any(np.isnan(coef)):
        coef = _ransac_polyfit(x, y, deg=degree)
    if coef is None:
        return None, None, None
    return coef, y.min(), y.max()


# ───────────────── drawing helpers ──────────────────────────────────
def draw_polynomial_lane(img, coeff, y0, y1,
                         color=(0, 255, 0), thickness=3):
    if coeff is None:
        return img
    h, w = img.shape[:2]
    y0, y1 = int(max(0, y0)), int(min(h - 1, y1))
    ys = np.arange(y0, y1 + 1)
    xs = np.polyval(coeff, ys)
    xs = np.clip(xs, 0, w - 1).astype(int)
    for i in range(len(ys) - 1):
        cv2.line(img, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                 color, thickness)
    return img

def draw_polylines(img, coeff_list, color=(0, 255, 0), thickness=3):
    """
    Draw every polynomial in coeff_list on img.
    Each element may be (coeff, y0, y1)  or just coeff.
    """
    for item in coeff_list:
        if item is None:
            continue
        if isinstance(item, (list, tuple)) and len(item) == 3:
            coeff, y0, y1 = item
        else:                        # fallback full-height
            coeff, y0, y1 = item, 0, img.shape[0]-1
        draw_polynomial_lane(img, coeff, y0, y1,
                             color=color, thickness=thickness)

def compute_center_path(coefL, coefR, height,
                        y_min, y_max, n_pts=60):
    if coefL is None or coefR is None:
        return None
    ys = np.linspace(y_min, y_max, n_pts)
    xs = (np.polyval(coefL, ys) + np.polyval(coefR, ys)) / 2
    pts = [(int(x), int(y)) for x, y in zip(xs, ys)]
    return pts

def compute_center_path_from_list(fits, h, n_pts=60):
    """
    Given a *sorted* list of (coef,y0,y1)  (left→right),
    return center-line pts over overlapping y-range.
    """
    if len(fits) < 2:
        return None
    coefL, yL0, yL1 = fits[0]
    coefR, yR0, yR1 = fits[-1]
    y_min, y_max = max(yL0, yR0), min(yL1, yR1)
    return compute_center_path(coefL, coefR, h, y_min, y_max, n_pts)

def draw_center_path(img, pts, color=(0, 0, 255), thickness=2):
    if not pts:
        return img
    for p1, p2 in zip(pts[:-1], pts[1:]):
        cv2.line(img, p1, p2, color, thickness)
    return img