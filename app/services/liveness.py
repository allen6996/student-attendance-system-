import cv2
import numpy as np

# ─────────────────────────────────────────────────
# Tunable constants — screen/photo spoof detection
# ─────────────────────────────────────────────────
TEXTURE_MIN      = 350    # Laplacian variance floor  (flat photos = low)
TEXTURE_MAX      = 6000   # Laplacian variance ceiling (screen glare = high)
SPECULAR_MAX     = 0.04   # fraction of overly-saturated "screen bleed" pixels
FREQ_RATIO_MAX   = 0.18   # high-freq edge ratio ceiling (LCD pixel grid / print dots)


def _texture_score(gray64: np.ndarray) -> float:
    """Laplacian variance — real faces have varied micro-texture.
    Printed photos are unnaturally flat; phone screens are unnaturally sharp."""
    lap = cv2.Laplacian(gray64, cv2.CV_64F)
    return float(lap.var())


def _specular_ratio(face_bgr: np.ndarray) -> float:
    """Fraction of pixels with high value AND high saturation.
    Phone/tablet screens produce strong bleed around light areas that
    real skin does not. Printed photos on glossy paper also trigger this."""
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] > 240) & (hsv[:, :, 1] > 200)
    return float(np.sum(mask)) / max(1, face_bgr.shape[0] * face_bgr.shape[1])


def _high_freq_ratio(gray64: np.ndarray) -> float:
    """Ratio of Canny edge pixels in a 64x64 face crop.
    LCD pixel grids and dot-matrix prints create unnaturally
    uniform fine edges that real skin does not have."""
    edges = cv2.Canny(gray64, 60, 130)
    return float(np.sum(edges > 0)) / (64 * 64)


def check_liveness(frame: np.ndarray, bbox: tuple) -> tuple[bool, float]:
    """
    Screen / photo spoof detection using three passive texture layers:
      1. Texture variance  — rejects flat printed photos
      2. Specular ratio    — rejects phone/tablet screens
      3. High-freq edges   — rejects LCD pixel grids & dot-matrix prints

    No blink detection, no head-movement tracking.

    Args:
        frame: Full BGR frame from camera.
        bbox:  (x1, y1, x2, y2) face bounding box.

    Returns:
        (is_live: bool, score: float)
        score is the texture variance when passing, or the failing metric value.
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Add small padding for better crop
    pad = 10
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    face_bgr = frame[y1:y2, x1:x2]
    if face_bgr.size == 0:
        return False, 0.0

    gray64 = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray64 = cv2.resize(gray64, (64, 64))

    # ── LAYER 1: TEXTURE VARIANCE ────────────────────
    texture = _texture_score(gray64)
    if not (TEXTURE_MIN < texture < TEXTURE_MAX):
        print(f"[liveness] FAIL texture={texture:.1f}  "
              f"(expected {TEXTURE_MIN}–{TEXTURE_MAX})")
        return False, texture

    # ── LAYER 2: SPECULAR / SCREEN BLEED ────────────
    spec = _specular_ratio(face_bgr)
    if spec > SPECULAR_MAX:
        print(f"[liveness] FAIL specular={spec:.4f}  (max {SPECULAR_MAX})")
        return False, spec

    # ── LAYER 3: HIGH-FREQUENCY EDGE RATIO ──────────
    freq = _high_freq_ratio(gray64)
    if freq > FREQ_RATIO_MAX:
        print(f"[liveness] FAIL freq={freq:.4f}  (max {FREQ_RATIO_MAX})")
        return False, freq

    print(f"[liveness] PASS texture={texture:.1f} spec={spec:.4f} freq={freq:.4f}")
    return True, texture


def reset_liveness_state():
    """No-op — kept for API compatibility with older callers."""
    pass
