import cv2
import numpy as np
import dlib
from scipy.spatial import distance

# ─────────────────────────────────────────────────
# Load dlib models (do this once at module level)
# ─────────────────────────────────────────────────
_detector  = dlib.get_frontal_face_detector()
_predictor = dlib.shape_predictor("app/models/shape_predictor_68_face_landmarks.dat")

# ─────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────
_prev_face      = None
_motion_buffer  = []   # rolling window of motion scores
_blink_counter  = 0
_blink_total    = 0
_live_counter   = 0

# ─────────────────────────────────────────────────
# Tunable constants
# ─────────────────────────────────────────────────
EAR_THRESHOLD       = 0.24   # EAR below this = eye closed
EAR_CONSEC_FRAMES   = 2      # consecutive frames needed = 1 blink
MOTION_WINDOW       = 8      # rolling average over N frames
MOTION_THRESHOLD    = 1.8    # min mean motion to count as alive
TEXTURE_MIN         = 350    # pixel variance floor  (photos = flat)
TEXTURE_MAX         = 5500   # pixel variance ceiling (screen glare)
FREQ_RATIO_MAX      = 0.18   # high-freq edge ratio ceiling (screens/prints)
SPECULAR_MAX        = 0.04   # fraction of saturated pixels allowed
LIVE_HOLD_FRAMES    = 20     # frames liveness stays True after blink


def _eye_aspect_ratio(landmarks, start, end):
    pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(start, end)]
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)


def _specular_ratio(face_bgr):
    """Fraction of pixels that are highly saturated (reflective screen/glare)."""
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    # High value + low saturation = specular highlight on real skin
    # Very high value + very high saturation = screen bleed / laminate glare
    mask = (hsv[:,:,2] > 240) & (hsv[:,:,1] > 200)
    return np.sum(mask) / max(1, face_bgr.shape[0] * face_bgr.shape[1])


def _texture_score(gray64):
    """Laplacian variance — real faces have varied micro-texture."""
    lap = cv2.Laplacian(gray64, cv2.CV_64F)
    return lap.var()


def _high_freq_ratio(gray64):
    """
    Ratio of high-frequency edge pixels via Canny.
    Printed photos + phone screens tend to show sharp uniform edges
    due to printing dots / pixel grid patterns.
    """
    edges = cv2.Canny(gray64, 60, 130)
    return np.sum(edges > 0) / (64 * 64)


def check_liveness(frame, bbox):
    """
    Multi-layer liveness detection combining:
      1. Texture analysis  (rejects flat printed photos)
      2. Specular check    (rejects phone/tablet screens)
      3. Frequency check   (rejects dot-matrix prints & LCD pixels)
      4. Micro-motion      (rejects perfectly still photo)
      5. Blink detection   (requires actual eye movement)

    Returns:
        (is_live: bool, score: float)
    """
    global _prev_face, _motion_buffer, _blink_counter, _blink_total, _live_counter

    x1, y1, x2, y2 = map(int, bbox)

    # Add padding for better landmark accuracy
    pad = 20
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    face_bgr = frame[y1:y2, x1:x2]
    if face_bgr.size == 0:
        return False, 0.0

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray64    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray64    = cv2.resize(gray64, (64, 64))

    # ── LAYER 1: TEXTURE ──────────────────────────────
    texture = _texture_score(gray64)
    if not (TEXTURE_MIN < texture < TEXTURE_MAX):
        print(f"[liveness] FAIL texture={texture:.1f}")
        return False, texture

    # ── LAYER 2: SPECULAR / SCREEN GLARE ─────────────
    spec = _specular_ratio(face_bgr)
    if spec > SPECULAR_MAX:
        print(f"[liveness] FAIL specular={spec:.4f}")
        return False, spec

    # ── LAYER 3: HIGH-FREQUENCY EDGE RATIO ───────────
    freq = _high_freq_ratio(gray64)
    if freq > FREQ_RATIO_MAX:
        print(f"[liveness] FAIL freq={freq:.4f}")
        return False, freq

    # ── LAYER 4: MICRO-MOTION ─────────────────────────
    if _prev_face is not None:
        diff   = cv2.absdiff(_prev_face, gray64)
        motion = float(np.mean(diff))
        _motion_buffer.append(motion)
        if len(_motion_buffer) > MOTION_WINDOW:
            _motion_buffer.pop(0)
        avg_motion = np.mean(_motion_buffer)
    else:
        avg_motion = 0.0

    _prev_face = gray64.copy()

    # ── LAYER 5: BLINK DETECTION ──────────────────────
    try:
        rect      = dlib.rectangle(x1, y1, x2, y2)
        landmarks = _predictor(gray_full, rect)

        left_ear  = _eye_aspect_ratio(landmarks, 36, 42)
        right_ear = _eye_aspect_ratio(landmarks, 42, 48)
        ear       = (left_ear + right_ear) / 2.0

        print(f"[liveness] EAR={ear:.3f} motion={avg_motion:.2f} "
              f"texture={texture:.0f} spec={spec:.4f} freq={freq:.3f}")

        if ear < EAR_THRESHOLD:
            _blink_counter += 1
        else:
            if _blink_counter >= EAR_CONSEC_FRAMES:
                _blink_total  += 1
                _live_counter  = LIVE_HOLD_FRAMES   # confirmed blink → stay live
                print(f"[liveness] ✅ BLINK confirmed  total={_blink_total}")
            _blink_counter = 0

    except Exception as e:
        print(f"[liveness] landmark error: {e}")
        # If landmarks fail, fall back to motion-only liveness
        if avg_motion > MOTION_THRESHOLD and len(_motion_buffer) >= 3:
            _live_counter = max(_live_counter, 4)

    # ── DECISION ──────────────────────────────────────
    if _live_counter > 0:
        _live_counter -= 1
        return True, ear if 'ear' in dir() else avg_motion

    # Partial liveness: strong motion alone (fallback for rare landmark failures)
    if avg_motion > MOTION_THRESHOLD * 2 and len(_motion_buffer) >= MOTION_WINDOW:
        return True, avg_motion

    return False, avg_motion


def reset_liveness_state():
    """Call this between sessions / when switching subjects."""
    global _prev_face, _motion_buffer, _blink_counter, _blink_total, _live_counter
    _prev_face     = None
    _motion_buffer = []
    _blink_counter = 0
    _blink_total   = 0
    _live_counter  = 0
