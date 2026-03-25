"""
import cv2
import dlib
from scipy.spatial import distance

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("app/models/shape_predictor_68_face_landmarks.dat")

# ── State ──
blink_counter = 0
blink_total   = 0
live_counter  = 0

EAR_THRESHOLD    = 0.25   # ← lowered from 0.30 (most webcams need this)
EAR_CONSEC_FRAMES = 2     # ← lowered from 3 (faster response)


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def check_blink(frame, bbox):
    global blink_counter, blink_total, live_counter

    x1, y1, x2, y2 = bbox

    # Padding
    padding = 20
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rect = dlib.rectangle(x1, y1, x2, y2)

    landmarks = predictor(gray, rect)

    leftEye  = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    rightEye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

    ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
    print(f"EAR: {ear:.3f} | blink_counter: {blink_counter} | live_counter: {live_counter}")

    # ── Blink Detection ──
    if ear < EAR_THRESHOLD:
        blink_counter += 1
    else:
        if blink_counter >= EAR_CONSEC_FRAMES:
            blink_total += 1
            print(f"✅ Blink detected! Total: {blink_total}")
            live_counter = 15  # ← Reset liveness IMMEDIATELY on valid blink
        blink_counter = 0

    # ── Liveness Memory ──
    if live_counter > 0:
        live_counter -= 1
        return True, ear

    return False, ear
"""