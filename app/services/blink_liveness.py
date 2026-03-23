import cv2
import dlib
from scipy.spatial import distance

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("app/models/shape_predictor_68_face_landmarks.dat")

# 🔥 state variables
blink_counter = 0
blink_total = 0
live_counter = 0   # NEW → keeps liveness active

EAR_THRESHOLD = 0.30
EAR_CONSEC_FRAMES = 3


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def check_blink(frame, bbox):

    global blink_counter, blink_total, live_counter

    x1, y1, x2, y2 = bbox

    # 🔥 add padding (important)
    padding = 15
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rect = dlib.rectangle(x1, y1, x2, y2)
    landmarks = predictor(gray, rect)

    leftEye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    rightEye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0

    print("EAR:", ear)

    # ---------- BLINK DETECTION ----------
    if ear < EAR_THRESHOLD:
        blink_counter += 1
    else:
        if blink_counter >= EAR_CONSEC_FRAMES:
            blink_total += 1
        blink_counter = 0

    # ---------- LIVENESS MEMORY ----------
    if blink_total >= 1:
        live_counter = 10   # keep alive for 10 frames
        blink_total = 0

    if live_counter > 0:
        live_counter -= 1
        return True, ear

    return False, ear