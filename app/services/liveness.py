import cv2
import numpy as np

prev_face = None
live_counter = 0

def check_liveness(frame, bbox):
    global prev_face, live_counter

    x1, y1, x2, y2 = map(int, bbox)
    face_img = frame[y1:y2, x1:x2]

    if face_img.size == 0:
        return False, 0.0

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))

    # ---------- 1. TEXTURE ----------
    texture = np.var(gray)

    if texture < 200:   # relaxed
        return False, texture

    # ---------- 2. EDGE CHECK ----------
    edges = cv2.Canny(gray, 50, 120)
    edge_ratio = np.sum(edges > 0) / (64 * 64)

    if edge_ratio > 0.35:   # relaxed
        return False, edge_ratio

    # ---------- 3. MOTION ----------
    if prev_face is None:
        prev_face = gray
        return False, 0.1

    diff = cv2.absdiff(prev_face, gray)
    motion = np.mean(diff)
    prev_face = gray

    # relaxed motion
    if motion > 3:
        live_counter += 1
    else:
        live_counter = max(0, live_counter - 1)

    if live_counter >= 2:
        return True, motion

    return False, motion