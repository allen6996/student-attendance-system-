import cv2
import numpy as np

prev_face = None
live_counter = 0

def check_liveness(frame, bbox):
    global prev_face, live_counter

    x1, y1, x2, y2 = map(int, bbox)  # ✅ FIXED

    face_img = frame[y1:y2, x1:x2]

    if face_img.size == 0:
        return False, 0.0

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))

    # ---------- TEXTURE CHECK ----------
    texture_var = np.var(gray)

    if texture_var < 300:
        return False, float(texture_var)

    # ---------- MOTION CHECK ----------
    if prev_face is None:
        prev_face = gray
        return False, 0.2

    diff = cv2.absdiff(prev_face, gray)
    motion = np.sum(diff) / 255
    prev_face = gray

    if 50 < motion < 4000:
        live_counter += 1
    else:
        live_counter = max(0, live_counter - 1)

    if live_counter >= 3:
        return True, float(motion)

    return False, float(motion)