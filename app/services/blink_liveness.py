import cv2
import dlib
from scipy.spatial import distance
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

blink_counter = 0
blink_total = 0
prev_gray = None
motion_frames = 0

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def check_liveness(frame):
    global blink_counter, blink_total, prev_gray, motion_frames

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # ---------- MOTION CHECK ----------
    small = cv2.resize(gray, (64,64))

    if prev_gray is None:
        prev_gray = small
    else:
        diff = cv2.absdiff(prev_gray, small)
        motion = np.sum(diff)/255
        prev_gray = small

        if 80 < motion < 4000:   # natural motion
            motion_frames += 1
        else:
            motion_frames = 0

    # ---------- BLINK CHECK ----------
    for face in faces:
        landmarks = predictor(gray, face)

        leftEye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36,42)]
        rightEye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42,48)]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < 0.26:
            blink_counter += 1
        else:
            if blink_counter >= 2:
                blink_total += 1
            blink_counter = 0

    # ---------- FINAL DECISION ----------
    if blink_total >= 1 and motion_frames >= 2:
        blink_total = 0
        motion_frames = 0
        return True, 0.99

    return False, 0.1
