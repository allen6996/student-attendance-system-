from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np

from app.supabase_client import supabase
from app.services.insightface_service import InsightFaceService
from app.services.face_matching import find_best_match
from app.services.attendance_service import mark_attendance

router = APIRouter()
face_engine = InsightFaceService()


@router.post("/mark-face")
async def mark_face_attendance(file: UploadFile = File(...)):

    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # detect ALL faces
    faces = face_engine.detect_all_faces(frame)

    if not faces:
        return {"message": "No face detected"}

    stored = supabase.table("student_faces") \
        .select("student_id, embedding") \
        .execute()

    if not stored.data:
        return {"error": "No registered faces in database"}

    results = []

    for face, embedding in faces:

        # ---------- FACE REGION ----------
        x1, y1, x2, y2 = map(int, face.bbox)
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            results.append({
                "status": "invalid_face_region"
            })
            continue

        # ---------- BLUR CHECK ----------
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < 50:
            results.append({
                "status": "low_clarity",
                "blur_score": float(blur_score)
            })
            continue

        # ---------- FACE MATCH ----------
        student_id, score = find_best_match(
            embedding,
            stored.data,
            threshold=0.40
        )

        if student_id is None:
            results.append({
                "status": "unknown",
                "match_score": float(score)
            })
            continue

        # ---------- GET STUDENT NAME ----------
        student = supabase.table("students") \
            .select("name") \
            .eq("id", student_id) \
            .execute()

        student_name = "Unknown"
        if student.data:
            student_name = student.data[0]["name"]

        # ---------- MARK ATTENDANCE ----------
        mark_attendance(student_id)

        results.append({
            "status": "recognized",
            "name": student_name,
            "match_score": float(score),
            "bbox": [x1, y1, x2, y2]
        })

    return {
        "message": "Processed",
        "total_faces_detected": len(faces),
        "results": results
    }