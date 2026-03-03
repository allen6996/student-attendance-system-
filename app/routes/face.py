from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np

from app.supabase_client import supabase
from app.services.insightface_service import InsightFaceService

router = APIRouter()
face_engine = InsightFaceService()

@router.post("/register/{student_id}")
async def register_face(student_id: str, file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    face, embedding = face_engine.detect_and_embed(frame)

    if face is None:
        return {"error": "No face detected (RetinaFace)"}

    result = (
        supabase.table("student_faces")
        .insert({"student_id": student_id, "embedding": embedding})
        .execute()
    )

    return {"message": "Face registered successfully", "data": result.data}
