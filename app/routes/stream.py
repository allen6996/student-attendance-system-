from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import cv2

from app.services.insightface_service import InsightFaceService
from app.services.face_matching import find_best_match
from app.supabase_client import supabase

router = APIRouter()

camera = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows to avoid warnings

face_engine = InsightFaceService()

# cache embeddings once (performance)
stored = supabase.table("student_faces").select("student_id, embedding").execute()


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # 🔥 detect all faces
        faces = face_engine.detect_all_faces(frame)

        names_detected = []

        for face, embedding in faces:

            # match face
            student_id, score = find_best_match(
                embedding,
                stored.data,
                threshold=0.40
            )

            if student_id:
                # 🔥 get name
                student = supabase.table("students") \
                    .select("name") \
                    .eq("id", student_id) \
                    .execute()

                if student.data:
                    name = student.data[0]["name"]
                    names_detected.append(name)

        # 🔥 overlay text (no bounding box)
        if names_detected:
            text = ", ".join(names_detected) + " PRESENT"

            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # encode frame
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@router.get("/video")
def video_feed():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )