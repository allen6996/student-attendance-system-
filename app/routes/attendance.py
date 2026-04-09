from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
from datetime import datetime, timezone

from app.supabase_client import supabase
from app.services.insightface_service import InsightFaceService
from app.services.face_matching import find_best_match
from app.services.attendance_service import mark_attendance
from app.services.liveness import check_liveness

router = APIRouter()
face_engine = InsightFaceService()

# ── Column names (match your Supabase schema exactly) ──
# attendance table:  id, student_id, date (date), time (timestamp without tz), status (text)
DATE_COL = "date"   # stores "YYYY-MM-DD"
TIME_COL = "time"   # stores full timestamp "YYYY-MM-DD HH:MM:SS"


# ── HELPERS ──────────────────────────────────────────────────────────────────

def _format_row(row: dict) -> dict:
    """
    Convert a raw Supabase attendance row into the shape the frontend expects:
      { name, reg_no, date: "YYYY-MM-DD", time: "HH:MM:SS", status }

    Schema:
      row["date"]  →  "2025-04-06"              (date column)
      row["time"]  →  "2025-04-06T14:35:22"     (timestamp without tz column)
                   or "2025-04-06 14:35:22"
    We extract only the time portion from the timestamp column.
    """
    student = row.get("students") or {}

    date_str = str(row.get(DATE_COL) or "")[:10]   # "YYYY-MM-DD"

    raw_time = str(row.get(TIME_COL) or "")
    # timestamp col looks like "2025-04-06T09:30:00" or "2025-04-06 09:30:00"
    # We only want "HH:MM:SS"
    if "T" in raw_time:
        time_str = raw_time.split("T")[1][:8]
    elif " " in raw_time:
        time_str = raw_time.split(" ")[1][:8]
    else:
        time_str = raw_time[:8]

    return {
        "name":    student.get("name",   "Unknown"),
        "reg_no":  student.get("reg_no", "-"),
        "date":    date_str,
        "time":    time_str,
        "status":  row.get("status", "present"),
    }


# ── 1. GET ALL ATTENDANCE ─────────────────────────────────────────────────────

@router.get("/")
def get_attendance():
    data = (
        supabase.table("attendance")
        .select("*, students(name, reg_no)")
        .order(DATE_COL, desc=True)
        .execute()
    )
    return {"data": [_format_row(r) for r in (data.data or [])]}


# ── 2. MARK FACE ATTENDANCE ───────────────────────────────────────────────────

@router.post("/mark-face")
async def mark_face_attendance(file: UploadFile = File(...)):
    contents = await file.read()
    np_img   = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    faces = face_engine.detect_all_faces(frame)
    if not faces:
        return {"message": "No face detected"}

    stored = supabase.table("student_faces").select("student_id, embedding").execute()
    if not stored.data:
        return {"error": "No registered faces in database"}

    results = []

    for face, embedding in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        # ── LIVENESS ──
        is_live, score = check_liveness(frame, (x1, y1, x2, y2))
        if not is_live:
            results.append({"status": "spoof", "liveness_score": float(score)})
            continue

        # ── FACE CROP ──
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            results.append({"status": "invalid_face_region"})
            continue

        # ── BLUR CHECK ──
        gray       = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 50:
            results.append({"status": "low_clarity", "blur_score": float(blur_score)})
            continue

        # ── MATCH ──
        student_id, score = find_best_match(embedding, stored.data, threshold=0.40)
        if student_id is None:
            results.append({"status": "unknown", "match_score": float(score)})
            continue

        # ── GET NAME ──
        student      = supabase.table("students").select("name").eq("id", student_id).execute()
        student_name = student.data[0]["name"] if student.data else "Unknown"

        # ── MARK ──
        mark_attendance(student_id)

        results.append({
            "status":      "recognized",
            "name":        student_name,
            "match_score": float(score),
            "bbox":        [x1, y1, x2, y2],
        })

    return {
        "message":              "Processed",
        "total_faces_detected": len(faces),
        "results":              results,
    }


# ── 3. MANUAL ATTENDANCE ──────────────────────────────────────────────────────

class ManualAttendancePayload(BaseModel):
    student_id: str
    reg_no:     str
    name:       str
    date:       str   # "YYYY-MM-DD"
    time:       str   # "HH:MM"
    status:     Optional[str] = "present"


@router.post("/mark-manual")
def mark_manual_attendance(payload: ManualAttendancePayload):

    # Validate student exists
    student = (
        supabase.table("students")
        .select("id, name, reg_no")
        .eq("id", payload.student_id)
        .execute()
    )
    if not student.data:
        return {"error": f"Student '{payload.reg_no}' not found in database"}

    # Validate date format
    try:
        datetime.strptime(payload.date, "%Y-%m-%d")
    except ValueError:
        return {"error": f"Invalid date format: {payload.date} (expected YYYY-MM-DD)"}

    # Validate time format
    try:
        datetime.strptime(payload.time, "%H:%M")
    except ValueError:
        return {"error": f"Invalid time format: {payload.time} (expected HH:MM)"}

    # Check for duplicate on the same calendar day
    existing = (
        supabase.table("attendance")
        .select("id")
        .eq("student_id", payload.student_id)
        .eq(DATE_COL, payload.date)
        .execute()
    )
    if existing.data:
        return {
            "error": f"{payload.name} already has attendance marked for {payload.date}"
        }

    # Build the timestamp string for the `time` column:
    # Supabase "timestamp without time zone" expects "YYYY-MM-DD HH:MM:SS"
    ts_str = f"{payload.date} {payload.time}:00"

    result = (
        supabase.table("attendance")
        .insert({
            "student_id": payload.student_id,
            "status":     payload.status or "present",
            DATE_COL:     payload.date,   # "YYYY-MM-DD"
            TIME_COL:     ts_str,         # "YYYY-MM-DD HH:MM:SS"
        })
        .execute()
    )

    if not result.data:
        return {"error": "Failed to insert attendance record"}

    return {
        "message":    f"Attendance marked for {payload.name} on {payload.date} at {payload.time}",
        "student_id": payload.student_id,
        "name":       payload.name,
        "reg_no":     payload.reg_no,
        "date":       payload.date,
        "time":       payload.time,
        "status":     payload.status,
    }