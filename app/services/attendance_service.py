from datetime import date
from app.supabase_client import supabase

def mark_attendance(student_id: str):
    today = str(date.today())

    existing = (
        supabase.table("attendance")
        .select("*")
        .eq("student_id", student_id)
        .eq("date", today)
        .execute()
    )

    if existing.data:
        return {"message": "Attendance already marked", "student_id": student_id}

    result = (
        supabase.table("attendance")
        .insert({"student_id": student_id, "date": today, "status": "present"})
        .execute()
    )

    return {"message": "Attendance marked", "data": result.data}
