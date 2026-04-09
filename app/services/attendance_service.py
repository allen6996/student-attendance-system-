from datetime import datetime
from app.supabase_client import supabase

# Match your Supabase attendance table schema:
#   date   → date column          stores "YYYY-MM-DD"
#   time   → timestamp without tz stores "YYYY-MM-DD HH:MM:SS"
DATE_COL = "date"
TIME_COL = "time"


def mark_attendance(student_id: str) -> dict | None:
    """
    Insert an attendance record for the given student using the current time.
    Skips silently if the student is already marked present today.

    Inserts:
      date  = "YYYY-MM-DD"           (date column)
      time  = "YYYY-MM-DD HH:MM:SS"  (timestamp without tz column)
      status = "present"
    """
    now       = datetime.utcnow()
    today_str = now.strftime("%Y-%m-%d")           # "2025-04-06"
    ts_str    = now.strftime("%Y-%m-%d %H:%M:%S")  # "2025-04-06 14:35:22"

    # Duplicate guard — one record per student per calendar day
    existing = (
        supabase.table("attendance")
        .select("id")
        .eq("student_id", student_id)
        .eq(DATE_COL, today_str)
        .execute()
    )
    if existing.data:
        print(f"[attendance] Already marked today for student {student_id}")
        return None

    result = (
        supabase.table("attendance")
        .insert({
            "student_id": student_id,
            "status":     "present",
            DATE_COL:     today_str,
            TIME_COL:     ts_str,
        })
        .execute()
    )

    if result.data:
        print(f"[attendance] Marked present: {student_id} at {ts_str}")
        return result.data[0]

    print(f"[attendance] Insert failed for {student_id}")
    return None