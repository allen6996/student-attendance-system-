from fastapi import APIRouter
from pydantic import BaseModel
from app.supabase_client import supabase

router = APIRouter()

class StudentCreate(BaseModel):
    name: str
    reg_no: str

@router.post("/")
def add_student(student: StudentCreate):
    result = supabase.table("students").insert(student.model_dump()).execute()
    return {"message": "Student added", "data": result.data}

@router.get("/")
def list_students():
    result = supabase.table("students").select("*").execute()
    return {"students": result.data}
