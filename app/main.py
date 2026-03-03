from fastapi import FastAPI
from app.routes import students, face, attendance, stream
from fastapi.staticfiles import StaticFiles
from app.routes import ui
from fastapi.responses import JSONResponse



app = FastAPI(title="Face Attendance System (RetinaFace + ArcFace)")
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)}
    )

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(ui.router, tags=["frontend"])
app.include_router(students.router, prefix="/students", tags=["Students"])
app.include_router(face.router, prefix="/face", tags=["Face Registration"])
app.include_router(attendance.router, prefix="/attendance", tags=["Attendance"])
app.include_router(stream.router, prefix="/stream", tags=["Live Stream"])
