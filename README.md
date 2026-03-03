# Face Attendance System (FastAPI + Supabase + Docker)

Working starter template using RetinaFace (detection) + ArcFace (embedding) via InsightFace.

## Setup
1) Copy `.env.example` to `.env` and fill Supabase credentials.
2) Create DB tables using `app/database/db_schema.sql`.

## Run
```bash
docker-compose up --build
```

## Swagger
http://localhost:8000/docs

## Live Stream
http://localhost:8000/stream/video
