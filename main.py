"""
Dars jadvali — OR-Tools CP-SAT web API (FastAPI).

Ishga tushirish (lokal):
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000

Endpoint:
    POST /generate   -> jadval yaratadi (JSON kirish/chiqish)
    GET  /health     -> "ok"
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
from solver import solve_timetable

app = FastAPI(title="Dars jadvali solver", version="1.0")

# Frontend (Netlify) dan chaqirish uchun CORS ochiq
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    school: Dict[str, Any]
    teachers: list
    subjects: list
    classes: list
    rooms: list = []
    assignments: list
    maxSeconds: int = 20


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    data = {
        "school": req.school,
        "teachers": req.teachers,
        "subjects": req.subjects,
        "classes": req.classes,
        "rooms": req.rooms,
        "assignments": req.assignments,
    }
    try:
        result = solve_timetable(data, max_seconds=max(3, min(90, int(req.maxSeconds))))
        return result
    except Exception as e:
        # Xizmat hech qachon qulamasin — xatoni tushunarli qaytaramiz
        return {
            "entries": [], "unfilled": [], "status": "ERROR",
            "error": str(e), "stats": {},
        }
