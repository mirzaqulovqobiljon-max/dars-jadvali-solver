"""
Dars jadvali — OR-Tools CP-SAT web API (FastAPI). XAVFSIZ (hardened) versiya.

Ishga tushirish (lokal):
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000

Endpoint:
    POST /generate   -> jadval yaratadi (JSON kirish/chiqish)
    GET  /health     -> "ok"

XAVFSIZLIK O'ZGARISHLARI:
  1) CORS faqat o'z domeningizga ochiq (ALLOWED_ORIGINS).
  2) Oddiy IP bo'yicha rate-limit (DoS'ga qarshi).
  3) Kirish hajmi cheklangan (ulkan payload -> 413).
  4) maxSeconds yuqori chegarasi 30 ga tushirildi.
  5) Umumiy request-body hajmi cheklangan (413).
"""
import os
import time
import threading
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict

from solver import solve_timetable

# --- Sozlamalar (Render'da Environment Variables orqali o'zgartiring) ---
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS",
    "https://darsjadvali2.netlify.app",   # <-- o'z domeningiz
).split(",") if o.strip()]

RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "20"))        # /generate: N so'rov
RATE_WINDOW = int(os.environ.get("RATE_WINDOW", "60"))       # har RATE_WINDOW soniyada
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(2 * 1024 * 1024)))  # 2 MB
MAX_SECONDS_CAP = int(os.environ.get("MAX_SECONDS_CAP", "30"))

# Kirish massivlari uchun chegaralar (haddan tashqari kattasini rad etamiz)
LIMITS = {
    "teachers": 2000,
    "subjects": 500,
    "classes": 2000,
    "rooms": 1000,
    "assignments": 20000,
    "fixedEntries": 20000,
}

app = FastAPI(title="Dars jadvali solver", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,      # <-- endi "*" EMAS
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ------------------------- Rate limiting -------------------------
_hits = defaultdict(deque)
_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    # Render/Netlify proxy ortida haqiqiy IP shu headerda bo'ladi
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_ok(ip: str) -> bool:
    now = time.time()
    with _lock:
        dq = _hits[ip]
        while dq and now - dq[0] > RATE_WINDOW:
            dq.popleft()
        if len(dq) >= RATE_LIMIT:
            return False
        dq.append(now)
        return True


@app.middleware("http")
async def guard(request: Request, call_next):
    # Katta body'ni rad etamiz (DoS'ga qarshi)
    if request.method == "POST":
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > MAX_BODY_BYTES:
            return JSONResponse(status_code=413, content={"error": "So'rov hajmi juda katta"})
        if request.url.path == "/generate":
            ip = _client_ip(request)
            if not _rate_ok(ip):
                return JSONResponse(
                    status_code=429,
                    content={"error": "Juda ko'p so'rov. Bir oz kuting va qayta urining."},
                )
    return await call_next(request)


# ------------------------- Model / validatsiya -------------------------
class GenerateRequest(BaseModel):
    school: Dict[str, Any]
    teachers: list
    subjects: list
    classes: list
    rooms: list = []
    assignments: list
    fixedEntries: list = []
    maxSeconds: int = Field(default=20, ge=1, le=MAX_SECONDS_CAP)

    @field_validator("teachers", "subjects", "classes", "rooms",
                     "assignments", "fixedEntries")
    @classmethod
    def _cap(cls, v, info):
        limit = LIMITS.get(info.field_name)
        if limit is not None and isinstance(v, list) and len(v) > limit:
            raise ValueError(f"{info.field_name}: juda ko'p element ({len(v)} > {limit})")
        return v


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
        "fixedEntries": req.fixedEntries,
    }
    try:
        secs = max(3, min(MAX_SECONDS_CAP, int(req.maxSeconds)))
        result = solve_timetable(data, max_seconds=secs)
        return result
    except Exception as e:
        # Xizmat hech qachon qulamasin — batafsil ichki xatoni OSHKOR QILMAYMIZ
        return {
            "entries": [], "unfilled": [], "status": "ERROR",
            "error": "Ichki xatolik yuz berdi. Ma'lumotlarni tekshirib qayta urining.",
            "stats": {},
        }
