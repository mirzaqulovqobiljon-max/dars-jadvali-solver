"""
Dars jadvali — OR-Tools CP-SAT web API (FastAPI).

Ishga tushirish (lokal):
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000

Endpoint:
    POST /generate    -> jadval yaratadi (JSON kirish/chiqish)
    POST /ai/explain  -> "nega joylashmadi?" tahlili (Gemini orqali)
    GET  /health      -> "ok"

MUHIT O'ZGARUVCHILARI (Render -> Environment):
    GEMINI_API_KEY = <aistudio.google.com/apikey dan olingan kalit>
    GEMINI_MODEL   = gemini-2.5-flash        (ixtiyoriy)
    AI_RATE_PER_HOUR = 30                    (ixtiyoriy, bitta IP uchun)

DIQQAT: kalit hech qachon kod ichida yozilmaydi va javobda qaytarilmaydi.
"""
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from solver import solve_timetable

app = FastAPI(title="Dars jadvali solver", version="1.1")

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
    fixedEntries: list = []
    maxSeconds: int = 20


@app.get("/health")
def health():
    return {"status": "ok", "ai": bool(os.environ.get("GEMINI_API_KEY"))}


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
        result = solve_timetable(data, max_seconds=max(3, min(90, int(req.maxSeconds))))
        return result
    except Exception as e:
        # Xizmat hech qachon qulamasin — xatoni tushunarli qaytaramiz
        return {
            "entries": [], "unfilled": [], "status": "ERROR",
            "error": str(e), "stats": {},
        }


# =====================================================================
#  AI TAHLIL — "Nega joylashmadi?"
# =====================================================================

MAX_SUMMARY_CHARS = 20000          # cheksiz matn yuborilmasin (xarajat himoyasi)
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/"
              "models/{model}:generateContent")

SYSTEM_PROMPT = """Sen O'zbekiston maktablari uchun dars jadvali tizimining yordamchisisan.
Senga jadval tuzilgandan keyingi CHEKLOVLAR XULOSASI beriladi. Ba'zi darslar
jadvalga sig'magan. Sening vazifang — SABABINI aniqlab, aniq yechim taklif qilish.

QOIDALAR:
1. Faqat o'zbek tilida (lotin alifbosida) yoz.
2. Qisqa yoz: 3-6 ta band, har biri 1-2 gap.
3. Har bandda AVVAL sabab, KEYIN aniq yechim bo'lsin.
4. Berilgan raqamlarga tayan. Ma'lumot yetmasa, "ma'lumot yetarli emas" deb yoz —
   hech qachon raqam o'ylab topma.
5. O'qituvchi va sinflar #12 kabi raqamlar bilan berilgan — javobda ham
   AYNAN shu raqamlarni ishlat, ism o'ylab topma.
6. "ORTIQCHA" deb belgilangan qatorlar eng muhim sabab — ularni birinchi yoz.
7. Oxirida bitta eng samarali qadamni "Eng tez yechim:" deb ko'rsat.
8. Markdown sarlavha ishlatma, oddiy matn va "-" belgili ro'yxat yoz."""

_rate = defaultdict(list)          # ip -> [timestamp, ...]


def _rate_ok(ip: str) -> bool:
    limit = int(os.environ.get("AI_RATE_PER_HOUR", "30"))
    now = time.time()
    hits = [t for t in _rate[ip] if now - t < 3600]
    _rate[ip] = hits
    if len(hits) >= limit:
        return False
    hits.append(now)
    return True


class ExplainRequest(BaseModel):
    summary: str
    question: str = ""


@app.post("/ai/explain")
def ai_explain(req: ExplainRequest, ):
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return {"ok": False,
                "error": "AI sozlanmagan: serverda GEMINI_API_KEY muhit o'zgaruvchisi yo'q."}

    summary = (req.summary or "").strip()
    if len(summary) < 20:
        return {"ok": False, "error": "Tahlil uchun ma'lumot yetarli emas."}
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[:MAX_SUMMARY_CHARS]

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    user_text = summary
    if req.question:
        user_text += "\n\nQO'SHIMCHA SAVOL: " + req.question[:500]

    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 900},
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        GEMINI_URL.format(model=model),
        data=data,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", "")
        except Exception:
            pass
        if e.code == 404:
            return {"ok": False,
                    "error": "Model topilmadi: '%s'. GEMINI_MODEL o'zgaruvchisini "
                             "to'g'rilang (masalan gemini-2.5-flash)." % model}
        if e.code in (401, 403):
            return {"ok": False, "error": "API kalit noto'g'ri yoki muddati tugagan."}
        if e.code == 429:
            return {"ok": False, "error": "Kunlik bepul limit tugadi. Ertaga qayta urinib ko'ring."}
        return {"ok": False, "error": "AI xizmati xatosi (%s). %s" % (e.code, detail[:200])}
    except Exception as e:
        return {"ok": False, "error": "AI xizmatiga ulanib bo'lmadi: %s" % str(e)[:200]}

    try:
        parts = payload["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
    except Exception:
        text = ""
    if not text:
        return {"ok": False, "error": "AI bo'sh javob qaytardi. Qayta urinib ko'ring."}
    return {"ok": True, "text": text, "model": model}
