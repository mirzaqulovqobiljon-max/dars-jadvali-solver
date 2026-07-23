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
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = None
    if key:
        try:
            model = _pick_model(key)
        except Exception:
            model = None
    return {"status": "ok", "ai": bool(key), "model": model}


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


LIST_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Ustuvorlik tartibi: arzon va tez modeldan boshlab qidiramiz.
# Google model nomlarini vaqti-vaqti bilan o'zgartiradi, shuning uchun
# ro'yxat API dan olinadi — kodga qattiq yozilmaydi.
MODEL_PREFERENCE = [
    "flash-lite",
    "flash",
    "pro",
]

_model_cache = {"name": None, "at": 0}


def _fetch_models(key: str):
    """Kalit uchun ruxsat etilgan modellar ro'yxati."""
    req = urllib.request.Request(
        LIST_URL, headers={"x-goog-api-key": key}, method="GET"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    out = []
    for m in payload.get("models", []):
        methods = m.get("supportedGenerationMethods", []) or m.get(
            "supportedActions", []
        )
        if "generateContent" not in methods:
            continue
        name = m.get("name", "")
        if name.startswith("models/"):
            name = name[len("models/"):]
        if not name:
            continue
        out.append(name)
    return out


def _pick_model(key: str) -> str:
    """GEMINI_MODEL berilmagan bo'lsa, mavjudlaridan mosini tanlaydi."""
    forced = os.environ.get("GEMINI_MODEL", "").strip()
    if forced:
        return forced
    now = time.time()
    if _model_cache["name"] and now - _model_cache["at"] < 3600:
        return _model_cache["name"]
    names = _fetch_models(key)
    chosen = None
    for want in MODEL_PREFERENCE:
        cands = [n for n in names if want in n and "vision" not in n
                 and "embedding" not in n and "image" not in n]
        if cands:
            # eng qisqa nom odatda barqaror (preview/exp suffikssiz) versiya
            cands.sort(key=lambda x: (("preview" in x) + ("exp" in x), len(x)))
            chosen = cands[0]
            break
    if not chosen and names:
        chosen = names[0]
    if not chosen:
        raise RuntimeError("Kalit uchun birorta model mavjud emas")
    _model_cache["name"] = chosen
    _model_cache["at"] = now
    return chosen


@app.get("/ai/models")
def ai_models():
    """Kalitingiz uchun qaysi modellar ochiqligini ko'rsatadi (nosozlikni topish uchun)."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return {"ok": False, "error": "GEMINI_API_KEY o'rnatilmagan"}
    try:
        names = _fetch_models(key)
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": "HTTP %s — kalit noto'g'ri bo'lishi mumkin" % e.code}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
    try:
        picked = _pick_model(key)
    except Exception as e:
        picked = None
    return {"ok": True, "count": len(names), "models": names,
            "selected": picked,
            "forced": os.environ.get("GEMINI_MODEL", "").strip() or None}


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

    try:
        model = _pick_model(key)
    except Exception as e:
        return {"ok": False,
                "error": "Model aniqlanmadi: %s. /ai/models manzilini ochib tekshiring."
                         % str(e)[:150]}
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
            _model_cache["name"] = None      # keshni tozalab, keyingi safar qayta qidiramiz
            return {"ok": False,
                    "error": "Model topilmadi: '%s'. Render'dagi GEMINI_MODEL "
                             "o'zgaruvchisini O'CHIRING — tizim mos modelni o'zi "
                             "tanlaydi. Mavjudlarini ko'rish: /ai/models" % model}
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
