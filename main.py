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


def _rank(name: str) -> tuple:
    """Kichik qiymat = afzalroq. Barqaror va arzon modellar oldinda."""
    pref = len(MODEL_PREFERENCE)
    for i, want in enumerate(MODEL_PREFERENCE):
        if want in name:
            pref = i
            break
    unstable = ("preview" in name) + ("exp" in name)
    return (pref, unstable, len(name))


def _candidates(key: str):
    """Sinab ko'rish uchun modellar ro'yxati (birinchisi afzal).

    Bitta modelga tayanmaymiz: bepul tarifda ayrim modellarning kunlik
    kvotasi juda kichik (429), shuning uchun keyingisiga o'tamiz.
    """
    forced = os.environ.get("GEMINI_MODEL", "").strip()
    if forced:
        return [forced]
    now = time.time()
    if _model_cache["name"] and now - _model_cache["at"] < 3600:
        return list(_model_cache["name"])
    names = [n for n in _fetch_models(key)
             if "vision" not in n and "embedding" not in n
             and "image" not in n and "tts" not in n and "live" not in n]
    names.sort(key=_rank)
    names = names[:5]                       # cheksiz urinmaymiz
    if not names:
        raise RuntimeError("Kalit uchun birorta model mavjud emas")
    _model_cache["name"] = names
    _model_cache["at"] = now
    return list(names)


def _pick_model(key: str) -> str:
    return _candidates(key)[0]


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
    try:
        order = _candidates(key)
    except Exception:
        order = []
    return {"ok": True, "count": len(names), "models": names,
            "try_order": order, "selected": picked,
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
        models = _candidates(key)
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

    tried = []
    last_err = None
    for model in models:
        tried.append(model)
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
            if e.code in (401, 403):
                return {"ok": False, "error": "API kalit noto'g'ri yoki muddati tugagan."}
            if e.code in (404, 429, 503):
                # Bu model band/mavjud emas — keyingisini sinaymiz
                _model_cache["at"] = 0       # keshni yangilashga majbur qilamiz
                last_err = (e.code, detail)
                continue
            return {"ok": False,
                    "error": "AI xizmati xatosi (%s). %s" % (e.code, detail[:200])}
        except Exception as e:
            last_err = (0, str(e))
            continue

        try:
            parts = payload["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
        except Exception:
            text = ""
        if not text:
            last_err = (0, "bo'sh javob")
            continue
        return {"ok": True, "text": text, "model": model}

    # Hamma nomzod ishlamadi
    code = last_err[0] if last_err else 0
    if code == 429:
        return {"ok": False,
                "error": "Barcha modellarda limit tugadi (%s). Bepul tarifda "
                         "kunlik/daqiqalik chegara bor — 1-2 daqiqadan keyin yoki "
                         "ertaga qayta urinib ko'ring." % ", ".join(tried)}
    if code == 404:
        return {"ok": False,
                "error": "Ishlaydigan model topilmadi (%s). /ai/models manzilini "
                         "ochib ro'yxatni ko'ring." % ", ".join(tried)}
    return {"ok": False,
            "error": "AI javob bermadi. Sinalgan modellar: %s. %s"
                     % (", ".join(tried), (last_err[1][:150] if last_err else ""))}
