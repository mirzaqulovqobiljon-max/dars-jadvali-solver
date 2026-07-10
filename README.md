# Dars jadvali — OR-Tools CP-SAT solver (Python xizmati)

Bu papka dars jadvalini **Google OR-Tools CP-SAT** yordamida optimal tuzuvchi
Python web-xizmati. HTML sayt (Netlify) shu xizmatga so'rov yuboradi va tayyor
jadvalni oladi.

## Fayllar
- `solver.py` — CP-SAT constraint modeli (asosiy mantiq)
- `main.py` — FastAPI web API (`/generate`, `/health`)
- `requirements.txt` — kutubxonalar
- `Dockerfile` — joylashtirish uchun

## Lokal ishga tushirish (kompyuterda sinash)
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
Keyin brauzerda: http://localhost:8000/health  → `{"status":"ok"}`

## Cheklovlar (constraints)
**Qat'iy (hech qachon buzilmaydi):**
- Sinf bir vaqtda bitta dars
- O'qituvchi bir vaqtda bitta dars (parallel yo'q) — bo'linadigan darsda 2-o'qituvchi ham
- Metodik kun / o'qituvchi bandligi / fan kun-soat cheklovi
- Xona bir vaqtda bitta dars
- Bir kunda bir xil fan ko'pi bilan 1 marta
- O'qituvchi haftalik maksimal soati
- **Bo'sh oyna yo'q** — har sinf har kunda darslar 1-soatdan ketma-ket

**Optimallashtiriladi (soft):**
- Imkon qadar ko'p dars joylashtirilsin
- Kunlar muvozanati (bir kun juda ko'p / kam bo'lmasin)
- Og'ir fanlar ertaroq soatlarga

## Joylashtirish (internetga chiqarish)

### Variant A — Render.com (tavsiya, bepul tarif bor)
1. https://render.com da ro'yxatdan o'ting
2. Bu papkani GitHub repozitoriyaga yuklang
3. Render → **New → Web Service** → repozitoriyani tanlang
4. Environment: **Docker** (Dockerfile avtomatik topiladi)
5. Deploy tugagach, sizga manzil beriladi, masalan:
   `https://dars-jadvali-solver.onrender.com`
6. Shu manzilni HTML saytdagi `SOLVER_API` ga yozing (pastga qarang)

### Variant B — Railway.app yoki Google Cloud Run
Dockerfile bilan xuddi shunday: repozitoriyani ulang, deploy qiling, manzilni oling.

## Saytga ulash
HTML fayl (index.html) ichida yuqoridagi qatorni toping:
```js
var SOLVER_API = "";
```
va xizmat manzilini yozing:
```js
var SOLVER_API = "https://dars-jadvali-solver.onrender.com";
```
Saqlab, saytni Netlify'ga qayta yuklang.

Endi "Jadvalni yaratish" bosilganda sayt CP-SAT xizmatiga so'rov yuboradi.
Agar xizmat ishlamasa (manzil bo'sh yoki server o'chiq), sayt avtomatik ravishda
ichki JS generatoriga o'tadi — ya'ni har doim ishlaydi.

## Eslatma (bepul tariflar haqida)
Render bepul tarifda xizmat bir muddat ishlatilmasa "uxlaydi" — birinchi so'rov
20-40 soniya sekin bo'lishi mumkin (uyg'onish vaqti). Doimiy tez ishlashi uchun
pullik tarif yoki "always on" sozlamasi kerak.
