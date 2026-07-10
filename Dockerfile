FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY solver.py main.py ./

# Render/Railway/Cloud Run PORT muhit o'zgaruvchisini beradi
ENV PORT=8000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
