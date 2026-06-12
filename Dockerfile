# Serving image — model is pre-trained and committed, so no training happens here.
FROM python:3.12-slim

# libgomp1 is required by the XGBoost wheel (OpenMP runtime).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY static/ static/
COPY model_service.py main.py ./

# Hugging Face Spaces (and Render) expect the app on 7860 / $PORT.
EXPOSE 7860
ENV PORT=7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
