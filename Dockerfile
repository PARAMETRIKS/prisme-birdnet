FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

ENV BIRDNET_APP_DATA=/data/birdnet
ENV PORT=7860

# Preload the compact model during the build so the first request is fast.
RUN mkdir -p /data/birdnet \
    && python -c 'import birdnet; birdnet.load("acoustic", "2.4", "tf", library="litert")'

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
