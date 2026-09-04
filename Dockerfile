FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libchromaprint-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

ENV BIRDNET_APP_DATA=/data/birdnet
ENV PORT=7860

# Fetch the compact model while building so the first user request never has to
# wait for a model download. The same cache path is used at runtime.
RUN mkdir -p /data/birdnet \
    && python -c 'import birdnet; birdnet.load("acoustic", "2.4", "tf", library="litert")'

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
