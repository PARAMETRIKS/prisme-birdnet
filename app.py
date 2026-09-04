import json
import os
import subprocess
import tempfile
from pathlib import Path
from threading import Lock

import birdnet
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

app = FastAPI(title="Prisme BirdNET", version="1.0.0")
_model = None
_model_lock = Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = birdnet.load("acoustic", "3.0", "onnx")
    return _model


def split_species(value: str):
    value = str(value or "").strip()
    if "_" in value:
        scientific, common = value.split("_", 1)
        return scientific.replace("_", " "), common.replace("_", " ")
    return "", value.replace("_", " ")


def prediction_records(predictions, limit: int):
    if hasattr(predictions, "to_dict"):
        rows = predictions.to_dict(orient="records")
    elif isinstance(predictions, list):
        rows = predictions
    else:
        rows = []

    best = {}
    for row in rows:
        species = row.get("species_name") or row.get("species") or row.get("label")
        confidence = float(row.get("confidence", row.get("score", 0)) or 0)
        scientific, common = split_species(species)
        if not common:
            continue
        key = common.casefold()
        candidate = {
            "common_name": common,
            "scientific_name": scientific,
            "confidence": confidence,
        }
        if key not in best or best[key]["confidence"] < confidence:
            best[key] = candidate
    return sorted(best.values(), key=lambda item: item["confidence"], reverse=True)[:limit]


@app.get("/")
@app.get("/health")
def health():
    return {"ok": True, "service": "Prisme BirdNET", "model": "BirdNET 3.0 ONNX"}


@app.post("/analyze")
async def analyze(
    audio: UploadFile | None = File(default=None),
    file: UploadFile | None = File(default=None),
    meta: str = Form(default="{}"),
):
    uploaded = audio or file
    if uploaded is None:
        raise HTTPException(status_code=400, detail="audio or file is required")

    try:
        options = json.loads(meta or "{}")
    except json.JSONDecodeError:
        options = {}
    limit = max(1, min(10, int(options.get("num_results", 5))))

    raw = await uploaded.read()
    if len(raw) < 4_000:
        raise HTTPException(status_code=400, detail="audio sample is too short")
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="audio sample is too large")

    suffix = Path(uploaded.filename or "sample").suffix or ".bin"
    with tempfile.TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / f"source{suffix}"
        wav = Path(temp_dir) / "sample.wav"
        source.write_bytes(raw)
        conversion = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-ac", "1", "-ar", "48000", str(wav)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if conversion.returncode != 0 or not wav.exists():
            raise HTTPException(status_code=422, detail="unsupported or unreadable audio")
        try:
            predictions = get_model().predict(str(wav))
            results = prediction_records(predictions, limit)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"BirdNET analysis failed: {type(exc).__name__}") from exc

    return {"results": results, "count": len(results)}
