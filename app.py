import json
import os
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from threading import Lock

import birdnet
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

app = FastAPI(title="Prisme Audio Recognition", version="1.3.0")
_model = None
_model_lock = Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = birdnet.load("acoustic", "2.4", "tf", library="litert")
    return _model


def split_species(value: str):
    value = str(value or "").strip()
    if "_" in value:
        scientific, common = value.split("_", 1)
        return scientific.replace("_", " "), common.replace("_", " ")
    return "", value.replace("_", " ")


def normalize_rows(predictions):
    if predictions is None:
        return []
    if isinstance(predictions, list):
        return predictions
    if isinstance(predictions, tuple):
        return list(predictions)
    if hasattr(predictions, "to_structured_array"):
        structured = predictions.to_structured_array()
        rows = []
        for record in structured:
            row = {}
            for name in structured.dtype.names or ():
                value = record[name]
                row[name] = value.item() if hasattr(value, "item") else value
            rows.append(row)
        return rows
    if hasattr(predictions, "to_pylist"):
        return predictions.to_pylist()
    if hasattr(predictions, "to_dicts"):
        return predictions.to_dicts()
    if hasattr(predictions, "to_pandas"):
        return predictions.to_pandas().to_dict(orient="records")
    if hasattr(predictions, "to_dict"):
        try:
            converted = predictions.to_dict(orient="records")
        except TypeError:
            converted = predictions.to_dict()
        if isinstance(converted, list):
            return converted
        if isinstance(converted, dict):
            columns = {}
            for key, values in converted.items():
                if isinstance(values, dict):
                    columns[key] = list(values.values())
                elif isinstance(values, (list, tuple)):
                    columns[key] = list(values)
                else:
                    columns[key] = [values]
            row_count = max((len(values) for values in columns.values()), default=0)
            return [
                {key: values[index] for key, values in columns.items() if index < len(values)}
                for index in range(row_count)
            ]
    return []


def prediction_records(predictions, limit: int):
    rows = normalize_rows(predictions)
    minimum_confidence = float(os.environ.get("BIRDNET_MIN_CONFIDENCE", "0.15"))
    best = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        species = row.get("species_name") or row.get("common_name") or row.get("species") or row.get("label")
        confidence = float(row.get("confidence", row.get("score", 0)) or 0)
        scientific, common = split_species(species)
        # Ignore non-bird events such as "Human vocal" and weak guesses.
        if not scientific or not common or confidence < minimum_confidence:
            continue
        key = common.casefold()
        candidate = {"common_name": common, "scientific_name": scientific, "confidence": confidence}
        if key not in best or best[key]["confidence"] < confidence:
            best[key] = candidate
    return sorted(best.values(), key=lambda item: item["confidence"], reverse=True)[:limit]


@app.get("/")
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "Prisme Audio Recognition",
        "model": "BirdNET 2.4 LiteRT",
        "music": bool(os.environ.get("ACOUSTID_CLIENT_KEY")),
        "music_engine": "AcoustID + Chromaprint",
    }


@app.post("/recognize-music")
async def recognize_music(audio: UploadFile | None = File(default=None), file: UploadFile | None = File(default=None)):
    uploaded = audio or file
    client_key = os.environ.get("ACOUSTID_CLIENT_KEY", "").strip()
    if uploaded is None:
        raise HTTPException(status_code=400, detail="audio or file is required")
    if not client_key:
        raise HTTPException(status_code=503, detail="AcoustID is not configured")
    raw = await uploaded.read()
    if len(raw) < 4_000:
        raise HTTPException(status_code=400, detail="audio sample is too short")
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="audio sample is too large")
    suffix = Path(uploaded.filename or "sample").suffix or ".bin"
    with tempfile.TemporaryDirectory() as temp_dir:
        source = Path(temp_dir) / f"source{suffix}"
        source.write_bytes(raw)
        try:
            calculated = subprocess.run(["fpcalc", "-json", "-length", "120", str(source)], capture_output=True, text=True, timeout=30, check=True)
            fingerprint = json.loads(calculated.stdout)
        except Exception as exc:
            raise HTTPException(status_code=422, detail="unable to fingerprint audio") from exc
    params = urllib.parse.urlencode({
        "client": client_key,
        "duration": int(round(float(fingerprint["duration"]))),
        "fingerprint": fingerprint["fingerprint"],
        "meta": "recordings releasegroups releases compress",
        "format": "json",
    }).encode()
    request = urllib.request.Request("https://api.acoustid.org/v2/lookup", data=params, headers={"User-Agent": "Prisme/1.0 (non-commercial)", "Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=502, detail="AcoustID lookup failed") from exc
    matches = payload.get("results") or []
    if not matches:
        return {"result": None}
    best = max(matches, key=lambda item: float(item.get("score") or 0))
    recordings = best.get("recordings") or []
    if not recordings:
        return {"result": None}
    recording = recordings[0]
    artists = ", ".join(a.get("name", "") for a in recording.get("artists", []) if a.get("name"))
    groups = recording.get("releasegroups") or []
    releases = recording.get("releases") or []
    return {"result": {
        "title": recording.get("title"),
        "artist": artists,
        "album": groups[0].get("title") if groups else None,
        "released": releases[0].get("date") if releases else None,
        "score": best.get("score"),
        "recording_id": recording.get("id"),
        "acoustid": best.get("id"),
    }}


@app.post("/analyze")
async def analyze(audio: UploadFile | None = File(default=None), file: UploadFile | None = File(default=None), meta: str = Form(default="{}")):
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
        conversion = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-ac", "1", "-ar", "48000", str(wav)], capture_output=True, text=True, timeout=30)
        if conversion.returncode != 0 or not wav.exists():
            raise HTTPException(status_code=422, detail="unsupported or unreadable audio")
        try:
            predictions = get_model().predict(str(wav))
            normalized = normalize_rows(predictions)
            results = prediction_records(normalized, limit)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"BirdNET analysis failed: {type(exc).__name__}") from exc
    response = {"results": results, "count": len(results)}
    if not results:
        response["diagnostic"] = {"container": type(predictions).__name__, "rows": len(normalized), "columns": list(normalized[0].keys()) if normalized and isinstance(normalized[0], dict) else []}
    return response
