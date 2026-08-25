from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageEnhance
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.services import catalog
from ml.categories import CLOUD_PATTERNS, IMD_CATEGORIES, TRACK_PATTERNS
from ml.image_synth import make_sample, to_pil
from ml.infer import gradcam, infer_image, infer_track

app = FastAPI(
    title="CycloneSense",
    description="AI/ML system for identification, classification, and prediction of tropical cyclone patterns over the North Indian Ocean.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Observation(BaseModel):
    time: str | None = None
    lat: float
    lon: float
    wind: float | None = 30
    pres: float | None = None


class PredictRequest(BaseModel):
    observations: list[Observation] = Field(min_length=1)
    hours: int = 72
    include_analogs: bool = True


def _read_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not read image: {exc}") from exc


def _cam_overlay(image: Image.Image, cam: np.ndarray) -> str:
    base = image.convert("RGB").resize((128, 128))
    heat = np.zeros((128, 128, 3), dtype=np.uint8)
    heat[..., 0] = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    heat[..., 1] = (np.clip(cam * 0.4, 0, 1) * 255).astype(np.uint8)
    overlay = Image.blend(base, Image.fromarray(heat), 0.45)
    overlay = ImageEnhance.Contrast(overlay).enhance(1.15)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "name": "CycloneSense",
        "catalog": catalog.STORMS_PATH.exists(),
        "n_storms": len(catalog.storms()),
        "image_model": (ROOT / "data" / "models" / "image_cnn.pt").exists(),
        "track_model": (ROOT / "data" / "models" / "track_lstm.pt").exists(),
    }


@app.get("/api/taxonomy")
def taxonomy() -> dict:
    return {
        "imd_categories": IMD_CATEGORIES,
        "cloud_patterns": CLOUD_PATTERNS,
        "track_patterns": TRACK_PATTERNS,
        "region": {
            "name": "North Indian Ocean",
            "basins": ["Arabian Sea (AS)", "Bay of Bengal (BOB)"],
            "bbox": {"lat": [0, 32], "lon": [45, 105]},
        },
    }


@app.get("/api/stats")
def stats() -> dict:
    return catalog.stats()


@app.get("/api/storms")
def storms(
    season: int | None = None,
    basin: str | None = None,
    category: str | None = None,
    q: str | None = None,
    limit: int = Query(80, le=300),
) -> dict:
    return {"storms": catalog.list_storms(season, basin, category, q, limit)}


@app.get("/api/storms/showcase")
def showcase() -> dict:
    return {"storms": catalog.recent_showcase()}


@app.get("/api/storms/{sid}")
def storm_detail(sid: str) -> dict:
    meta = catalog.get_storm(sid)
    if not meta:
        raise HTTPException(404, "Storm not found")
    track = catalog.storm_track(meta["sid"])
    return {"storm": meta, "track": track}


@app.post("/api/identify")
async def identify(file: UploadFile = File(...)) -> JSONResponse:
    image = _read_image(await file.read())
    result = infer_image(image)
    try:
        cam = gradcam(image)
        result["gradcam"] = _cam_overlay(image, cam)
    except Exception as exc:  # noqa: BLE001
        result["gradcam"] = None
        result["gradcam_error"] = str(exc)
    thumb = image.convert("L").resize((128, 128))
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    result["preview"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return JSONResponse(result)


@app.post("/api/classify")
async def classify(file: UploadFile = File(...)) -> JSONResponse:
    return await identify(file)


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict:
    obs = [o.model_dump() for o in req.observations]
    lstm = infer_track(obs, hours=req.hours)
    analog = catalog.analog_forecast(obs, hours=req.hours) if req.include_analogs else {"analogs": [], "forecast": []}
    return {
        "lstm": lstm,
        "analog": analog,
        "observations": obs,
    }


@app.get("/api/predict/{sid}")
def predict_storm(sid: str, hours: int = 72) -> dict:
    meta = catalog.get_storm(sid)
    if not meta:
        raise HTTPException(404, "Storm not found")
    track = catalog.storm_track(meta["sid"])
    if len(track) < 4:
        raise HTTPException(400, "Not enough track points")
    # Use the first 8 points so the remainder is a verification overlay
    hist = track[:8] if len(track) > 20 else track[: max(4, len(track) // 2)]
    obs = [{"time": p["time"], "lat": p["lat"], "lon": p["lon"], "wind": p["wind_kt"], "pres": p["pres_hpa"]} for p in hist]
    lstm = infer_track(obs, hours=hours)
    analog = catalog.analog_forecast(obs, hours=hours)
    return {
        "storm": meta,
        "history": hist,
        "observed_future": track[len(hist) :],
        "lstm": lstm,
        "analog": analog,
    }


@app.get("/api/sample-image")
def sample_image(kind: str = "random") -> dict:
    rng = np.random.default_rng()
    img, lab = make_sample()
    pil = to_pil(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return {"image": b64, "label": lab, "kind": kind, "seed": int(rng.integers(0, 1_000_000))}
