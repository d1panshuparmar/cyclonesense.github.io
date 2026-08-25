"""Inference helpers for the image CNN and track LSTM."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .categories import CLOUD_PATTERNS, IMD_CATEGORIES, PATTERN_CODES, wind_to_imd
from .models import CycloneImageNet, TrackLSTM

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data" / "models"

_image_model: CycloneImageNet | None = None
_track_model: TrackLSTM | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_model() -> CycloneImageNet:
    global _image_model
    if _image_model is None:
        model = CycloneImageNet()
        path = MODELS / "image_cnn.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=_device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
        model.to(_device).eval()
        _image_model = model
    return _image_model


def load_track_model() -> TrackLSTM:
    global _track_model
    if _track_model is None:
        model = TrackLSTM()
        path = MODELS / "track_lstm.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=_device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
        model.to(_device).eval()
        _track_model = model
    return _track_model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    gray = image.convert("L").resize((128, 128), Image.BILINEAR)
    arr = np.array(gray, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def infer_image(image: Image.Image) -> dict:
    model = load_image_model()
    x = preprocess_image(image).to(_device)
    out = model(x)
    presence_p = torch.softmax(out["presence"], dim=-1)[0].cpu().numpy()
    pattern_p = torch.softmax(out["pattern"], dim=-1)[0].cpu().numpy()
    cat_p = torch.softmax(out["category"], dim=-1)[0].cpu().numpy()
    wind = float(out["wind"][0].cpu())
    wind = float(np.clip(wind, 0, 180))
    is_cyclone = bool(presence_p[1] >= 0.45)
    pi = int(pattern_p.argmax())
    ci = int(cat_p.argmax())
    imd = IMD_CATEGORIES[ci] if is_cyclone else IMD_CATEGORIES[0]
    wind_imd = wind_to_imd(wind)
    if is_cyclone:
        imd = wind_imd
    return {
        "is_cyclone": is_cyclone,
        "cyclone_confidence": round(float(presence_p[1]), 4),
        "pattern": {
            "code": PATTERN_CODES[pi],
            "name": CLOUD_PATTERNS[pi]["name"],
            "dvorak": CLOUD_PATTERNS[pi]["dvorak"],
            "description": CLOUD_PATTERNS[pi]["description"],
            "confidence": round(float(pattern_p[pi]), 4),
            "distribution": [
                {"code": CLOUD_PATTERNS[i]["code"], "name": CLOUD_PATTERNS[i]["name"], "p": round(float(pattern_p[i]), 4)}
                for i in range(len(CLOUD_PATTERNS))
            ],
        },
        "category": {
            "code": imd["code"],
            "name": imd["name"],
            "color": imd["color"],
            "min_kt": imd["min_kt"],
            "max_kt": imd["max_kt"],
            "confidence": round(float(cat_p[ci]), 4),
        },
        "wind_kt": round(wind, 1),
        "category_from_wind": wind_imd,
        "notes": "CNN trained on physically motivated IR pattern synthesis; retrain on INSAT-3D TIR for operational use.",
    }


def encode_obs(obs: list[dict]) -> torch.Tensor:
    """obs: dicts with lat, lon, wind, pres?, time ISO."""
    import pandas as pd

    rows = []
    prev = None
    for o in obs:
        lat = float(o["lat"])
        lon = float(o["lon"])
        wind = float(o.get("wind") or 30)
        pres = float(o.get("pres") or 1000)
        ts = pd.to_datetime(o.get("time"), utc=True, errors="coerce")
        month = float(ts.month) if not pd.isna(ts) else 10.0
        dlat = 0.0 if prev is None else lat - prev[0]
        dlon = 0.0 if prev is None else lon - prev[1]
        dwind = 0.0 if prev is None else wind - prev[2]
        basin = 0.0 if lon < 77.5 else 1.0
        rows.append(
            [
                lat / 40.0,
                (lon - 40.0) / 80.0,
                wind / 150.0,
                (1020.0 - pres) / 100.0,
                dlat,
                dlon,
                dwind,
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12),
                basin,
            ]
        )
        prev = (lat, lon, wind)
    x = np.array(rows, dtype=np.float32)
    if len(x) < 8:
        pad = np.repeat(x[:1], 8 - len(x), axis=0)
        x = np.concatenate([pad, x], axis=0)
    else:
        x = x[-8:]
    return torch.from_numpy(x).unsqueeze(0)


@torch.no_grad()
def infer_track(obs: list[dict], hours: int = 72) -> dict:
    model = load_track_model()
    x = encode_obs(obs).to(_device)
    deltas = model(x)[0].cpu().numpy()  # (12, 3) 6-hourly
    last = obs[-1]
    lat, lon, wind = float(last["lat"]), float(last["lon"]), float(last.get("wind") or 30)
    import pandas as pd

    t0 = pd.to_datetime(last.get("time"), utc=True, errors="coerce")
    if pd.isna(t0):
        t0 = pd.Timestamp.now(tz="UTC")
    n = min(12, max(1, hours // 6))
    points = []
    for i in range(n):
        lat += float(deltas[i, 0])
        lon += float(deltas[i, 1])
        wind = float(np.clip(wind + deltas[i, 2], 0, 180))
        t = t0 + pd.Timedelta(hours=6 * (i + 1))
        imd = wind_to_imd(wind)
        # Simple cone: 50 / 80 / 120 km-ish growth with lead time (NIO operational scale)
        radius_km = 40 + 18 * (i + 1)
        points.append(
            {
                "time": t.isoformat(),
                "lead_hours": 6 * (i + 1),
                "lat": round(lat, 3),
                "lon": round(lon, 3),
                "wind_kt": round(wind, 1),
                "category": imd["code"],
                "category_name": imd["name"],
                "color": imd["color"],
                "uncertainty_km": radius_km,
            }
        )
    peak = max(points, key=lambda p: p["wind_kt"])
    return {
        "horizon_hours": 6 * n,
        "forecast": points,
        "peak_forecast_wind_kt": peak["wind_kt"],
        "peak_forecast_category": peak["category"],
        "land_threat": _land_threat(points),
    }


def _land_threat(points: list[dict]) -> dict:
    """Very coarse NIO coastline box check for demo alerting."""
    india = dict(lat=(6, 25), lon=(68, 92))
    hits = [p for p in points if india["lat"][0] <= p["lat"] <= india["lat"][1] and india["lon"][0] <= p["lon"] <= india["lon"][1]]
    if not hits:
        return {"level": "WATCH_OCEAN", "message": "Forecast track remains over the open North Indian Ocean."}
    first = hits[0]
    if first["lead_hours"] <= 24:
        level = "RED"
    elif first["lead_hours"] <= 48:
        level = "ORANGE"
    else:
        level = "YELLOW"
    return {
        "level": level,
        "message": f"Track approaches the Indian subcontinent around H+{first['lead_hours']} near {first['lat']:.1f}N, {first['lon']:.1f}E.",
        "eta_hours": first["lead_hours"],
    }


def gradcam(image: Image.Image) -> np.ndarray:
    """Grad-CAM on the last conv layer, targeting the cyclone-presence class."""
    model = load_image_model()
    model.eval()
    activations: dict[str, torch.Tensor] = {}
    handle = model.b3.net[3].register_forward_hook(lambda _m, _i, o: activations.update(v=o))
    x = preprocess_image(image).to(_device)
    x.requires_grad_(True)
    try:
        out = model(x)
        score = out["presence"][0, 1]
        act = activations["v"]
        grad = torch.autograd.grad(score, act, retain_graph=False)[0]
    finally:
        handle.remove()
    weights = grad[0].mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * act[0]).sum(0))
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    cam = torch.nn.functional.interpolate(
        cam.detach()[None, None], size=(128, 128), mode="bilinear", align_corners=False
    )[0, 0]
    return cam.cpu().numpy()
