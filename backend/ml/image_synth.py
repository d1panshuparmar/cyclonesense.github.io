"""Generate IR-like synthetic cyclone patterns for CNN pretraining.

Real INSAT-3D/3DR TIR imagery (MOSDAC) can replace this generator via
`train_images.py --data-dir`. The synthetic set is physically motivated:
cold cloud = bright in inverted IR, warm ocean/eye = dark.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
from PIL import Image

try:
    from .categories import CLOUD_PATTERNS, IMD_CATEGORIES, wind_to_imd
except ImportError:
    from categories import CLOUD_PATTERNS, IMD_CATEGORIES, wind_to_imd


SIZE = 128


def _noise(h: int, w: int, scale: int = 8) -> np.ndarray:
    sh = max(2, (h + scale - 1) // scale)
    sw = max(2, (w + scale - 1) // scale)
    small = np.random.randn(sh, sw).astype(np.float32)
    tiled = np.repeat(np.repeat(small, scale, axis=0), scale, axis=1)
    return tiled[:h, :w]


def _spiral_field(yy: np.ndarray, xx: np.ndarray, turns: float, tightness: float) -> np.ndarray:
    r = np.sqrt(xx * xx + yy * yy) + 1e-6
    theta = np.arctan2(yy, xx)
    return np.sin(theta * turns + r * tightness)


def _radial(yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    return np.sqrt(xx * xx + yy * yy)


def render_pattern(
    pattern: str,
    wind_kt: float,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    rng = rng or random.Random()
    n = SIZE
    cy = n / 2 + rng.uniform(-8, 8)
    cx = n / 2 + rng.uniform(-8, 8)
    y = np.arange(n, dtype=np.float32)
    x = np.arange(n, dtype=np.float32)
    yy, xx = np.meshgrid(y - cy, x - cx, indexing="ij")
    r = _radial(yy, xx)
    field = 0.22 + 0.08 * _noise(n, n, 16)

    intensity = np.clip((wind_kt - 20) / 120.0, 0.05, 1.0)
    cold = 0.35 + 0.55 * intensity

    if pattern == "EYE":
        eye_r = 4 + 6 * (1 - intensity)
        wall = np.exp(-((r - (eye_r + 7)) ** 2) / (2 * (5 + 3 * intensity) ** 2))
        cdo = np.exp(-(r**2) / (2 * (28 + 10 * intensity) ** 2))
        spiral = 0.35 * np.clip(_spiral_field(yy, xx, 2.2, 0.18), 0, 1) * np.exp(-r / 55)
        eye = np.exp(-(r**2) / (2 * eye_r**2))
        field += cold * (0.7 * cdo + 0.9 * wall + spiral) - 0.55 * eye

    elif pattern == "CDO":
        blob = np.exp(-(r**2) / (2 * (22 + 12 * intensity) ** 2))
        field += cold * (blob + 0.2 * np.clip(_noise(n, n, 6), 0, None) * blob)

    elif pattern == "CURVED_BAND":
        spiral = np.clip(_spiral_field(yy, xx, 1.6, 0.22), 0, None)
        band = spiral * np.exp(-r / (38 + 10 * intensity))
        core = np.exp(-(r**2) / (2 * 16**2)) * 0.35 * intensity
        field += cold * (0.95 * band + core)

    elif pattern == "SHEAR":
        shift = 18 + rng.uniform(0, 12)
        ang = rng.uniform(0, 2 * math.pi)
        sx, sy = shift * math.cos(ang), shift * math.sin(ang)
        r2 = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
        cloud = np.exp(-(r2**2) / (2 * (20 + 8 * intensity) ** 2))
        llc = np.exp(-(r**2) / (2 * 6**2)) * 0.15
        field += cold * cloud + llc

    elif pattern == "EMBEDDED_CENTER":
        overcast = np.exp(-(r**2) / (2 * (26 + 8 * intensity) ** 2))
        swirl = 0.4 * np.clip(_spiral_field(yy, xx, 2.8, 0.28), 0, 1) * overcast
        field += cold * (overcast + swirl)

    else:
        field += 0.25 * np.abs(_noise(n, n, 5))

    field += 0.05 * _noise(n, n, 4)
    field = np.clip(field, 0.0, 1.0)
    return field.astype(np.float32)


def render_non_cyclone(rng: Optional[random.Random] = None) -> np.ndarray:
    rng = rng or random.Random()
    n = SIZE
    field = 0.18 + 0.12 * np.abs(_noise(n, n, 10))
    kind = rng.choice(["convection", "itcz", "clear", "front"])
    y = np.arange(n, dtype=np.float32)
    x = np.arange(n, dtype=np.float32)
    yy, xx = np.meshgrid(y - n / 2, x - n / 2, indexing="ij")
    if kind == "convection":
        for _ in range(rng.randint(4, 10)):
            cy, cx = rng.uniform(10, n - 10), rng.uniform(10, n - 10)
            r = np.sqrt((yy + n / 2 - cy) ** 2 + (xx + n / 2 - cx) ** 2)
            field += rng.uniform(0.15, 0.4) * np.exp(-(r**2) / (2 * rng.uniform(6, 14) ** 2))
    elif kind == "itcz":
        band = np.exp(-((yy - rng.uniform(-20, 20)) ** 2) / (2 * 12**2))
        field += 0.35 * band * (0.6 + 0.4 * np.abs(_noise(n, n, 7)))
    elif kind == "front":
        line = np.exp(-((yy - 0.4 * xx) ** 2) / (2 * 8**2))
        field += 0.4 * line
    field = np.clip(field + 0.04 * _noise(n, n, 3), 0, 1)
    return field.astype(np.float32)


def sample_label(rng: Optional[random.Random] = None) -> dict:
    rng = rng or random.Random()
    if rng.random() < 0.22:
        return {
            "presence": 0,
            "pattern": 0,
            "category": 0,
            "wind": float(rng.uniform(0, 16)),
            "is_cyclone": False,
        }
    pattern = rng.choice(CLOUD_PATTERNS)["code"]
    # Pattern-conditioned intensity (operational Dvorak-like prior)
    wind_ranges = {
        "CURVED_BAND": (20, 55),
        "SHEAR": (25, 60),
        "CDO": (40, 85),
        "EMBEDDED_CENTER": (50, 95),
        "EYE": (65, 140),
    }
    lo, hi = wind_ranges[pattern]
    wind = float(rng.uniform(lo, hi))
    cat = wind_to_imd(wind)
    return {
        "presence": 1,
        "pattern": [p["code"] for p in CLOUD_PATTERNS].index(pattern),
        "pattern_code": pattern,
        "category": [c["code"] for c in IMD_CATEGORIES].index(cat["code"]),
        "wind": wind,
        "is_cyclone": True,
    }


def make_sample(rng: Optional[random.Random] = None) -> tuple[np.ndarray, dict]:
    rng = rng or random.Random()
    label = sample_label(rng)
    if not label["is_cyclone"]:
        img = render_non_cyclone(rng)
    else:
        img = render_pattern(label["pattern_code"], label["wind"], rng)
    return img, label


def to_pil(img: np.ndarray) -> Image.Image:
    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")
