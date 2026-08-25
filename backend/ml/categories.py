"""IMD tropical cyclone taxonomy and Dvorak-style cloud patterns."""

from __future__ import annotations

from typing import Optional

# IMD (2015–present) maximum sustained surface wind (3-minute, knots)
IMD_CATEGORIES = [
    {"code": "LPA", "name": "Low Pressure Area", "min_kt": 0, "max_kt": 16, "color": "#94a3b8"},
    {"code": "D", "name": "Depression", "min_kt": 17, "max_kt": 27, "color": "#38bdf8"},
    {"code": "DD", "name": "Deep Depression", "min_kt": 28, "max_kt": 33, "color": "#22d3ee"},
    {"code": "CS", "name": "Cyclonic Storm", "min_kt": 34, "max_kt": 47, "color": "#facc15"},
    {"code": "SCS", "name": "Severe Cyclonic Storm", "min_kt": 48, "max_kt": 63, "color": "#fb923c"},
    {"code": "VSCS", "name": "Very Severe Cyclonic Storm", "min_kt": 64, "max_kt": 89, "color": "#f97316"},
    {"code": "ESCS", "name": "Extremely Severe Cyclonic Storm", "min_kt": 90, "max_kt": 119, "color": "#ef4444"},
    {"code": "SuCS", "name": "Super Cyclonic Storm", "min_kt": 120, "max_kt": 300, "color": "#a855f7"},
]

IMD_CODES = [c["code"] for c in IMD_CATEGORIES]
STORM_CODES = [c["code"] for c in IMD_CATEGORIES if c["code"] != "LPA"]

# Satellite cloud patterns used in operational Dvorak analysis
CLOUD_PATTERNS = [
    {
        "code": "CURVED_BAND",
        "name": "Curved Band",
        "dvorak": "T1.0–T3.0",
        "description": "Spiral convective band wrapping around a developing centre.",
    },
    {
        "code": "SHEAR",
        "name": "Shear Pattern",
        "dvorak": "T1.5–T3.5",
        "description": "Exposed low-level centre displaced from the dense cloud mass.",
    },
    {
        "code": "CDO",
        "name": "Central Dense Overcast",
        "dvorak": "T2.5–T4.5",
        "description": "Uniform cold cloud shield hiding the centre; typical of intensifying storms.",
    },
    {
        "code": "EMBEDDED_CENTER",
        "name": "Embedded Centre",
        "dvorak": "T3.5–T4.5",
        "description": "Centre embedded in cold overcast before a clear eye forms.",
    },
    {
        "code": "EYE",
        "name": "Eye Pattern",
        "dvorak": "T4.0–T7.0",
        "description": "Warm eye surrounded by a cold eyewall; mature intense cyclone.",
    },
]

PATTERN_CODES = [p["code"] for p in CLOUD_PATTERNS]

TRACK_PATTERNS = [
    {"code": "WESTWARD", "name": "Westward / Straight"},
    {"code": "NORTHWEST", "name": "Northwestward"},
    {"code": "RECURVING", "name": "Recurving (NE)"},
    {"code": "LOOPING", "name": "Looping / Erratic"},
    {"code": "NORTHWARD", "name": "Northward"},
]


def wind_to_imd(wind_kt: Optional[float]) -> dict:
    if wind_kt is None or wind_kt != wind_kt:  # NaN
        return IMD_CATEGORIES[0]
    for cat in reversed(IMD_CATEGORIES):
        if wind_kt >= cat["min_kt"]:
            return cat
    return IMD_CATEGORIES[0]


def imd_index(code: str) -> int:
    try:
        return IMD_CODES.index(code)
    except ValueError:
        return 0


def pattern_index(code: str) -> int:
    try:
        return PATTERN_CODES.index(code)
    except ValueError:
        return 0


def basin_from_lon(lon: float) -> str:
    """Arabian Sea vs Bay of Bengal split near the Indian west/east coasts."""
    return "AS" if lon < 77.5 else "BOB"


def classify_track_pattern(lats: list[float], lons: list[float]) -> str:
    if len(lats) < 4:
        return "NORTHWEST"
    dlat = lats[-1] - lats[0]
    dlon = lons[-1] - lons[0]
    # Recurvature: early westward/northward then later eastward
    mid = len(lons) // 2
    early_dlon = lons[mid] - lons[0]
    late_dlon = lons[-1] - lons[mid]
    headings = []
    for i in range(1, len(lons)):
        headings.append((lons[i] - lons[i - 1], lats[i] - lats[i - 1]))
    sign_flips = 0
    for i in range(1, len(headings)):
        if headings[i][0] * headings[i - 1][0] < 0:
            sign_flips += 1
    if sign_flips >= 3:
        return "LOOPING"
    if early_dlon < -0.5 and late_dlon > 0.8:
        return "RECURVING"
    if abs(dlon) < 1.2 and dlat > abs(dlon):
        return "NORTHWARD"
    if dlon < -1.0 and abs(dlat) < abs(dlon):
        return "WESTWARD"
    return "NORTHWEST"
