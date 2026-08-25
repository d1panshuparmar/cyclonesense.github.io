"""Storm catalog and analog-track lookup over processed IBTrACS."""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from ml.categories import TRACK_PATTERNS, wind_to_imd
from app.config import DATA_PROCESSED

STORMS_PATH = DATA_PROCESSED / "storms.json"
TRACKS_PATH = DATA_PROCESSED / "tracks_6h.csv"


@lru_cache(maxsize=1)
def storms() -> list[dict]:
    if not STORMS_PATH.exists():
        return []
    return [row for row in json.loads(STORMS_PATH.read_text(encoding="utf-8")) if (row.get("season") or 0) >= 1980]


@lru_cache(maxsize=1)
def tracks() -> pd.DataFrame:
    if not TRACKS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(TRACKS_PATH, parse_dates=["ISO_TIME"])
    return df


def list_storms(
    season: int | None = None,
    basin: str | None = None,
    category: str | None = None,
    q: str | None = None,
    limit: int = 80,
) -> list[dict]:
    rows = storms()
    if season:
        rows = [r for r in rows if r.get("season") == season]
    if basin:
        rows = [r for r in rows if r.get("basin") == basin.upper()]
    if category:
        rows = [r for r in rows if r.get("category") == category.upper()]
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in str(r.get("name", "")).lower() or ql in str(r.get("sid", "")).lower()]
    rows = sorted(rows, key=lambda r: (r.get("season") or 0, r.get("peak_wind_kt") or 0), reverse=True)
    return rows[:limit]


def get_storm(sid: str) -> dict | None:
    sid_u = sid.upper()
    for r in storms():
        if str(r["sid"]).upper() == sid_u or str(r["name"]).upper() == sid_u:
            return r
    return None


def storm_track(sid: str) -> list[dict]:
    df = tracks()
    if df.empty:
        return []
    g = df[df["SID"].astype(str).str.upper() == sid.upper()].sort_values("ISO_TIME")
    if g.empty:
        # try by name
        meta = get_storm(sid)
        if meta:
            g = df[df["SID"].astype(str) == meta["sid"]].sort_values("ISO_TIME")
    out = []
    for _, row in g.iterrows():
        wind = float(row["WIND"]) if pd.notna(row["WIND"]) else None
        imd = wind_to_imd(wind)
        out.append(
            {
                "time": pd.Timestamp(row["ISO_TIME"]).isoformat(),
                "lat": round(float(row["LAT"]), 3),
                "lon": round(float(row["LON"]), 3),
                "wind_kt": round(wind, 1) if wind is not None else None,
                "pres_hpa": round(float(row["PRES"]), 1) if pd.notna(row["PRES"]) else None,
                "category": imd["code"],
                "color": imd["color"],
            }
        )
    return out


def stats() -> dict:
    rows = storms()
    if not rows:
        return {"n_storms": 0, "ready": False}
    seasons = [r["season"] for r in rows if r.get("season")]
    by_cat: dict[str, int] = {}
    by_basin: dict[str, int] = {}
    by_pattern: dict[str, int] = {}
    by_season: dict[str, int] = {}
    for r in rows:
        by_cat[r.get("category") or "UNK"] = by_cat.get(r.get("category") or "UNK", 0) + 1
        by_basin[r.get("basin") or "UNK"] = by_basin.get(r.get("basin") or "UNK", 0) + 1
        by_pattern[r.get("track_pattern") or "UNK"] = by_pattern.get(r.get("track_pattern") or "UNK", 0) + 1
        s = str(r.get("season") or "unk")
        by_season[s] = by_season.get(s, 0) + 1
    strongest = max(rows, key=lambda r: r.get("peak_wind_kt") or 0)
    return {
        "ready": True,
        "n_storms": len(rows),
        "season_min": min(seasons) if seasons else None,
        "season_max": max(seasons) if seasons else None,
        "by_category": by_cat,
        "by_basin": by_basin,
        "by_track_pattern": by_pattern,
        "by_season": [{"season": int(k), "n": v} for k, v in sorted(by_season.items()) if k.isdigit()],
        "strongest": strongest,
        "track_patterns": TRACK_PATTERNS,
    }


@lru_cache(maxsize=1)
def _analog_index() -> dict:
    df = tracks()
    feats, keys = [], []
    for sid, g in df.groupby("SID"):
        g = g.sort_values("ISO_TIME")
        if len(g) < 20:
            continue
        lats = g["LAT"].to_numpy()
        lons = g["LON"].to_numpy()
        winds = g["WIND"].fillna(30).to_numpy()
        for i in range(0, len(g) - 20, 3):
            feats.append(
                [
                    lats[i + 7],
                    lons[i + 7],
                    winds[i + 7],
                    lats[i + 7] - lats[i],
                    lons[i + 7] - lons[i],
                ]
            )
            keys.append((str(sid), int(i + 7)))
    if not feats:
        return {"feat": np.zeros((0, 5)), "keys": []}
    return {"feat": np.asarray(feats, dtype=np.float32), "keys": keys}


def analog_forecast(obs: list[dict], k: int = 8, hours: int = 72) -> dict:
    """Nearest-neighbour analog tracks from IBTrACS (interpretable baseline)."""
    df = tracks()
    if df.empty or len(obs) < 3:
        return {"analogs": [], "forecast": []}
    last = obs[-1]
    lat, lon, wind = float(last["lat"]), float(last["lon"]), float(last.get("wind") or 30)
    dlat = lat - float(obs[0]["lat"])
    dlon = lon - float(obs[0]["lon"])
    n_out = hours // 6
    index = _analog_index()
    if len(index["keys"]) == 0:
        return {"analogs": [], "forecast": []}
    q = np.array([lat, lon, wind, dlat, dlon], dtype=np.float32)
    scale = np.array([1.0, 1.0, 0.28, 2.0, 2.0], dtype=np.float32)
    dist = np.sqrt((((index["feat"] - q) * scale) ** 2).sum(axis=1))
    order = np.argsort(dist)[: max(k * 4, 16)]
    picked = [(float(dist[i]), index["keys"][i][0], index["keys"][i][1]) for i in order]
    seen: set[str] = set()
    analogs = []
    futures = []
    for dist, sid, idx in picked:
        if sid in seen:
            continue
        seen.add(sid)
        meta = get_storm(sid) or {"name": sid, "sid": sid}
        g = df[df["SID"].astype(str) == sid].sort_values("ISO_TIME")
        future = g.iloc[idx + 1 : idx + 1 + n_out]
        if len(future) < n_out:
            continue
        if len(analogs) >= k:
            break
        analogs.append(
            {
                "sid": sid,
                "name": meta.get("name"),
                "season": meta.get("season"),
                "distance": round(float(dist), 3),
                "peak_wind_kt": meta.get("peak_wind_kt"),
            }
        )
        futures.append(
            {
                "dlat": future["LAT"].to_numpy() - float(g.iloc[idx]["LAT"]),
                "dlon": future["LON"].to_numpy() - float(g.iloc[idx]["LON"]),
                "wind": future["WIND"].fillna(wind).to_numpy(),
            }
        )
    if not futures:
        return {"analogs": analogs, "forecast": []}
    dlat = np.mean([f["dlat"][:n_out] for f in futures], axis=0)
    dlon = np.mean([f["dlon"][:n_out] for f in futures], axis=0)
    wind_f = np.mean([f["wind"][:n_out] for f in futures], axis=0)
    t0 = pd.to_datetime(last.get("time"), utc=True, errors="coerce")
    if pd.isna(t0):
        t0 = pd.Timestamp.now(tz="UTC")
    forecast = []
    for i in range(len(dlat)):
        w = float(wind_f[i])
        imd = wind_to_imd(w)
        forecast.append(
            {
                "time": (t0 + pd.Timedelta(hours=6 * (i + 1))).isoformat(),
                "lead_hours": 6 * (i + 1),
                "lat": round(lat + float(dlat[i]), 3),
                "lon": round(lon + float(dlon[i]), 3),
                "wind_kt": round(w, 1),
                "category": imd["code"],
                "category_name": imd["name"],
                "color": imd["color"],
                "source": "analog",
            }
        )
    return {"analogs": analogs, "forecast": forecast}


def recent_showcase(n: int = 12) -> list[dict]:
    rows = sorted(storms(), key=lambda r: r.get("season") or 0, reverse=True)
    # prefer named intense storms
    named = [r for r in rows if r.get("name") and str(r["name"]).upper() not in ("UNNAMED", "NOT_NAMED", "NAMELESS")]
    pool = named or rows
    intense = [r for r in pool if (r.get("peak_wind_kt") or 0) >= 48]
    chosen = (intense or pool)[:n]
    return chosen
