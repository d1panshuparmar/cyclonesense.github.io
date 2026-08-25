"""Clean IBTrACS North Indian Ocean tracks into 6-hourly sequences."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .categories import basin_from_lon, classify_track_pattern, wind_to_imd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ibtracs.NI.list.v04r01.csv"
PROCESSED = ROOT / "data" / "processed"


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_raw(path: Path = RAW) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://www.ncei.noaa.gov/data/"
            "international-best-track-archive-for-climate-stewardship-ibtracs/"
            "v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
        )
        print(f"Downloading IBTrACS NI from {url}")
        import urllib.request

        urllib.request.urlretrieve(url, path)
    df = pd.read_csv(path, skiprows=[1], low_memory=False)
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], utc=True, errors="coerce")
    df["LAT"] = _to_num(df["LAT"])
    df["LON"] = _to_num(df["LON"])
    wind_cols = [c for c in ("NEWDELHI_WIND", "WMO_WIND", "USA_WIND") if c in df.columns]
    pres_cols = [c for c in ("NEWDELHI_PRES", "WMO_PRES", "USA_PRES") if c in df.columns]
    df["WIND"] = None
    for c in wind_cols:
        df["WIND"] = df["WIND"].fillna(_to_num(df[c]))
    df["PRES"] = None
    for c in pres_cols:
        df["PRES"] = df["PRES"].fillna(_to_num(df[c]))
    df["WIND"] = _to_num(df["WIND"])
    df["PRES"] = _to_num(df["PRES"])
    df["SEASON"] = _to_num(df["SEASON"]).astype("Int64")
    df = df.dropna(subset=["SID", "ISO_TIME", "LAT", "LON"])
    df = df[(df["LAT"].between(-5, 35)) & (df["LON"].between(40, 110))]
    return df


def interpolate_storm(g: pd.DataFrame) -> pd.DataFrame | None:
    g = g.sort_values("ISO_TIME").drop_duplicates("ISO_TIME")
    if len(g) < 4:
        return None
    g = g.set_index("ISO_TIME")
    t0, t1 = g.index.min(), g.index.max()
    try:
        start, end = t0.floor("6h"), t1.ceil("6h")
        idx = pd.date_range(start, end, freq="6h", tz="UTC")
    except ValueError:
        start, end = t0.floor("6H"), t1.ceil("6H")
        idx = pd.date_range(start, end, freq="6H", tz="UTC")
    if len(idx) < 6:
        return None
    num = g[["LAT", "LON", "WIND", "PRES"]].copy()
    num = num.reindex(num.index.union(idx)).sort_index()
    num[["LAT", "LON"]] = num[["LAT", "LON"]].interpolate(method="time", limit=4)
    num["WIND"] = num["WIND"].interpolate(method="time", limit=4)
    num["PRES"] = num["PRES"].interpolate(method="time", limit=4)
    num = num.reindex(idx)
    num = num.dropna(subset=["LAT", "LON"])
    if len(num) < 6:
        return None
    num["WIND"] = num["WIND"].ffill().bfill().fillna(25)
    num["PRES"] = num["PRES"].ffill().bfill().fillna(1000)
    num["SID"] = g["SID"].iloc[0]
    num["NAME"] = g["NAME"].iloc[0] if "NAME" in g.columns else "UNNAMED"
    num["SEASON"] = int(g["SEASON"].iloc[0]) if pd.notna(g["SEASON"].iloc[0]) else None
    num.index.name = "ISO_TIME"
    return num.reset_index()


def storm_summary(points: pd.DataFrame) -> dict:
    winds = points["WIND"].dropna()
    peak = float(winds.max()) if len(winds) else 0.0
    cat = wind_to_imd(peak)
    lats = points["LAT"].tolist()
    lons = points["LON"].tolist()
    return {
        "sid": str(points["SID"].iloc[0]),
        "name": str(points["NAME"].iloc[0]).title() if pd.notna(points["NAME"].iloc[0]) else "Unnamed",
        "season": int(points["SEASON"].iloc[0]) if pd.notna(points["SEASON"].iloc[0]) else None,
        "start": points["ISO_TIME"].iloc[0].isoformat(),
        "end": points["ISO_TIME"].iloc[-1].isoformat(),
        "peak_wind_kt": round(peak, 1),
        "min_pressure_hpa": round(float(points["PRES"].min()), 1) if points["PRES"].notna().any() else None,
        "category": cat["code"],
        "category_name": cat["name"],
        "basin": basin_from_lon(float(np.median(lons))),
        "track_pattern": classify_track_pattern(lats, lons),
        "n_points": int(len(points)),
        "genesis_lat": round(lats[0], 2),
        "genesis_lon": round(lons[0], 2),
    }


def build_sequences(
    tracks: pd.DataFrame,
    in_steps: int = 8,
    out_steps: int = 12,
) -> dict[str, np.ndarray]:
    """Windowed sequences for the LSTM.

    Features per step: lat_n, lon_n, wind_n, pres_n, dlat, dlon, dwind,
    month_sin, month_cos, basin
    Targets per future step: dlat, dlon, dwind
    """
    xs, ys, meta = [], [], []
    for sid, g in tracks.groupby("SID"):
        g = g.sort_values("ISO_TIME")
        if len(g) < in_steps + out_steps:
            continue
        lat = g["LAT"].to_numpy(dtype=np.float32)
        lon = g["LON"].to_numpy(dtype=np.float32)
        wind = g["WIND"].fillna(25).to_numpy(dtype=np.float32)
        pres = g["PRES"].fillna(1000).to_numpy(dtype=np.float32)
        month = g["ISO_TIME"].dt.month.to_numpy(dtype=np.float32)
        basin = np.array([0.0 if basin_from_lon(float(x)) == "AS" else 1.0 for x in lon], dtype=np.float32)
        dlat = np.diff(lat, prepend=lat[0])
        dlon = np.diff(lon, prepend=lon[0])
        dwind = np.diff(wind, prepend=wind[0])
        lat_n = lat / 40.0
        lon_n = (lon - 40.0) / 80.0
        wind_n = wind / 150.0
        pres_n = (1020.0 - pres) / 100.0
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        feats = np.stack(
            [lat_n, lon_n, wind_n, pres_n, dlat, dlon, dwind, month_sin, month_cos, basin],
            axis=1,
        )
        for i in range(0, len(g) - in_steps - out_steps + 1, 2):
            x = feats[i : i + in_steps]
            future_dlat = dlat[i + in_steps : i + in_steps + out_steps]
            future_dlon = dlon[i + in_steps : i + in_steps + out_steps]
            future_dwind = dwind[i + in_steps : i + in_steps + out_steps]
            y = np.stack([future_dlat, future_dlon, future_dwind], axis=1)
            xs.append(x)
            ys.append(y)
            meta.append(
                {
                    "sid": str(sid),
                    "season": int(g["SEASON"].iloc[i]) if pd.notna(g["SEASON"].iloc[i]) else 0,
                    "i": i,
                }
            )
    return {
        "x": np.stack(xs).astype(np.float32),
        "y": np.stack(ys).astype(np.float32),
        "meta": meta,
    }


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print(f"Reading {RAW}")
    raw = load_raw()
    print(f"Raw points: {len(raw):,}  storms: {raw['SID'].nunique()}")
    frames = []
    summaries = []
    for sid, g in raw.groupby("SID"):
        interp = interpolate_storm(g)
        if interp is None:
            continue
        frames.append(interp)
        summaries.append(storm_summary(interp))
    tracks = pd.concat(frames, ignore_index=True)
    tracks.to_csv(PROCESSED / "tracks_6h.csv", index=False)
    (PROCESSED / "storms.json").write_text(json.dumps(summaries, indent=2))
    print(f"Interpolated storms: {len(summaries)}  points: {len(tracks):,}")

    seq = build_sequences(tracks)
    seasons = np.array([m["season"] for m in seq["meta"]])
    split = np.where(seasons < 2018, "train", np.where(seasons < 2022, "val", "test"))
    np.savez_compressed(
        PROCESSED / "track_sequences.npz",
        x=seq["x"],
        y=seq["y"],
        season=seasons,
        split=split,
    )
    print(f"Sequences: {len(seq['x']):,}  x={seq['x'].shape} y={seq['y'].shape}")
    for name in ("train", "val", "test"):
        print(f"  {name}: {(split == name).sum()}")


if __name__ == "__main__":
    main()
