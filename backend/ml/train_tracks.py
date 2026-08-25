"""Train the track/intensity LSTM on IBTrACS 6-hourly sequences."""

from __future__ import annotations

import json
from pathlib import Path

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import TrackLSTM

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "track_sequences.npz"
OUT = ROOT / "data" / "models"
IN_FEATURES = 10
OUT_STEPS = 12


def metrics(pred: np.ndarray, y: np.ndarray, x: np.ndarray) -> dict:
    """pred, y: (N, T, 3) dlat/dlon/dwind; reconstruct positions from last input lat/lon/wind."""
    lat0 = x[:, -1, 0] * 40.0
    lon0 = x[:, -1, 1] * 80.0 + 40.0
    wind0 = x[:, -1, 2] * 150.0
    pred_lat = lat0[:, None] + np.cumsum(pred[:, :, 0], axis=1)
    pred_lon = lon0[:, None] + np.cumsum(pred[:, :, 1], axis=1)
    pred_wind = wind0[:, None] + np.cumsum(pred[:, :, 2], axis=1)
    true_lat = lat0[:, None] + np.cumsum(y[:, :, 0], axis=1)
    true_lon = lon0[:, None] + np.cumsum(y[:, :, 1], axis=1)
    true_wind = wind0[:, None] + np.cumsum(y[:, :, 2], axis=1)
    # Equirectangular km error
    mid_lat = np.radians((pred_lat + true_lat) / 2)
    dkm = np.sqrt(((pred_lat - true_lat) * 111.0) ** 2 + ((pred_lon - true_lon) * 111.0 * np.cos(mid_lat)) ** 2)
    wind_mae = np.mean(np.abs(pred_wind - true_wind), axis=0)
    out = {}
    for hours, idx in ((24, 3), (48, 7), (72, 11)):
        if idx < dkm.shape[1]:
            out[f"track_mae_km_{hours}h"] = float(np.mean(dkm[:, idx]))
            out[f"wind_mae_kt_{hours}h"] = float(wind_mae[idx])
    out["track_mae_km_mean"] = float(np.mean(dkm))
    out["wind_mae_kt_mean"] = float(np.mean(np.abs(pred_wind - true_wind)))
    return out


def run_epoch(model, loader, opt, device, train: bool) -> float:
    model.train(train)
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train:
            opt.zero_grad()
        pred = model(xb)
        loss = torch.nn.functional.smooth_l1_loss(pred, yb)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(n, 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    blob = np.load(DATA, allow_pickle=True)
    x, y, split = blob["x"], blob["y"], blob["split"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  n={len(x)}")

    def loader(name: str, shuffle: bool, bs: int = 64) -> DataLoader:
        m = split == name
        ds = TensorDataset(torch.from_numpy(x[m]), torch.from_numpy(y[m]))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

    model = TrackLSTM(in_features=IN_FEATURES, hidden=96, layers=2, out_steps=OUT_STEPS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    train_loader, val_loader = loader("train", True), loader("val", False)
    best = 1e9
    history = []
    for epoch in range(1, 41):
        tr = run_epoch(model, train_loader, opt, device, True)
        va = run_epoch(model, val_loader, opt, device, False)
        sched.step()
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(f"epoch {epoch:02d}  train {tr:.4f}  val {va:.4f}")
        if va < best:
            best = va
            torch.save({"state_dict": model.state_dict(), "val": va, "epoch": epoch}, OUT / "track_lstm.pt")

    ckpt = torch.load(OUT / "track_lstm.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    report = {"best_val": ckpt["val"], "epoch": ckpt["epoch"], "history": history}
    for name in ("val", "test"):
        m = split == name
        if m.sum() == 0:
            continue
        with torch.no_grad():
            pred = model(torch.from_numpy(x[m]).to(device)).cpu().numpy()
        report[name] = metrics(pred, y[m], x[m])
        print(name, report[name])
    (OUT / "track_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"saved {OUT / 'track_lstm.pt'}")


if __name__ == "__main__":
    main()
