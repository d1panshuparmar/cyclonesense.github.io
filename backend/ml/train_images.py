"""Train the multi-task IR CNN on synthetic (or folder) imagery."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_synth import make_sample
from models import CycloneImageNet, image_loss

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "models"


class SynthDataset(Dataset):
    def __init__(self, n: int, seed: int) -> None:
        self.n = n
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = random.Random(self.seed + idx)
        img, lab = make_sample(rng)
        x = torch.from_numpy(img).unsqueeze(0)
        return {
            "image": x,
            "presence": torch.tensor(lab["presence"], dtype=torch.long),
            "pattern": torch.tensor(lab["pattern"], dtype=torch.long),
            "category": torch.tensor(lab["category"], dtype=torch.long),
            "wind": torch.tensor(lab["wind"], dtype=torch.float32),
        }


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "presence": torch.stack([b["presence"] for b in batch]),
        "pattern": torch.stack([b["pattern"] for b in batch]),
        "category": torch.stack([b["category"] for b in batch]),
        "wind": torch.stack([b["wind"] for b in batch]),
    }


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    n = 0
    correct = {"presence": 0, "pattern": 0, "category": 0}
    wind_err = 0.0
    for batch in loader:
        img = batch["image"].to(device)
        pred = model(img)
        n += len(img)
        for key in ("presence", "pattern", "category"):
            correct[key] += (pred[key].argmax(-1).cpu() == batch[key]).sum().item()
        wind_err += (pred["wind"].cpu() - batch["wind"]).abs().sum().item()
    return {
        "presence_acc": correct["presence"] / n,
        "pattern_acc": correct["pattern"] / n,
        "category_acc": correct["category"] / n,
        "wind_mae_kt": wind_err / n,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    train_ds, val_ds = SynthDataset(1800, 0), SynthDataset(400, 50_000)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate, num_workers=0)
    model = CycloneImageNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best = 0.0
    history = []
    for epoch in range(1, 9):
        model.train()
        losses = []
        for batch in train_loader:
            img = batch["image"].to(device)
            targets = {k: batch[k].to(device) for k in ("presence", "pattern", "category", "wind")}
            opt.zero_grad()
            pred = model(img)
            loss = image_loss(pred, targets)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        stats = evaluate(model, val_loader, device)
        stats["train_loss"] = float(np.mean(losses))
        history.append({"epoch": epoch, **stats})
        score = 0.4 * stats["presence_acc"] + 0.35 * stats["pattern_acc"] + 0.25 * stats["category_acc"]
        print(
            f"epoch {epoch:02d}  loss {stats['train_loss']:.3f}  "
            f"presence {stats['presence_acc']:.3f}  pattern {stats['pattern_acc']:.3f}  "
            f"imd {stats['category_acc']:.3f}  wind MAE {stats['wind_mae_kt']:.1f} kt",
            flush=True,
        )
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "stats": stats, "epoch": epoch}, OUT / "image_cnn.pt")
    report = {"best_score": best, "history": history}
    (OUT / "image_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"saved {OUT / 'image_cnn.pt'}")


if __name__ == "__main__":
    main()
