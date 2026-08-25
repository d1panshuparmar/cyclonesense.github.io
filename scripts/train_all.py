"""Prepare IBTrACS sequences and train both models."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print("\n>>", " ".join(args), flush=True)
    subprocess.check_call(args, cwd=ROOT)


def main() -> None:
    py = sys.executable
    env_prefix = [py]
    run(env_prefix + ["-m", "backend.ml.prepare_ibtracs"])
    run(env_prefix + [str(ROOT / "backend" / "ml" / "train_tracks.py")])
    run(env_prefix + [str(ROOT / "backend" / "ml" / "train_images.py")])
    print("\nTraining complete. Models in data/models/")


if __name__ == "__main__":
    main()
