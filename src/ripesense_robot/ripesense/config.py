"""Load config.yaml once; paths resolve relative to the project root."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load(path: str | None = None) -> dict:
    p = Path(path or os.environ.get("RIPESENSE_CONFIG", ROOT / "config.yaml"))
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("RIPESENSE_MOCK", "").lower() in ("1", "true", "yes"):
        cfg["mock"] = True
    if os.environ.get("RIPESENSE_CAMERA_ONLY", "").lower() in ("1", "true", "yes"):
        cfg["camera_only"] = True
        cfg["camera"]["backend"] = "webcam"
    if os.environ.get("RIPESENSE_WEBCAM"):
        cfg["camera"]["webcam_index"] = int(os.environ["RIPESENSE_WEBCAM"])
    if os.environ.get("RIPESENSE_DATA_DIR"):
        cfg["data_dir"] = os.environ["RIPESENSE_DATA_DIR"]
    if os.environ.get("RIPESENSE_MODELS_DIR"):
        cfg["models_dir"] = os.environ["RIPESENSE_MODELS_DIR"]
    return cfg


def path(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def data_dir() -> Path:
    d = path(load()["data_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    d = path(load()["models_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d
