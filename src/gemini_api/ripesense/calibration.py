"""White-reference and empty-chamber calibration, stored in data/calibration.json."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .hardware.spectral import measure


def _file(cfg) -> Path:
    return cfg.data_dir / "calibration.json"


def load(cfg) -> dict:
    f = _file(cfg)
    return json.loads(f.read_text()) if f.exists() else {}


def save(cfg, data: dict):
    _file(cfg).write_text(json.dumps(data, indent=2))


def calibrate_white(cfg, devices) -> dict:
    """Run with the white puck in the seat and the door shut."""
    m = measure(devices.spectral, devices.io, cfg["spectral"])
    data = load(cfg)
    data["white"] = {"dark": m["dark"], "white": m["white"], "ir": m["ir"], "t": time.time(),
                     "saturated": m["saturated"]}
    save(cfg, data)
    return data["white"]


def capture_empty(cfg, devices) -> dict:
    """Run with the chamber empty: stores a reference colour image and depth map."""
    io = devices.io
    io.set("led_strip", 1)
    try:
        frame = devices.camera.capture()
    finally:
        io.set("led_strip", 0)
    d = cfg.data_dir
    cv2.imwrite(str(d / "empty_color.png"), frame.color)
    if frame.depth is not None:
        np.save(d / "empty_depth.npy", frame.depth.astype(np.float32))
    data = load(cfg)
    floor = float(np.nanmedian(frame.depth)) if frame.depth is not None else None
    data["empty"] = {"t": time.time(), "floor_m": floor}
    save(cfg, data)
    return data["empty"]


def empty_reference(cfg):
    d = cfg.data_dir
    img = cv2.imread(str(d / "empty_color.png")) if (d / "empty_color.png").exists() else None
    depth = np.load(d / "empty_depth.npy") if (d / "empty_depth.npy").exists() else None
    return img, depth
