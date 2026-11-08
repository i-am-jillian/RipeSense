"""Configuration: defaults below, overridden by config.toml and RIPESENSE_* env vars."""
from __future__ import annotations

import copy
import os
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

DEFAULTS: dict = {
    "app": {"data_dir": "data", "simulate": False, "default_fruit": "banana", "mode": "home",
            "host": "0.0.0.0", "port": 8000},
    "pins": {"touch": 5, "door": 6, "led_strip": 12, "led_white": 13, "led_ir": 19, "fan": 26},
    "camera": {"kind": "realsense", "color_size": [640, 480], "depth_size": [480, 270], "fps": 15,
               "warmup_frames": 20, "opencv_index": 0},
    "spectral": {"kind": "as7341", "gain": 64, "atime": 100, "astep": 999, "reads": 3, "led_settle_s": 0.15},
    "gas": {"kind": "mq_ads1115", "ads_channel": 0, "sample_period_s": 1.0, "scan_seconds": 120,
            "baseline_window_s": 30, "warmup_minutes": 20},
    "model": {"path": "models/ripesense.joblib", "image_model": "models/fruit_classifier.onnx",
              "min_confidence": 0.6},
    "gemini": {"enabled": True, "model": "gemini-3.8-flash", "fallback_models": ["gemini-2.5-flash"],
               "timeout_s": 15},
    "qr": {"target": "lan", "public_url": ""},
    "store": {"currency": "$", "prices": {"banana": 0.35, "avocado": 1.50, "tomato": 0.80, "apple": 0.90,
                                         "mango": 1.75, "other": 1.00}},
    "timeseries": {"postgres_url": ""},
}

# When simulating, use the mock devices and a short gas wait so a scan takes seconds.
SIM_OVERRIDES = {"camera": {"kind": "mock"}, "spectral": {"kind": "mock", "led_settle_s": 0.0},
                 "gas": {"kind": "mock", "scan_seconds": 4, "sample_period_s": 0.2, "warmup_minutes": 0}}


def _merge(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


class Config(dict):
    """Nested dict with attribute-style section access: cfg.camera['kind']."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(name) from e

    @property
    def root(self) -> Path:
        return Path(self.get("_root", "."))

    def path(self, p: str) -> Path:
        """Resolve a path from the config relative to the project root."""
        q = Path(p)
        return q if q.is_absolute() else self.root / q

    @property
    def data_dir(self) -> Path:
        d = self.path(self["app"]["data_dir"])
        d.mkdir(parents=True, exist_ok=True)
        return d


def load(path: str | os.PathLike | None = None, simulate: bool | None = None, **overrides) -> Config:
    cfg = copy.deepcopy(DEFAULTS)
    root = Path(__file__).resolve().parent.parent
    candidate = Path(path) if path else root / "config.toml"
    file_cfg: dict = {}
    if candidate.exists():
        with open(candidate, "rb") as f:
            file_cfg = tomllib.load(f)
        root = candidate.resolve().parent
    sim = bool(file_cfg.get("app", {}).get("simulate", False))
    if os.environ.get("RIPESENSE_SIMULATE") in ("1", "true", "yes"):
        sim = True
    if simulate is not None:
        sim = simulate
    if sim:   # simulator defaults first, so values in config.toml still win
        _merge(cfg, copy.deepcopy(SIM_OVERRIDES))
    _merge(cfg, file_cfg)
    cfg["app"]["simulate"] = sim
    if sim:   # never touch real hardware while simulating
        for sec in ("camera", "spectral", "gas"):
            cfg[sec]["kind"] = SIM_OVERRIDES[sec]["kind"]
    for section, values in overrides.items():
        _merge(cfg.setdefault(section, {}), values)
    cfg["_root"] = str(root)
    return Config(cfg)
