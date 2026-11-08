"""Training from labelled scans, plus a synthetic-data generator for testing the pipeline."""
from __future__ import annotations

import logging
import random
import time

from .features import (FEATURES, fruit_code, gas_features, image_features, segment, spectral_features)
from .hardware.camera import MockCamera
from .hardware.gas import GasSample
from .hardware.sim import WORLD
from .hardware.spectral import MockSpectral, measure
from .model import FusionModel, HeuristicModel

log = logging.getLogger(__name__)


def train(cfg, store, include_synthetic: bool = False) -> dict:
    rows = store.training_rows(include_synthetic=include_synthetic)
    if len(rows) < 12:
        raise ValueError(f"only {len(rows)} labelled scans; label at least 12 (ideally 100+) first")
    model = FusionModel(cfg["model"]["min_confidence"])
    metrics = model.fit(rows)
    path = cfg.path(cfg["model"]["path"])
    model.save(path)
    metrics["saved_to"] = str(path)
    return metrics


def stage_for(r: float) -> str:
    return "unripe" if r < 0.22 else "ripe" if r < 0.55 else "overripe" if r < 0.82 else "spoiled"


class _SimIO:
    def __init__(self):
        self.state = {}

    def set(self, k, v):
        self.state[k] = v


class _GasMath:
    rises_with_gas = True

    def response(self, raw, base):
        return raw / base


def simulate_dataset(cfg, store, n_fruits: int = 30, scans_per_fruit: int = 8, seed: int = 0) -> int:
    """Synthetic scans of simulated fruit ripening over days. Marked synthetic=1; for testing only."""
    rng = random.Random(seed)
    cam = MockCamera(cfg["camera"], delay=0)
    io = _SimIO()
    spec = MockSpectral(cfg["spectral"], io)
    WORLD.puck = True
    m = measure(spec, io, {**cfg["spectral"], "led_settle_s": 0})
    calib = {"dark": m["dark"], "white": m["white"], "ir": m["ir"]}
    WORLD.puck = False
    gm = _GasMath()
    n = 0
    fruits = ["banana", "avocado", "tomato"]
    for i in range(n_fruits):
        fruit = fruits[i % len(fruits)]
        start = rng.uniform(0.0, 0.3)
        speed = rng.uniform(0.08, 0.14)          # ripeness gained per day
        for k in range(scans_per_fruit):
            day = k * 0.5
            r = min(1.0, start + speed * day + rng.gauss(0, 0.02))
            WORLD.set_fruit(fruit, r)
            frame = cam.capture()
            mask, _, floor = segment(frame.color, frame.depth)
            img_f, _ = image_features(frame, mask, floor)
            spec_f = spectral_features(measure(spec, io, {**cfg["spectral"], "led_settle_s": 0}), calib)
            amp = (0.02 + 0.5 * r ** 2) * 0.55 * rng.uniform(0.7, 1.3)
            t0 = time.time()
            samples = [GasSample(t0 + j, 0.4 + amp * j / 20 + rng.gauss(0, 0.002)) for j in range(21)]
            gas_f = gas_features(samples, 0.4, gm, img_f["i_volume_cm3"])
            features = {"fruit_code": float(fruit_code(fruit)), **img_f, **spec_f, **gas_f}
            # days until 'spoiled' (r = 0.82) at this fruit's speed
            days_left = max(0.0, (0.82 - r) / speed)
            pred = HeuristicModel().predict(features, fruit)
            store.add(fruit_type=fruit, fruit_id=f"sim-{i:03d}", pred=pred, mode="home",
                      features={k2: features.get(k2) for k2 in FEATURES}, raw={"sim_ripeness": r},
                      result={"synthetic": True}, synthetic=True, label_stage=stage_for(r),
                      label_days=round(days_left, 1), ts=time.time())
            n += 1
    return n

