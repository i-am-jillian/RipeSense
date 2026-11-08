"""Generate a synthetic scan log that mimics the real collection protocol
(fruits × twice-daily sessions × 3 turned scans) so the whole training
pipeline can be exercised before real data exists.

  RIPESENSE_DATA_DIR=data_sim python -m ripesense.sim.make_dataset --fruits 16

Synthetic data proves the code runs. It says nothing about real accuracy.
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta

from .. import config, store
from ..hw import gas as gas_mod
from ..hw import spectral as spec_mod
from . import fruit as simf


def simulate_scan(f: simf.SimFruit, cfg: dict, ts: datetime, drop_p: float = 0.08) -> dict:
    cam = cfg["camera"]
    bgr, depth = f.render(cam["width"], cam["height"], cam["tray_depth_mm"])
    # Spectral: mock sensor counts with a per-scan LED brightness wobble.
    ms = spec_mod.MockSpectral({})
    simf.current = f
    wobble = random.uniform(0.93, 1.07)
    ms.WHITE_GAIN = {k: v * wobble for k, v in ms.WHITE_GAIN.items()}
    ms.set_illumination("off"); dark = ms.read(1)
    ms.set_illumination("white"); wl = ms.read(1)
    ms.set_illumination("ir"); ir = ms.read(1)
    ms.target = "white"
    ms.set_illumination("white"); ww = ms.read(1)
    ms.set_illumination("ir"); wir = ms.read(1)
    white = spec_mod.combine_illuminations(ww, wir)
    refl = spec_mod.reflectance(spec_mod.combine_illuminations(wl, ir), dark, white)
    # Gas: 120 s of 1 Hz samples, BME680-style raw resistance.
    base = 120000.0 * random.uniform(0.95, 1.05)
    series = []
    for t in range(0, 121):
        resp = f.gas_response(t) + random.gauss(0, 0.004)
        raw = base * (1 - resp)
        series.append((float(t), gas_mod.response(raw, base, "bme680")))
    # E-nose: BME688 heater-profile resistances, empty chamber vs after 120 s.
    w = f.enose_weights()
    pb = [b * random.uniform(0.95, 1.05) for b in (260000, 180000, 120000, 80000, 52000)]
    resp_end = f.gas_response(120)
    profile = {"baseline": pb, "end": [b * (1 - min(0.95, resp_end * wi) - random.gauss(0, 0.003)) for b, wi in zip(pb, w)]}
    if random.random() < drop_p:      # cold/failed gas sensor
        series, profile = [], None
    if random.random() < drop_p / 2:  # failed spectral read
        refl = {}
    key = f"{ts:%Y%m%d_%H%M%S}_{f.fruit_id}_{random.randrange(1000)}"
    rgb_p, depth_p = store.save_images(key, bgr, depth)
    ir_img, ir_d = f.render_ir(424, 240, cam["tray_depth_mm"])
    ir_p, ir_dp = store.save_ir(key, ir_img, ir_d)
    days_label = max(0, round(f.days_left)) if f.stage != "spoiled" else 0
    return {
        "fruit_id": f.fruit_id, "fruit_type": f.fruit_type,
        "timestamp": ts.isoformat(timespec="seconds"), "session": store.session_name(ts), "mode": "home",
        "rgb_path": rgb_p, "depth_path": depth_p,
        **{f"spec_{c}": refl.get(c) for c in simf.SPECTRAL_CHANNELS},
        "spec_white_lit": wl, "spec_ir_lit": ir, "white_ref": white, "dark_ref": dark,
        "gas_kind": "bme680", "gas_baseline": base, "gas_120s": base * (1 - series[-1][1]) if series else None,
        "gas_series": series, "temp": 21.5, "humidity": 45.0,
        "stage_label": f.stage, "days_left_label": days_label, "source": "sim",
        "ir_path": ir_p, "ir_depth_path": ir_dp, "gas_profile": profile,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruits", type=int, default=16)
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--scans", type=int, default=3)
    ap.add_argument("--types", default="banana,avocado")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    cfg = config.load()
    types = a.types.split(",")
    start = datetime(2026, 11, 5, 9, 0)
    n = 0
    for i in range(a.fruits):
        ft = types[i % len(types)]
        d0 = random.uniform(-1.0, 8.0)
        bruises = random.choice([0, 0, 1, 2])
        for s in range(a.sessions):
            ts0 = start + timedelta(hours=12 * s)
            for k in range(a.scans):
                f = simf.SimFruit(fruit_type=ft, days_left=d0 - 0.5 * s, fruit_id=f"{ft[0].upper()}{i:02d}",
                                  rotation=random.uniform(-60, 60), hidden_bruises=bruises,
                                  seed=hash((i, s, k)) & 0x3FFFFFFF)
                store.insert(simulate_scan(f, cfg, ts0 + timedelta(minutes=3 * k)))
                n += 1
    print(f"wrote {n} synthetic scans → {config.data_dir()}/scans.csv")


if __name__ == "__main__":
    main()
