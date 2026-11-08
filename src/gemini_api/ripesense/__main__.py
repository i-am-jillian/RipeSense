"""Command line:  python -m ripesense <command>

  run              start the kiosk server (touch pad + screen + /collect page)
  calibrate        white reference (puck) and empty-chamber reference, with prompts
  train            train the fusion model from labelled scans
  simulate-data    add synthetic labelled scans (testing the pipeline only)
  doctor           check every sensor and print what was found
Add --sim to any command to run without hardware.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from . import calibration, config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ripesense", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "calibrate", "train", "simulate-data", "doctor"])
    ap.add_argument("--config", help="path to config.toml")
    ap.add_argument("--sim", action="store_true", help="simulate all hardware")
    ap.add_argument("--synthetic", action="store_true", help="train: include synthetic scans")
    ap.add_argument("--fruits", type=int, default=30, help="simulate-data: number of simulated fruits")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = config.load(a.config, simulate=True if a.sim else None)

    if a.command == "run":
        from .web import create_app
        app = create_app(cfg)
        print(f"RipeSense on http://{cfg['app']['host']}:{cfg['app']['port']}  (kiosk: /, data collection: /collect)")
        app.run(host=cfg["app"]["host"], port=cfg["app"]["port"], threaded=True, use_reloader=False)

    elif a.command == "calibrate":
        from .hardware import build_devices
        from .hardware.sim import WORLD
        dev = build_devices(cfg)
        try:
            input("1/2  Put the WHITE puck in the seat, shut the door, press Enter… ")
            WORLD.puck = cfg["app"]["simulate"]
            w = calibration.calibrate_white(cfg, dev)
            WORLD.puck = False
            print("     white reference saved" + ("  (WARNING: saturated, lower [spectral] gain)" if w["saturated"] else ""))
            input("2/2  Take everything out, shut the door, press Enter… ")
            WORLD.fruit_present = False if cfg["app"]["simulate"] else WORLD.fruit_present
            e = calibration.capture_empty(cfg, dev)
            print(f"     empty chamber saved, floor at {e['floor_m'] * 1000:.0f} mm" if e["floor_m"] else
                  "     empty chamber saved (no depth)")
        finally:
            dev.close()

    elif a.command == "train":
        from .storage import Store
        from .train import train
        try:
            m = train(cfg, Store(cfg.data_dir), include_synthetic=a.synthetic)
        except ValueError as e:
            sys.exit(f"cannot train: {e}")
        print(json.dumps(m, indent=1))

    elif a.command == "simulate-data":
        from .storage import Store
        from .train import simulate_dataset
        n = simulate_dataset(cfg, Store(cfg.data_dir), n_fruits=a.fruits)
        print(f"added {n} synthetic labelled scans (train with --synthetic to use them)")

    elif a.command == "doctor":
        from .doctor import run_doctor
        sys.exit(run_doctor(cfg))


if __name__ == "__main__":
    main()
