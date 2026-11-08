"""`python -m ripesense doctor`: find each part and say what is missing, in plain words."""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from pathlib import Path

I2C = {0x39: "AS7341 spectral", 0x48: "ADS1115 ADC (MQ gas)", 0x76: "BME680/688", 0x77: "BME680/688"}


def _ok(name, good, detail=""):
    print(f"  [{'ok' if good else '!!'}] {name:28s} {detail}")
    return good


def run_doctor(cfg) -> int:
    bad = 0
    print("Python packages")
    for mod, why in [("pyrealsense2", "RealSense camera"), ("board", "Blinka (I2C)"),
                     ("adafruit_as7341", "AS7341 driver"), ("adafruit_ads1x15", "ADS1115 driver"),
                     ("adafruit_bme680", "BME680 driver"), ("gpiozero", "GPIO"), ("onnxruntime", "image model"),
                     ("sklearn", "fusion model"), ("google.genai", "Gemini tips (optional)")]:
        try:
            importlib.import_module(mod)
            _ok(why, True, mod)
        except Exception as e:
            optional = mod in ("onnxruntime", "google.genai", "adafruit_bme680", "adafruit_ads1x15")
            _ok(why, optional, f"{mod} missing" + (" (optional)" if optional else f": {e.__class__.__name__}"))
            bad += 0 if optional else 1

    print("I2C bus 1")
    found = set()
    if shutil.which("i2cdetect"):
        out = subprocess.run(["i2cdetect", "-y", "1"], capture_output=True, text=True).stdout
        for line in out.splitlines()[1:]:
            for cell in line.split()[1:]:
                if cell not in ("--", "UU"):
                    try:
                        found.add(int(cell, 16))
                    except ValueError:
                        pass
        for addr, name in I2C.items():
            if addr in found or addr in (0x39,):
                bad += 0 if _ok(name, addr in found, f"0x{addr:02x}") else 1
    else:
        bad += 0 if _ok("i2c-tools", False, "sudo apt install i2c-tools; enable I2C in raspi-config") else 1

    print("Camera")
    try:
        import pyrealsense2 as rs
        devs = rs.context().query_devices()
        names = [d.get_info(rs.camera_info.name) for d in devs]
        bad += 0 if _ok("RealSense", bool(names), ", ".join(names) or "not detected (USB 3 port?)") else 1
    except Exception as e:
        bad += 0 if _ok("RealSense", False, str(e)) else 1

    print("Screen and files")
    _ok("SPI screen framebuffer", Path("/dev/fb1").exists() or Path("/dev/fb0").exists(),
        "/dev/fb1" if Path("/dev/fb1").exists() else "check the screen driver")
    cal = Path(cfg.data_dir / "calibration.json")
    _ok("calibration", cal.exists(), "run: python -m ripesense calibrate" if not cal.exists() else str(cal))
    model = cfg.path(cfg["model"]["path"])
    _ok("trained model", model.exists(), str(model) if model.exists() else "heuristic until you train")
    _ok("GEMINI_API_KEY", bool(os.environ.get("GEMINI_API_KEY")), "optional")
    print("\nAll required parts found." if bad == 0 else f"\n{bad} required item(s) missing.")
    return 1 if bad else 0
