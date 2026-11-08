"""AS7341 11-channel spectral sensor (Adafruit CircuitPython driver via Blinka).

Reads raw counts; reflectance is computed later with the dark reading and the
white calibration puck: R_i = (S_i - D_i) / (W_i - D_i).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..sim.fruit import SPECTRAL_CHANNELS

log = logging.getLogger(__name__)

_ATTRS = {
    "f1_415": "channel_415nm", "f2_445": "channel_445nm", "f3_480": "channel_480nm",
    "f4_515": "channel_515nm", "f5_555": "channel_555nm", "f6_590": "channel_590nm",
    "f7_630": "channel_630nm", "f8_680": "channel_680nm", "clear": "channel_clear",
    "nir": "channel_nir",
}


class AS7341Sensor:
    def __init__(self, cfg: dict):
        import board
        from adafruit_as7341 import AS7341, Gain
        i2c = board.I2C()
        self.dev = AS7341(i2c, address=cfg.get("i2c_address", 0x39))
        gains = {0.5: Gain.GAIN_0_5X, 1: Gain.GAIN_1X, 2: Gain.GAIN_2X, 4: Gain.GAIN_4X,
                 8: Gain.GAIN_8X, 16: Gain.GAIN_16X, 32: Gain.GAIN_32X, 64: Gain.GAIN_64X,
                 128: Gain.GAIN_128X, 256: Gain.GAIN_256X, 512: Gain.GAIN_512X}
        self.dev.gain = gains[cfg.get("gain", 64)]
        self.dev.atime = cfg.get("atime", 100)
        self.dev.astep = cfg.get("astep", 999)
        self.dev.led = False  # the on-board LED is not used; ours are on MOSFETs

    def read(self, n: int = 3) -> dict:
        """Average of n reads of every channel (raw counts)."""
        acc = {k: 0.0 for k in _ATTRS}
        for _ in range(n):
            # all_channels triggers both SMUX passes; read clear/nir after.
            vals = self.dev.all_channels
            for k, v in zip(list(_ATTRS)[:8], vals):
                acc[k] += v
            acc["clear"] += self.dev.channel_clear
            acc["nir"] += self.dev.channel_nir
        return {k: v / n for k, v in acc.items()}


class MockSpectral:
    """Simulates counts = (reflectance * illum + dark) for the current sim fruit.
    `illum` is set by the scan controller through set_illumination()."""

    DARK = 40.0
    WHITE_GAIN = {"white": 9000.0, "ir": 6000.0, "off": 0.0}

    def __init__(self, cfg: dict):
        self.illum = "off"
        self.target = "fruit"  # or "white" while calibrating

    def set_illumination(self, mode: str):
        self.illum = mode

    def read(self, n: int = 3) -> dict:
        import random

        from ..sim import fruit as sim
        time.sleep(0.15 * n)
        refl = ({k: 0.95 for k in SPECTRAL_CHANNELS} if self.target == "white"
                else sim.current.reflectance())
        g = self.WHITE_GAIN[self.illum]
        out = {}
        for k in SPECTRAL_CHANNELS:
            # White LEDs give almost no NIR; the IR LED gives almost no visible.
            if self.illum == "white":
                eff = g * (0.05 if k == "nir" else 1.0)
            elif self.illum == "ir":
                eff = g * (1.0 if k == "nir" else 0.02)
            else:
                eff = 0.0
            out[k] = self.DARK + refl[k] * eff * random.uniform(0.98, 1.02)
        return out


def make_spectral(cfg: dict, mock: bool):
    return MockSpectral(cfg["spectral"]) if mock else AS7341Sensor(cfg["spectral"])


# ---------------------------------------------------------------- calibration
def load_white_ref(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_white_ref(path: Path, ref: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref, indent=2))


def reflectance(sample: dict, dark: dict, white: dict, white_dark: dict | None = None) -> dict:
    """R_i = (S_i - D_i) / (W_i - D_i), clipped to a sane range.

    `dark` is the LEDs-off read taken during this scan; `white_dark` is the dark
    read taken with the white puck (defaults to `dark`)."""
    wd = white_dark or dark
    out = {}
    for k in SPECTRAL_CHANNELS:
        den = white.get(k, 0) - wd.get(k, 0)
        out[k] = float("nan") if den <= 1e-6 else max(-0.1, min(2.0, (sample[k] - dark[k]) / den))
    return out


def combine_illuminations(white_lit: dict, ir_lit: dict) -> dict:
    """Visible channels from the white-LED read, NIR from the IR-LED read."""
    out = {k: white_lit[k] for k in SPECTRAL_CHANNELS if k != "nir"}
    out["nir"] = ir_lit["nir"]
    return out
