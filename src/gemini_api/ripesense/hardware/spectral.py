"""AS7341 11-channel spectral sensor (I2C 0x39), or a mock that follows the simulated fruit."""
from __future__ import annotations

import logging
import time

import numpy as np

from .sim import WORLD

log = logging.getLogger(__name__)

# AS7341 channel names in read order and their centre wavelengths (nm)
CHANNELS = ["f415", "f445", "f480", "f515", "f555", "f590", "f630", "f680", "clear", "nir"]
WAVELENGTHS = [415, 445, 480, 515, 555, 590, 630, 680]
FULL_SCALE = 65535


class AS7341Sensor:
    def __init__(self, cfg: dict):
        import board
        from adafruit_as7341 import AS7341
        self.cfg = cfg
        self.dev = AS7341(board.I2C())
        self.dev.atime = cfg["atime"]
        self.dev.astep = cfg["astep"]
        self._set_gain(cfg["gain"])

    def _set_gain(self, gain):
        try:
            from adafruit_as7341 import Gain
            name = "GAIN_0_5X" if gain == 0.5 else f"GAIN_{int(gain)}X"
            self.dev.gain = getattr(Gain, name)
        except Exception as e:  # pragma: no cover - library version differences
            log.warning("could not set AS7341 gain %s: %s", gain, e)

    def read(self) -> dict:
        """One reading: 8 spectral channels + clear + NIR, raw counts."""
        f = list(self.dev.all_channels)
        vals = f + [self.dev.channel_clear, self.dev.channel_nir]
        return dict(zip(CHANNELS, (float(v) for v in vals)))


class MockSpectral:
    """Reflectance of the simulated fruit x the active LED spectrum + dark offset + noise."""

    def __init__(self, cfg: dict, io):
        self.cfg = cfg
        self.io = io
        self.rng = np.random.default_rng()

    @staticmethod
    def fruit_reflectance(r: float) -> np.ndarray:
        wl = np.array(WAVELENGTHS + [580, 870], float)   # clear ~ broadband, nir ~ 870
        green = 0.10 + 0.25 * np.exp(-((wl - 550) / 40) ** 2) + 0.45 * (wl > 700)
        green[wl == 680] = 0.06                            # chlorophyll absorption
        yellow = 0.08 + 0.65 / (1 + np.exp(-(wl - 505) / 15)) + 0.1 * (wl > 700)
        brown = 0.05 + 0.25 / (1 + np.exp(-(wl - 560) / 40)) + 0.15 * (wl > 700)
        if r < 0.35:
            k = r / 0.35
            return (1 - k) * green + k * yellow
        k = min(1.0, (r - 0.35) / 0.6)
        return (1 - k) * yellow + k * brown

    def read(self) -> dict:
        white = np.array([3e3, 9e3, 6e3, 11e3, 14e3, 13e3, 11e3, 7e3, 40e3, 300.0])
        ir = np.array([20, 20, 20, 20, 20, 20, 30, 60, 900, 18e3], float)
        light = white * self.io.state.get("led_white", 0) + ir * self.io.state.get("led_ir", 0)
        refl = self.fruit_reflectance(WORLD.ripeness) if WORLD.fruit_present else np.full(10, 0.02)
        if WORLD.puck:
            refl = np.full(10, 0.95)
        counts = refl * light + 120 + self.rng.normal(0, 25, 10)
        return dict(zip(CHANNELS, np.clip(counts, 0, FULL_SCALE).astype(float)))


def make_spectral(cfg: dict, io):
    if cfg["kind"] == "as7341":
        try:
            return AS7341Sensor(cfg)
        except Exception as e:  # pragma: no cover - hardware only
            log.error("AS7341 not found (%s); using mock spectral sensor", e)
    return MockSpectral(cfg, io)


def read_averaged(sensor, n: int) -> dict:
    vals = [sensor.read() for _ in range(max(1, n))]
    return {k: float(np.mean([v[k] for v in vals])) for k in CHANNELS}


def measure(sensor, io, cfg: dict) -> dict:
    """Dark, white-LED and IR-LED readings with everything else off."""
    settle = cfg["led_settle_s"]
    n = cfg["reads"]
    io.set("led_strip", 0)
    io.set("led_white", 0)
    io.set("led_ir", 0)
    time.sleep(settle)
    dark = read_averaged(sensor, n)
    io.set("led_white", 1)
    time.sleep(settle)
    white = read_averaged(sensor, n)
    io.set("led_white", 0)
    io.set("led_ir", 1)
    time.sleep(settle)
    ir = read_averaged(sensor, n)
    io.set("led_ir", 0)
    saturated = any(v >= 0.98 * FULL_SCALE for d in (white, ir) for v in d.values())
    return {"dark": dark, "white": white, "ir": ir, "saturated": saturated}
