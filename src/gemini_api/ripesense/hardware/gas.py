"""Gas sensing: MQ module through an ADS1115, a BME680/688, or a mock.

A GasMonitor thread samples all the time, so the empty-chamber baseline from
just before the door opened is always available when a scan starts.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from .sim import WORLD

log = logging.getLogger(__name__)


@dataclass
class GasSample:
    t: float
    raw: float                 # MQ: volts (up = more gas). BME68x: ohms (down = more gas)
    temp: float | None = None
    humidity: float | None = None


class MQviaADS1115:
    rises_with_gas = True

    def __init__(self, cfg: dict):
        import board
        try:  # adafruit-circuitpython-ads1x15 >= 3
            from adafruit_ads1x15 import ADS1115, AnalogIn, ads1x15
            ads = ADS1115(board.I2C())
            pin = getattr(ads1x15.Pin, f"A{cfg['ads_channel']}")
        except ImportError:  # 2.x
            import adafruit_ads1x15.ads1115 as ADS
            from adafruit_ads1x15.analog_in import AnalogIn
            ads = ADS.ADS1115(board.I2C())
            pin = getattr(ADS, f"P{cfg['ads_channel']}")
        self.chan = AnalogIn(ads, pin)

    def read(self) -> GasSample:
        return GasSample(time.time(), float(self.chan.voltage))


class BME680Sensor:
    rises_with_gas = False

    def __init__(self, cfg: dict):
        import adafruit_bme680
        import board
        self.dev = adafruit_bme680.Adafruit_BME680_I2C(board.I2C())

    def read(self) -> GasSample:
        return GasSample(time.time(), float(self.dev.gas), float(self.dev.temperature),
                         float(self.dev.relative_humidity))


class MockGas:
    """Baseline 0.40 V; a closed door with fruit inside adds gas in proportion to ripeness."""
    rises_with_gas = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rng = np.random.default_rng()

    def read(self) -> GasSample:
        v = 0.40 + self.rng.normal(0, 0.002)
        if WORLD.fruit_present and WORLD.door_closed:
            dt = time.time() - WORLD.door_closed_at
            tau = max(self.cfg["scan_seconds"], 1) * 1.2
            amp = 0.02 + 0.5 * WORLD.ripeness ** 2
            v += amp * (1 - np.exp(-dt / tau))
        return GasSample(time.time(), float(v), 22.0 + self.rng.normal(0, 0.1), 45.0)


def make_gas(cfg: dict):
    try:
        if cfg["kind"] == "mq_ads1115":
            return MQviaADS1115(cfg)
        if cfg["kind"] == "bme680":
            return BME680Sensor(cfg)
    except Exception as e:  # pragma: no cover - hardware only
        log.error("gas sensor '%s' not found (%s); using mock", cfg["kind"], e)
    return MockGas(cfg)


class GasMonitor(threading.Thread):
    def __init__(self, sensor, cfg: dict):
        super().__init__(daemon=True, name="gas-monitor")
        self.sensor = sensor
        self.cfg = cfg
        self.buf: collections.deque[GasSample] = collections.deque(maxlen=4000)
        self.started_at = time.time()
        self._halt = threading.Event()
        self._lock = threading.Lock()

    def run(self):
        while not self._halt.is_set():
            try:
                s = self.sensor.read()
                with self._lock:
                    self.buf.append(s)
            except Exception as e:  # pragma: no cover
                log.warning("gas read failed: %s", e)
            self._halt.wait(self.cfg["sample_period_s"])

    def stop(self):
        self._halt.set()

    @property
    def warm(self) -> bool:
        return time.time() - self.started_at >= 60 * self.cfg["warmup_minutes"]

    @property
    def warmup_left_s(self) -> float:
        return max(0.0, 60 * self.cfg["warmup_minutes"] - (time.time() - self.started_at))

    def window(self, t0: float, t1: float) -> list[GasSample]:
        with self._lock:
            return [s for s in self.buf if t0 <= s.t <= t1]

    def latest(self) -> GasSample | None:
        with self._lock:
            return self.buf[-1] if self.buf else None

    def baseline(self, opened: float | None, closed: float | None) -> tuple[float | None, str]:
        """Empty-chamber reference for the scan.

        Best: room air sampled while the door stood open (the old fruit is out, fresh air is in).
        Next: the window just before the door opened. Returns (value, source).
        """
        if opened and closed and closed - opened > 1.0:
            s = self.window(opened + min(2.0, 0.25 * (closed - opened)), closed)
            if len(s) >= 3:
                return float(np.median([x.raw for x in s])), "door-open air"
        if opened:
            w = self.cfg["baseline_window_s"]
            s = self.window(opened - w, opened)
            if len(s) >= 3:
                return float(np.median([x.raw for x in s])), "before door opened"
        return None, "none"

    def response(self, raw: float, baseline: float) -> float:
        """1.0 = same as the empty chamber; above 1 = more volatiles, for either sensor type."""
        if baseline is None or raw <= 0 or baseline <= 0:
            return float("nan")
        return raw / baseline if self.sensor.rises_with_gas else baseline / raw
