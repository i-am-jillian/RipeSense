"""Gas sensor: an MQ module through an ADS1115, or a BME680/BME688 on I2C.

Both are reported as a *response* that rises with VOCs, so downstream code
doesn't care which sensor is fitted:
  MQ:     response = (V - V0) / V0        (output voltage rises with gas)
  BME68x: response = (R0 - R) / R0        (gas resistance falls with gas)
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque

log = logging.getLogger(__name__)


class MQSensor:
    kind = "mq"

    def __init__(self, cfg: dict):
        import adafruit_ads1x15.ads1115 as ADS
        import board
        from adafruit_ads1x15.analog_in import AnalogIn
        ads = ADS.ADS1115(board.I2C(), address=cfg.get("ads1115_address", 0x48))
        ads.gain = 2 / 3  # ±6.144 V full scale, safe for a divided 5 V MQ output
        pin = [ADS.P0, ADS.P1, ADS.P2, ADS.P3][cfg.get("ads1115_channel", 0)]
        self.chan = AnalogIn(ads, pin)
        self.ratio = float(cfg.get("divider_ratio", 1.0))

    def read(self) -> dict:
        return {"raw": self.chan.voltage * self.ratio, "temp": None, "humidity": None}


class BME680Sensor:
    kind = "bme680"

    def __init__(self, cfg: dict):
        import adafruit_bme680
        import board
        self.dev = adafruit_bme680.Adafruit_BME680_I2C(board.I2C(), address=cfg.get("bme680_address", 0x77))
        # E-nose mode (BME688 best, BME680 works): step the hotplate through
        # several temperatures each sample. Different VOCs react most at different
        # temperatures, so one sensor gives a 5-channel "smell fingerprint".
        self.profile = list(cfg.get("heater_profile") or []) if hasattr(self.dev, "set_gas_heater") else []
        self.step_ms = int(cfg.get("heater_step_ms", 100))
        if cfg.get("heater_profile") and not self.profile:
            log.warning("adafruit_bme680 has no set_gas_heater(); e-nose profile disabled (pip install -U adafruit-circuitpython-bme680)")
        if self.profile:
            self.dev._min_refresh_time = 0   # each property read = a fresh forced measurement

    def read(self) -> dict:
        out = {"temp": float(self.dev.temperature), "humidity": float(self.dev.relative_humidity)}
        if self.profile:
            prof = []
            for t in self.profile:
                self.dev.set_gas_heater(int(t), self.step_ms)
                prof.append(float(self.dev.gas))
            self.dev.set_gas_heater(320, 150)
            out["profile"] = prof
        out["raw"] = float(self.dev.gas)
        return out


class MockGas:
    """BME680-like: resistance drops as the sim fruit emits VOCs once 'sealed'."""

    def __init__(self, cfg: dict):
        self.kind = cfg.get("type", "bme680")
        self.sealed_at: float | None = None
        self.time_scale = 1.0  # >1 compresses time in fast mock mode
        self.profile = list(cfg.get("heater_profile") or []) if self.kind != "mq" else []

    def seal(self, sealed: bool):
        self.sealed_at = time.time() if sealed else None

    def read(self) -> dict:
        import random

        from ..sim import fruit as sim
        resp = 0.0
        if self.sealed_at is not None:
            resp = sim.current.gas_response((time.time() - self.sealed_at) * self.time_scale)
        noise = random.gauss(0, 0.003)
        out = {"temp": 21.5 + random.gauss(0, 0.05), "humidity": 45 + random.gauss(0, 0.3)}
        if self.kind == "mq":
            out["raw"] = 0.9 * (1 + resp + noise)
        else:
            out["raw"] = 120000.0 * (1 - resp - noise)
            if self.profile:
                w = sim.current.enose_weights() if self.sealed_at is not None else [0.0] * len(self.profile)
                base = [260000, 180000, 120000, 80000, 52000]
                out["profile"] = [b * (1 - min(0.95, resp * wi) - random.gauss(0, 0.003)) for b, wi in zip(base, w)]
        return out


def response(raw: float, baseline: float, kind: str) -> float:
    if not baseline:
        return float("nan")
    return (raw - baseline) / baseline if kind == "mq" else (baseline - raw) / baseline


def make_gas(cfg: dict, mock: bool):
    g = cfg["gas"]
    if mock:
        return MockGas(g)
    return MQSensor(g) if g["type"] == "mq" else BME680Sensor(g)


class GasMonitor:
    """Samples the gas sensor in the background at ~1 Hz into a ring buffer.

    The doc's baseline is the *empty* chamber just before the door opened, so
    the 120 s reading measures only what the fruit added. mark_door_open() pins
    that baseline; if there's no reed switch, the scan falls back to the median
    of the first few seconds."""

    def __init__(self, sensor, hz: float = 1.0, keep_s: int = 900):
        self.sensor = sensor
        self.period = 1.0 / hz
        self.buf: deque = deque(maxlen=int(keep_s * hz) + 10)
        self.lock = threading.Lock()
        self.empty_baseline: float | None = None
        self.empty_profile: list[float] | None = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True, name="gas-monitor")

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                r = self.sensor.read()
                with self.lock:
                    self.buf.append((time.time(), r))
            except Exception as e:  # noqa: BLE001
                log.warning("gas read failed: %s", e)
            self._stop.wait(self.period)

    def latest(self) -> dict | None:
        with self.lock:
            return self.buf[-1][1] if self.buf else None

    def window(self, t0: float, t1: float) -> list:
        with self.lock:
            return [(t, r) for t, r in self.buf if t0 <= t <= t1]

    def mark_door_open(self):
        """Call when the door opens: median of the previous 10 s = empty baseline."""
        now = time.time()
        win = self.window(now - 10, now)
        pts = sorted(r["raw"] for _, r in win)
        if pts:
            self.empty_baseline = pts[len(pts) // 2]
        self.empty_profile = median_profile([r for _, r in win])


def median_profile(readings: list[dict]) -> list[float] | None:
    """Per-step median of the heater-profile resistances (None if no profile)."""
    profs = [r["profile"] for r in readings if r.get("profile")]
    if not profs:
        return None
    import numpy as np
    return [float(v) for v in np.median(np.array(profs, float), axis=0)]
