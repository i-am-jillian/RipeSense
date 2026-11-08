"""GPIO: touch pad, door reed switch and the four MOSFET outputs (gpiozero, works on the Pi 5)."""
from __future__ import annotations

import logging
import time
from typing import Callable

from .sim import WORLD

log = logging.getLogger(__name__)
OUTPUTS = ("led_strip", "led_white", "led_ir", "fan")


class GpioIO:
    def __init__(self, pins: dict):
        from gpiozero import Button, DigitalOutputDevice
        self.state: dict[str, int] = {}
        self.outs = {name: DigitalOutputDevice(pins[name]) for name in OUTPUTS}
        self.touch = Button(pins["touch"], pull_up=False, bounce_time=0.05)   # TTP223 drives HIGH
        self.door = Button(pins["door"], pull_up=True, bounce_time=0.05) if pins.get("door", -1) >= 0 else None
        self.door_opened_at: float | None = None
        self.door_closed_at: float | None = None
        if self.door is not None:
            self.door.when_released = self._door_opened   # released = magnet gone = door open
            self.door.when_pressed = self._door_closed

    def _door_opened(self):
        self.door_opened_at = time.time()

    def _door_closed(self):
        self.door_closed_at = time.time()

    def on_touch(self, fn: Callable[[], None]):
        self.touch.when_pressed = fn

    def door_closed(self) -> bool | None:
        return None if self.door is None else bool(self.door.is_pressed)

    def set(self, name: str, value: int):
        self.state[name] = int(value)
        self.outs[name].value = int(value)

    def all_off(self):
        for n in OUTPUTS:
            self.set(n, 0)

    def close(self):
        self.all_off()


class MockIO:
    def __init__(self, pins: dict | None = None):
        self.state: dict[str, int] = {n: 0 for n in OUTPUTS}
        self._touch_cb: Callable[[], None] | None = None

    @property
    def door_opened_at(self):
        return WORLD.door_opened_at

    @property
    def door_closed_at(self):
        return WORLD.door_closed_at if WORLD.door_closed else None

    def on_touch(self, fn):
        self._touch_cb = fn

    def press(self):
        """Simulate a finger on the pad."""
        if self._touch_cb:
            self._touch_cb()

    def door_closed(self) -> bool | None:
        return WORLD.door_closed

    def set(self, name: str, value: int):
        self.state[name] = int(value)

    def all_off(self):
        for n in OUTPUTS:
            self.set(n, 0)

    def close(self):
        self.all_off()


def make_io(pins: dict, simulate: bool):
    if not simulate:
        try:
            return GpioIO(pins)
        except Exception as e:  # pragma: no cover - hardware only
            log.error("GPIO unavailable (%s); using mock IO", e)
    return MockIO(pins)
