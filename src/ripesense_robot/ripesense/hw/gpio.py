"""Touch pad, door reed switch and MOSFET outputs via gpiozero (lgpio on Pi 5)."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class PiIO:
    def __init__(self, cfg: dict):
        from gpiozero import Button, DigitalOutputDevice
        g = cfg["gpio"]
        self.require_door = g.get("require_door", True)
        # TTP223 drives its OUT pin high while touched.
        self.touch = Button(g["touch_pin"], pull_up=False, bounce_time=0.05)
        # Reed switch to GND with internal pull-up: low = magnet present = door shut.
        self.reed = Button(g["reed_pin"], pull_up=True, bounce_time=0.05) if self.require_door else None
        self._closed_level = g.get("reed_closed_level", 0)
        self.led_strip = DigitalOutputDevice(g["led_strip_pin"])
        self.spec_white = DigitalOutputDevice(g["spectral_white_pin"])
        self.spec_ir = DigitalOutputDevice(g["spectral_ir_pin"])
        self.fan = DigitalOutputDevice(g["fan_pin"])

    def door_closed(self) -> bool:
        if self.reed is None:
            return True
        pressed = self.reed.is_pressed  # True when pulled low
        return pressed if self._closed_level == 0 else not pressed

    def on_touch(self, cb):
        self.touch.when_pressed = cb

    def on_door_open(self, cb):
        if self.reed is None:
            return
        if self._closed_level == 0:
            self.reed.when_released = cb
        else:
            self.reed.when_pressed = cb

    def set(self, name: str, on: bool):
        getattr(self, name).value = 1 if on else 0

    def all_off(self):
        for n in ("led_strip", "spec_white", "spec_ir", "fan"):
            self.set(n, False)


class MockIO:
    def __init__(self, cfg: dict):
        self.require_door = cfg["gpio"].get("require_door", True)
        self.closed = True
        self.state = {"led_strip": False, "spec_white": False, "spec_ir": False, "fan": False}
        self._touch_cb = None
        self._door_cb = None

    def door_closed(self) -> bool:
        return self.closed

    def on_touch(self, cb):
        self._touch_cb = cb

    def on_door_open(self, cb):
        self._door_cb = cb

    def set_door(self, closed: bool):
        was = self.closed
        self.closed = closed
        if was and not closed and self._door_cb:
            self._door_cb()

    def press(self):
        if self._touch_cb:
            self._touch_cb()

    def set(self, name: str, on: bool):
        self.state[name] = on

    def all_off(self):
        for k in self.state:
            self.state[k] = False


def make_io(cfg: dict, mock: bool):
    return MockIO(cfg) if mock else PiIO(cfg)
