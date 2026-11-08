"""Bench test every device, one at a time, on the Pi:

    python -m ripesense.hw.check            # all
    python -m ripesense.hw.check spectral gas outputs

Prints PASS/FAIL per device so wiring faults show up before the full scan.
"""
from __future__ import annotations

import sys
import time

from .. import config


def _ok(name, msg=""):
    print(f"  PASS  {name}  {msg}")


def _fail(name, e):
    print(f"  FAIL  {name}  {e}")


def i2c():
    try:
        import board
        bus = board.I2C()
        while not bus.try_lock():
            pass
        addrs = [hex(a) for a in bus.scan()]
        bus.unlock()
        _ok("i2c", f"devices: {addrs}  (expect 0x39 AS7341; 0x48 ADS1115 or 0x77 BME680)")
    except Exception as e:  # noqa: BLE001
        _fail("i2c", e)


def spectral(cfg):
    from . import gpio, spectral as sp
    try:
        io = gpio.make_io(cfg, False)
        s = sp.make_spectral(cfg, False)
        io.all_off()
        time.sleep(0.3)
        dark = s.read(1)
        io.set("spec_white", True)
        time.sleep(0.3)
        white = s.read(1)
        io.set("spec_white", False)
        io.set("spec_ir", True)
        time.sleep(0.3)
        ir = s.read(1)
        io.set("spec_ir", False)
        print("        channel     dark    white       ir")
        for k in dark:
            print(f"        {k:8s} {dark[k]:8.0f} {white[k]:8.0f} {ir[k]:8.0f}")
        bad = [k for k in ("f5_555", "f8_680") if white[k] <= dark[k] * 1.5]
        if bad:
            _fail("spectral", f"white LEDs barely raise {bad}: check LED MOSFET / aim / gain")
        elif ir["nir"] <= dark["nir"] * 1.5:
            _fail("spectral", "IR LED doesn't raise NIR: check 850 nm LED wiring")
        else:
            _ok("spectral")
    except Exception as e:  # noqa: BLE001
        _fail("spectral", e)


def gas(cfg):
    from . import gas as g
    try:
        s = g.make_gas(cfg, False)
        vals = []
        for _ in range(5):
            vals.append(s.read())
            time.sleep(1)
        raw = [v["raw"] for v in vals]
        _ok(f"gas ({s.kind})", f"raw {min(raw):.4g}…{max(raw):.4g}  temp {vals[-1]['temp']}  rh {vals[-1]['humidity']}")
        print("        Breathe near it or hold an open alcohol wipe nearby: the reading should move.")
    except Exception as e:  # noqa: BLE001
        _fail("gas", e)


def outputs(cfg):
    from . import gpio
    try:
        io = gpio.make_io(cfg, False)
        for n in ("led_strip", "spec_white", "spec_ir", "fan"):
            print(f"        {n} ON for 2 s…")
            io.set(n, True)
            time.sleep(2)
            io.set(n, False)
        _ok("outputs", "(did each one switch? the IR LED is only visible on a phone camera)")
    except Exception as e:  # noqa: BLE001
        _fail("outputs", e)


def inputs(cfg):
    from . import gpio
    try:
        io = gpio.make_io(cfg, False)
        print(f"        door closed: {io.door_closed()}  — open/close it and tap the pad (10 s)…")
        hits = []
        io.on_touch(lambda: hits.append("touch"))
        last = io.door_closed()
        t0 = time.time()
        while time.time() - t0 < 10:
            d = io.door_closed()
            if d != last:
                hits.append(f"door {'closed' if d else 'open'}")
                last = d
            time.sleep(0.05)
        _ok("inputs", f"events: {hits or 'none (check wiring)'}")
    except Exception as e:  # noqa: BLE001
        _fail("inputs", e)


def camera(cfg):
    from . import camera as c
    try:
        f = c.make_camera(cfg, False).capture()
        msg = f"{f.source} {f.bgr.shape}"
        if f.depth_mm is not None:
            import numpy as np
            v = f.depth_mm[f.depth_mm > 0]
            msg += f"  depth median {np.median(v):.0f} mm (empty tray ≈ {cfg['camera']['tray_depth_mm']})"
        _ok("camera", msg)
    except Exception as e:  # noqa: BLE001
        _fail("camera", e)


def main():
    cfg = config.load()
    which = sys.argv[1:] or ["i2c", "camera", "spectral", "gas", "outputs", "inputs"]
    fns = {"i2c": lambda: i2c(), "camera": lambda: camera(cfg), "spectral": lambda: spectral(cfg),
           "gas": lambda: gas(cfg), "outputs": lambda: outputs(cfg), "inputs": lambda: inputs(cfg)}
    for w in which:
        print(f"[{w}]")
        fns[w]()


if __name__ == "__main__":
    main()
