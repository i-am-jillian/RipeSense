"""A simulated fruit whose camera image, spectrum and gas output all follow one
ripeness value, so the mock pipeline produces coherent, learnable data.

This is for development only. None of these numbers are measurements: real
training must use real scans.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..stages import STAGES, stage_from_days

SPECTRAL_CHANNELS = ["f1_415", "f2_445", "f3_480", "f4_515", "f5_555",
                     "f6_590", "f7_630", "f8_680", "clear", "nir"]
WAVELENGTHS = [415, 445, 480, 515, 555, 590, 630, 680]

# Rough hue (OpenCV 0-180) path from green → yellow → brown per fruit.
FRUIT_SHAPES = {
    # length_mm, width_mm, height_mm, (hue_unripe, hue_ripe, hue_over)
    "banana": (180, 38, 36, (42, 26, 15)),
    "avocado": (105, 70, 65, (45, 40, 10)),
    "tomato": (70, 70, 60, (50, 5, 2)),
    "apple": (78, 78, 72, (40, 8, 12)),
}


@dataclass
class SimFruit:
    fruit_type: str = "banana"
    days_left: float = 3.0           # ground truth; ≤0 overripe, < -1.5 spoiled
    fruit_id: str = "sim-0"
    rotation: float = 0.0            # degrees in the camera image
    hidden_bruises: int = 0          # bruises visible at 850 nm but not yet in colour
    seed: int = field(default_factory=lambda: random.randrange(1 << 30))

    @property
    def stage(self) -> str:
        if self.days_left < -1.5:
            return "spoiled"
        return stage_from_days(round(self.days_left))

    @property
    def ripeness(self) -> float:
        """0 = fully green, 1 = spoiled. Monotone in days_left."""
        return float(np.clip((7.0 - self.days_left) / 9.5, 0.0, 1.0))

    # ------------------------------------------------------------------ camera
    def render(self, width=640, height=480, tray_depth_mm=188.0, fx=615.0):
        """Return (bgr uint8 HxWx3, depth uint16 mm HxW) as a RealSense would."""
        rng = np.random.default_rng(self.seed)
        L, W_, Hh, hues = FRUIT_SHAPES.get(self.fruit_type, FRUIT_SHAPES["banana"])
        r = self.ripeness
        # Hue: green→yellow over r∈[0,.5], yellow→brown over r∈[.5,1]
        if r < 0.5:
            hue = hues[0] + (hues[1] - hues[0]) * (r / 0.5)
        else:
            hue = hues[1] + (hues[2] - hues[1]) * ((r - 0.5) / 0.5)
        sat = 200 - 60 * max(0.0, r - 0.6) / 0.4
        val = 210 - 150 * max(0.0, r - 0.55) / 0.45

        # Background: dark tray plus cradle.
        bgr = np.full((height, width, 3), (35, 35, 38), np.uint8)
        depth = np.full((height, width), int(tray_depth_mm), np.uint16)
        cx, cy = width // 2, height // 2
        top_z = tray_depth_mm - Hh  # distance camera→top of fruit
        ax = int(L / 2 * fx / top_z)
        ay = int(W_ / 2 * fx / top_z)
        mask = np.zeros((height, width), np.uint8)
        cv2.ellipse(mask, (cx, cy), (ax, ay), self.rotation, 0, 360, 255, -1)

        hsv = np.zeros((height, width, 3), np.uint8)
        hsv[..., 0] = np.uint8(np.clip(hue + rng.normal(0, 1.5, (height, width)), 0, 179))
        hsv[..., 1] = np.uint8(np.clip(sat + rng.normal(0, 8, (height, width)), 0, 255))
        hsv[..., 2] = np.uint8(np.clip(val + rng.normal(0, 8, (height, width)), 0, 255))
        fruit = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Brown/black spots grow with ripeness.
        n_spots = int(max(0.0, r - 0.45) * 90)
        ys, xs = np.nonzero(mask)
        for _ in range(n_spots):
            i = rng.integers(len(xs))
            rad = int(rng.integers(2, 4 + int(10 * r)))
            cv2.circle(fruit, (int(xs[i]), int(ys[i])), rad, (20, 35, 50), -1)
        if self.stage == "spoiled":
            for _ in range(3):
                i = rng.integers(len(xs))
                cv2.circle(fruit, (int(xs[i]), int(ys[i])), int(rng.integers(15, 30)), (15, 20, 25), -1)

        bgr[mask > 0] = fruit[mask > 0]
        # Depth: half-ellipsoid height map.
        yy, xx = np.mgrid[0:height, 0:width]
        th = math.radians(self.rotation)
        dx, dy = xx - cx, yy - cy
        u = (dx * math.cos(th) + dy * math.sin(th)) / max(ax, 1)
        v = (-dx * math.sin(th) + dy * math.cos(th)) / max(ay, 1)
        rr = np.clip(1 - u ** 2 - v ** 2, 0, 1)
        h = Hh * np.sqrt(rr)
        d = tray_depth_mm - h + rng.normal(0, 0.8, (height, width))
        depth = np.where(mask > 0, d, depth).astype(np.uint16)
        return bgr, depth

    def render_ir(self, width=424, height=240, tray_depth_mm=188.0, fx=380.0):
        """IR camera view (850 nm, projector off, IR LEDs on) + its native depth.
        Skin is fairly uniform in NIR; hidden bruises are dark, soft-edged patches."""
        rng = np.random.default_rng(self.seed + 7)
        L, W_, Hh, _ = FRUIT_SHAPES.get(self.fruit_type, FRUIT_SHAPES["banana"])
        top_z = tray_depth_mm - Hh
        ax, ay = int(L / 2 * fx / top_z), int(W_ / 2 * fx / top_z)
        cx, cy = width // 2, height // 2
        mask = np.zeros((height, width), np.uint8)
        cv2.ellipse(mask, (cx, cy), (ax, ay), self.rotation, 0, 360, 255, -1)
        yy, xx = np.mgrid[0:height, 0:width]
        th = math.radians(self.rotation)
        dx, dy = xx - cx, yy - cy
        u = (dx * math.cos(th) + dy * math.sin(th)) / max(ax, 1)
        v = (-dx * math.sin(th) + dy * math.cos(th)) / max(ay, 1)
        rr = np.clip(1 - u ** 2 - v ** 2, 0, 1)
        shade = 0.55 + 0.45 * np.sqrt(rr)                     # curved surface falls off
        brown = max(0.0, (self.ripeness - 0.55) / 0.45)
        ir = (175 - 40 * brown) * shade + rng.normal(0, 3, (height, width))
        inner = np.argwhere(cv2.erode(mask, np.ones((15, 15), np.uint8)) > 0)
        for _ in range(self.hidden_bruises):
            y0, x0 = inner[rng.integers(len(inner))]
            rad = int(rng.integers(7, 13))
            spot = np.zeros((height, width), np.float32)
            cv2.circle(spot, (int(x0), int(y0)), rad, 1.0, -1)
            ir *= 1 - 0.32 * cv2.GaussianBlur(spot, (0, 0), rad / 2.5)
        ir = np.where(mask > 0, ir, 12 + rng.normal(0, 2, (height, width)))
        d = np.where(mask > 0, tray_depth_mm - Hh * np.sqrt(rr), tray_depth_mm)
        return np.clip(ir, 0, 255).astype(np.uint8), d.astype(np.uint16)

    # ---------------------------------------------------------------- spectral
    def reflectance(self) -> dict:
        """True diffuse reflectance per AS7341 channel (0-1)."""
        r = self.ripeness
        chl = max(0.0, 1.0 - r / 0.55)           # chlorophyll fades by "ripe"
        car = min(1.0, r / 0.45)                  # carotenoids show as chl fades
        brown = max(0.0, (r - 0.55) / 0.45)       # browning darkens everything
        out = {}
        for ch, wl in zip(SPECTRAL_CHANNELS[:8], WAVELENGTHS):
            base = 0.25 + 0.5 * (1 / (1 + math.exp(-(wl - 500) / 25)))  # yellowish skin
            base -= 0.35 * chl * math.exp(-((wl - 680) / 18) ** 2)       # chl red dip
            base -= 0.15 * chl * math.exp(-((wl - 445) / 30) ** 2)
            base += 0.10 * chl * math.exp(-((wl - 550) / 30) ** 2)       # green peak
            base -= 0.20 * car * math.exp(-((wl - 480) / 25) ** 2)       # carotenoid dip
            base *= (1 - 0.65 * brown)
            out[ch] = max(0.01, base)
        out["clear"] = float(np.mean([out[c] for c in SPECTRAL_CHANNELS[:8]]))
        out["nir"] = max(0.05, 0.75 - 0.35 * brown)
        return out

    # --------------------------------------------------------------------- gas
    def enose_weights(self) -> list[float]:
        """Relative response per heater step (200…400 °C). Ripening esters/aldehydes
        respond best on a cooler hotplate; ethanol from fermentation (overripe/
        spoiling) on a hotter one, so the profile's shape tracks ripeness."""
        r = self.ripeness
        ester = math.exp(-((r - 0.45) / 0.25) ** 2)
        ethanol = max(0.0, r - 0.55) / 0.45
        steps = [200, 250, 300, 350, 400]
        return [0.4 + ester * math.exp(-((t - 250) / 80) ** 2) + 1.6 * ethanol * (t - 200) / 200 for t in steps]

    def gas_response(self, t_s: float) -> float:
        """Fractional sensor response above baseline after t seconds sealed."""
        r = self.ripeness
        # Ethanol/esters rise sharply once overripe (cf. e-nose paper, days 2-3 of decay)
        emission = 0.02 + 0.10 * r + 0.9 * max(0.0, r - 0.6) ** 1.6
        vol_scale = {"banana": 1.0, "avocado": 0.9, "tomato": 0.6, "apple": 0.7}.get(self.fruit_type, 1.0)
        return emission * vol_scale * (1 - math.exp(-t_s / 90.0))


def random_fruit(fruit_type: str | None = None, stage: str | None = None, fruit_id: str = "sim") -> SimFruit:
    ft = fruit_type or random.choice(list(FRUIT_SHAPES))
    ranges = {"unripe": (5, 7.5), "ripe": (3, 4.4), "ripe_to_overripe": (1, 2.4),
              "overripe": (-1.4, 0.4), "spoiled": (-2.5, -1.6)}
    st = stage or random.choice(STAGES)
    lo, hi = ranges[st]
    d = random.uniform(lo, hi)
    return SimFruit(fruit_type=ft, days_left=d, fruit_id=fruit_id, rotation=random.uniform(-40, 40),
                    hidden_bruises=random.choice([0, 0, 1, 2]) if d < 4.5 else 0)


# The fruit currently "inside" the mock cabinet. The /dev page can change it.
current: SimFruit = random_fruit("banana", "ripe", "sim-demo")
