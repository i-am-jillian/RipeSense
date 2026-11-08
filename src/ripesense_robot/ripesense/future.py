"""'Your banana in 3 days': a simulated preview of how the fruit will look.

Not a generative model: the fruit pixels are moved along the colour path the
fruit type follows as it ripens (hue shift for green→yellow, darkening and
browning later), and brown spots are grown in proportion to the forecast
ripeness change. Labelled 'simulated' on screen. The forecast decides how far
to move; this only draws it.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import features

# OpenCV hue (0-180) at ripeness score 0 (unripe) / 1 (ripe) / 2 (overripe).
HUE_PATH = {"banana": (40, 25, 19), "mango": (40, 22, 16), "tomato": (40, 4, 2),
            "avocado": (45, 40, 20), "pear": (35, 28, 20), "papaya": (45, 20, 14)}
SPOT_START = {"banana": 1.1}          # spots start appearing past this score


def fruit_mask_any(bgr: np.ndarray) -> np.ndarray:
    """Fruit mask without depth: saturation threshold, else GrabCut from a centre box."""
    m = features.fruit_mask(bgr, None, 188.0, 8.0)
    frac = (m > 0).mean()
    if 0.03 < frac < 0.85:
        return m
    h, w = bgr.shape[:2]
    gc = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.08), int(h * 0.08), int(w * 0.84), int(h * 0.84))
    try:
        cv2.grabCut(bgr, gc, rect, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64), 3,
                    cv2.GC_INIT_WITH_RECT)
        m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        m = np.full((h, w), 255, np.uint8)
    return m


def _hue_at(path, q):
    q = min(2.0, max(0.0, q))
    a, b, c = path
    return a + (b - a) * q if q <= 1 else b + (c - b) * (q - 1)


def preview(bgr: np.ndarray, fruit: str, q_now: float, q_future: float, seed: int = 0,
            mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None or not mask.any():
        mask = fruit_mask_any(bgr)
    sel = mask > 0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    dq = q_future - q_now
    if dq <= 0.01:
        return bgr.copy()
    # 1 · Colour: shift hue along the fruit's ripening path.
    path = HUE_PATH.get(fruit)
    if path:
        dh = _hue_at(path, q_future) - _hue_at(path, q_now)
        hsv[..., 0] = np.where(sel, (hsv[..., 0] + dh) % 180, hsv[..., 0])
        # Chlorophyll loss: yellow skin is brighter and more saturated than green.
        up = min(q_future, 1.0) - min(q_now, 1.0)
        if up > 0:
            hsv[..., 2] = np.where(sel, hsv[..., 2] * (1 + 0.18 * up), hsv[..., 2])
            hsv[..., 1] = np.where(sel, hsv[..., 1] * (1 + 0.12 * up), hsv[..., 1])
    # 2 · Past ripe: darker, duller skin (all fruit).
    late = max(0.0, q_future - max(q_now, 1.0))
    k_v = 1 - 0.22 * min(late, 1.5)
    k_s = 1 - 0.18 * min(late, 1.5)
    hsv[..., 2] = np.where(sel, hsv[..., 2] * k_v, hsv[..., 2])
    hsv[..., 1] = np.where(sel, hsv[..., 1] * k_s, hsv[..., 1])
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    # 3 · Spots: count grows with ripeness past the spot onset; rot patches near spoiled.
    start = SPOT_START.get(fruit, 1.6)
    n_new = int(max(0.0, q_future - start) * 60) - int(max(0.0, q_now - start) * 60)
    inner = np.argwhere(cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0)
    if len(inner) and n_new > 0:
        rng = np.random.default_rng(seed)
        size = max(mask.shape) / 640.0
        alpha = np.zeros(mask.shape, np.float32)
        for _ in range(n_new):
            y, x = inner[rng.integers(len(inner))]
            r = int(max(2, rng.uniform(2, 3 + 6 * min(q_future, 2.5)) * size))
            cv2.ellipse(alpha, (int(x), int(y)), (r, int(r * rng.uniform(0.5, 1))), float(rng.uniform(0, 180)),
                        0, 360, float(rng.uniform(0.6, 0.95)), -1)
        if q_future > 2.4:
            for _ in range(3):
                y, x = inner[rng.integers(len(inner))]
                cv2.circle(alpha, (int(x), int(y)), int(rng.uniform(18, 34) * size), 0.9, -1)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.5 * size + 0.5) * sel
        brown = np.array([25, 40, 60], np.float32)
        out = out * (1 - alpha[..., None]) + brown * alpha[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return np.where(sel[..., None], out, bgr)   # background untouched (no HSV round-trip noise)
