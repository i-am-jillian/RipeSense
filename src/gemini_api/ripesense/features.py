"""Turn raw sensor readings into model features.

Prefixes: c_ colour, i_ image geometry, s_ spectral, g_ gas.  Missing values are NaN,
which the gradient-boosted model handles natively.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from .hardware.spectral import CHANNELS

log = logging.getLogger(__name__)
NAN = float("nan")
FRUITS = ["banana", "avocado", "tomato", "apple", "mango", "other"]

IMAGE_FEATURES = ["c_hue", "c_sat", "c_val", "c_L", "c_a", "c_b", "c_green_frac", "c_yellow_frac",
                  "c_brown_frac", "c_dark_frac", "c_spot_density", "i_area_cm2", "i_volume_cm3",
                  "i_height_mm", "i_length_mm", "i_width_mm", "i_rot_prob"]
SPECTRAL_FEATURES = [f"s_r{c[1:]}" for c in CHANNELS[:8]] + ["s_rclear", "s_rnir", "s_chl", "s_ripen",
                                                              "s_carot", "s_red", "s_calibrated"]
GAS_FEATURES = ["g_rise", "g_slope", "g_rise_per_100cm3", "g_temp", "g_humidity"]
FEATURES = ["fruit_code"] + IMAGE_FEATURES + SPECTRAL_FEATURES + GAS_FEATURES
MODALITIES = {"camera": IMAGE_FEATURES, "spectral": SPECTRAL_FEATURES, "gas": GAS_FEATURES}


def fruit_code(name: str) -> int:
    return FRUITS.index(name) if name in FRUITS else FRUITS.index("other")


# ----------------------------------------------------------------- segmentation
def segment(color: np.ndarray, depth: np.ndarray | None, empty_color=None, empty_depth=None):
    """Mask of the fruit. Depth first (anything 8 mm above the floor), then colour difference."""
    h, w = color.shape[:2]
    mask, method, floor = None, "none", None
    if depth is not None and np.isfinite(depth).mean() > 0.3:
        if empty_depth is not None and empty_depth.shape == depth.shape:
            floor_map = empty_depth
            floor = float(np.nanmedian(empty_depth))
        else:
            floor = float(np.nanpercentile(depth, 90))
            floor_map = np.full_like(depth, floor)
        with np.errstate(invalid="ignore"):
            mask = ((floor_map - depth) > 0.008).astype(np.uint8) * 255
        method = "depth"
    if (mask is None or mask.sum() == 0) and empty_color is not None and empty_color.shape == color.shape:
        diff = cv2.absdiff(cv2.cvtColor(color, cv2.COLOR_BGR2LAB), cv2.cvtColor(empty_color, cv2.COLOR_BGR2LAB))
        mask = (diff.max(axis=2) > 25).astype(np.uint8) * 255
        method = "difference"
    if mask is None or mask.sum() == 0:
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        mask = ((hsv[..., 2] > 70) & (hsv[..., 1] > 50)).astype(np.uint8) * 255
        method = "colour"
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros((h, w), np.uint8), method, floor
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == big).astype(np.uint8) * 255
    if mask.sum() / 255 < 0.003 * h * w:
        return np.zeros((h, w), np.uint8), method, floor
    return mask, method, floor


# ----------------------------------------------------------------- image features
class ImageClassifier:
    """Optional ONNX model from tools/train_image_model.py: labels like 'banana__fresh', 'banana__rotten'."""

    def __init__(self, path: Path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.labels = json.loads(Path(str(path) + ".labels.json").read_text())
        self.size = 224

    def predict(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        img = cv2.resize(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), (self.size, self.size)).astype(np.float32) / 255
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        x = img.transpose(2, 0, 1)[None].astype(np.float32)
        logits = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        fruit_p: dict[str, float] = {}
        rot_p: dict[str, float] = {}
        for lab, pi in zip(self.labels, p):
            fruit, _, cond = lab.partition("__")
            fruit_p[fruit] = fruit_p.get(fruit, 0.0) + float(pi)
            if cond in ("rotten", "bad"):
                rot_p[fruit] = rot_p.get(fruit, 0.0) + float(pi)
        fruit = max(fruit_p, key=fruit_p.get)
        return fruit, rot_p.get(fruit, 0.0) / max(fruit_p[fruit], 1e-6)   # P(rotten | this fruit)


def image_features(frame, mask, floor, classifier: ImageClassifier | None = None) -> tuple[dict, str | None]:
    f = {k: NAN for k in IMAGE_FEATURES}
    sel = mask > 0
    if sel.sum() < 50:
        return f, None
    color = frame.color
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB).astype(np.float32)
    H, S, V = hsv[..., 0][sel].astype(float), hsv[..., 1][sel].astype(float), hsv[..., 2][sel].astype(float)
    ang = H * 2 * math.pi / 180
    f["c_hue"] = (math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) + 360) % 360
    f["c_sat"], f["c_val"] = S.mean() / 255, V.mean() / 255
    f["c_L"] = lab[..., 0][sel].mean() * 100 / 255
    f["c_a"] = lab[..., 1][sel].mean() - 128
    f["c_b"] = lab[..., 2][sel].mean() - 128
    f["c_green_frac"] = float(((H >= 35) & (H <= 85) & (S > 60)).mean())
    f["c_yellow_frac"] = float(((H >= 18) & (H < 35) & (S > 60) & (V > 100)).mean())
    f["c_brown_frac"] = float((((H < 18) | (H > 170)) & (V < 150) & (V >= 60)).mean())
    f["c_dark_frac"] = float((V < 60).mean())
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8))
    dark = ((hsv[..., 2] < 90) & (inner > 0)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    spots = int(((stats[1:, cv2.CC_STAT_AREA] >= 6) & (stats[1:, cv2.CC_STAT_AREA] <= 2000)).sum()) if n > 1 else 0
    f["c_spot_density"] = spots / (sel.sum() / 10000)

    if frame.depth is not None and frame.fx and floor:
        z = np.where(sel, frame.depth, np.nan)
        px_area = (z / frame.fx) * (z / frame.fy)                       # m^2 per pixel at its own depth
        height = np.clip(floor - z, 0, None)
        f["i_area_cm2"] = float(np.nansum(px_area) * 1e4)
        f["i_volume_cm3"] = float(np.nansum(height * px_area) * 1e6)
        f["i_height_mm"] = float(np.nanpercentile(height[sel], 98) * 1000)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, _), (a, b), _ = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        scale = float(np.nanmedian(z[sel])) / frame.fx * 1000             # mm per pixel
        f["i_length_mm"], f["i_width_mm"] = max(a, b) * scale, min(a, b) * scale

    fruit = None
    if classifier is not None:
        x, y, w, h = cv2.boundingRect(mask)
        try:
            fruit, f["i_rot_prob"] = classifier.predict(color[y:y + h, x:x + w])
        except Exception as e:  # pragma: no cover
            log.warning("image classifier failed: %s", e)
    return f, fruit


# ----------------------------------------------------------------- spectral features
def spectral_features(meas: dict, calib: dict | None) -> dict:
    f = {k: NAN for k in SPECTRAL_FEATURES}
    dark, white, ir = meas["dark"], meas["white"], meas["ir"]
    if calib:
        cd, cw, ci = calib["dark"], calib["white"], calib["ir"]
        R = {c: (white[c] - dark[c]) / max(cw[c] - cd[c], 1.0) for c in CHANNELS}
        R["nir"] = (ir["nir"] - dark["nir"]) / max(ci["nir"] - cd["nir"], 1.0)
        f["s_calibrated"] = 1.0
    else:  # shape only: each channel as a share of the total
        tot = sum(max(white[c] - dark[c], 0) for c in CHANNELS[:8]) or 1.0
        R = {c: max(white[c] - dark[c], 0) / tot for c in CHANNELS}
        R["nir"] = max(ir["nir"] - dark["nir"], 0) / max(ir["clear"] - dark["clear"], 1.0)
        f["s_calibrated"] = 0.0
    for c in CHANNELS[:8]:
        f[f"s_r{c[1:]}"] = R[c]
    f["s_rclear"], f["s_rnir"] = R["clear"], R["nir"]
    eps = 1e-6
    f["s_chl"] = R["f680"] / (R["nir"] + eps)
    f["s_ripen"] = (R["f555"] - R["f680"]) / (R["f555"] + R["f680"] + eps)
    f["s_carot"] = R["f480"] / (R["f555"] + eps)
    f["s_red"] = R["f630"] / (R["f515"] + eps)
    return f


# ----------------------------------------------------------------- gas features
def gas_features(samples, baseline: float | None, monitor, volume_cm3: float) -> dict:
    f = {k: NAN for k in GAS_FEATURES}
    if baseline is None or len(samples) < 3:
        return f
    t = np.array([s.t for s in samples])
    r = np.array([monitor.response(s.raw, baseline) for s in samples])
    ok = np.isfinite(r)
    if ok.sum() < 3:
        return f
    t, r = t[ok], r[ok]
    f["g_rise"] = float(np.median(r[-5:]) - 1.0)
    f["g_slope"] = float(np.polyfit((t - t[0]) / 60.0, r, 1)[0]) if t[-1] > t[0] else NAN
    if np.isfinite(volume_cm3) and volume_cm3 > 0:
        f["g_rise_per_100cm3"] = f["g_rise"] / max(volume_cm3, 10.0) * 100
    last = samples[-1]
    f["g_temp"] = last.temp if last.temp is not None else NAN
    f["g_humidity"] = last.humidity if last.humidity is not None else NAN
    return f


def vector(features: dict) -> np.ndarray:
    return np.array([features.get(k, NAN) for k in FEATURES], dtype=float)
