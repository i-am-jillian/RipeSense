"""Per-sensor feature extraction (doc p.11-12).

Every feature function tolerates a missing sensor and returns NaNs, so a scan
still works if the gas sensor is cold or the spectral read fails; LightGBM
handles NaN natively and the model is trained with modality dropout.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .sim.fruit import SPECTRAL_CHANNELS

NAN = float("nan")

CAMERA_FEATURES = ["length_mm", "width_mm", "area_cm2", "volume_cm3",
                   "hue_mean", "sat_mean", "val_mean", "lab_l", "lab_a", "lab_b",
                   "brown_pct", "black_pct", "green_pct", "spot_count",
                   "img_rot_prob",      # image_model: P(rotten | fruit), public fruit datasets
                   "img_stage_score",   # stage_model: 0 unripe … 2 overripe, public banana dataset
                   "ir_bruise_pct", "ir_bruise_count", "ir_bruise_area_cm2"]  # 850 nm IR: hidden bruises
SPECTRAL_FEATURES = [f"r_{c}" for c in SPECTRAL_CHANNELS] + [
    "chl_index", "ripen_index", "carot_ratio", "red_ratio", "ndvi",
    "spec_prior_score"]  # spectral_prior: 0 unripe … 2 overripe, pre-trained on DeepHS
# BME688 heater-profile "e-nose": response at each hotplate temperature, plus the
# hot/cold ratio (different VOCs peak at different temperatures).
HEATER_STEPS = [200, 250, 300, 350, 400]
ENOSE_FEATURES = [f"enose_r{t}" for t in HEATER_STEPS] + ["enose_shape"]
GAS_FEATURES = ["gas_rise", "gas_slope_per_min", "gas_rise_per_cm3", "temp_c", "humidity_pct"] + ENOSE_FEATURES
ALL_FEATURES = CAMERA_FEATURES + SPECTRAL_FEATURES + GAS_FEATURES
MODALITIES = {"camera": CAMERA_FEATURES, "spectral": SPECTRAL_FEATURES, "gas": GAS_FEATURES}


# ---------------------------------------------------------------- camera
def fruit_mask(bgr: np.ndarray, depth_mm: np.ndarray | None, tray_depth_mm: float,
               min_height_mm: float) -> np.ndarray:
    """Fruit = everything standing above the tray plane. Without depth, fall back
    to a saturation/brightness threshold against the black chamber."""
    if depth_mm is not None:
        d = depth_mm.astype(np.float32)
        m = ((d > 0) & (d < tray_depth_mm - min_height_mm)).astype(np.uint8) * 255
    else:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = (((hsv[..., 1] > 60) | (hsv[..., 2] > 90))).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    # Keep the largest blob only.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros_like(m)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == i).astype(np.uint8) * 255


def camera_features(bgr, depth_mm, fx, fy, tray_depth_mm=188.0, min_height_mm=8.0,
                    img_rot_prob: float | None = None) -> tuple[dict, np.ndarray]:
    f = {k: NAN for k in CAMERA_FEATURES}
    if bgr is None:
        return f, None
    mask = fruit_mask(bgr, depth_mm, tray_depth_mm, min_height_mm)
    sel = mask > 0
    if sel.sum() < 200:
        return f, mask

    # Geometry. mm per pixel at the fruit's mean top surface distance.
    if depth_mm is not None:
        z = depth_mm.astype(np.float32)
        z_mean = float(np.median(z[sel]))
        h = np.clip(tray_depth_mm - z, 0, None)
        px_area_mm2 = (z / fx) * (z / fy)
        f["volume_cm3"] = float((h[sel] * px_area_mm2[sel]).sum() / 1000.0)
    else:
        z_mean = tray_depth_mm - 30.0
    mm_per_px = z_mean / fx
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, _), (w, hgt), _ = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    f["length_mm"] = float(max(w, hgt) * mm_per_px)
    f["width_mm"] = float(min(w, hgt) * mm_per_px)
    f["area_cm2"] = float(sel.sum() * mm_per_px ** 2 / 100.0)

    # Colour.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    H = hsv[..., 0][sel].astype(np.float32)
    S = hsv[..., 1][sel].astype(np.float32)
    V = hsv[..., 2][sel].astype(np.float32)
    ang = H * (2 * math.pi / 180.0)  # hue is circular (red wraps 179→0)
    f["hue_mean"] = float((math.atan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * math.pi)) * 180 / (2 * math.pi))
    f["sat_mean"] = float(S.mean())
    f["val_mean"] = float(V.mean())
    f["lab_l"] = float(lab[..., 0][sel].mean() * 100 / 255)
    f["lab_a"] = float(lab[..., 1][sel].mean() - 128)
    f["lab_b"] = float(lab[..., 2][sel].mean() - 128)
    black = V < 60
    brown = (~black) & (V < 130) & ((H < 25) | (H > 165))
    green = (H >= 33) & (H <= 85) & (S > 60) & (V > 60)
    f["black_pct"] = float(black.mean() * 100)
    f["brown_pct"] = float(brown.mean() * 100)
    f["green_pct"] = float(green.mean() * 100)

    # Spots: dark blobs inside the (eroded) fruit mask.
    inner = cv2.erode(mask, np.ones((7, 7), np.uint8))
    dark = ((hsv[..., 2] < 110) & (inner > 0)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    f["spot_count"] = float(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 6))
    if img_rot_prob is not None:
        f["img_rot_prob"] = float(img_rot_prob)
    return f, mask


# ------------------------------------------------------------- IR bruises
def ir_bruises(ir: np.ndarray | None, depth_mm: np.ndarray | None = None, fx: float = 380.0,
               tray_depth_mm: float = 188.0, min_height_mm: float = 8.0) -> tuple[dict, np.ndarray | None]:
    """Hidden bruises in the RealSense IR image (850 nm, projector off, IR LEDs on).

    Bruised tissue has broken cells and water-soaked spaces that absorb near-IR,
    so a fresh bruise shows as a dark patch at 850 nm hours before the skin
    browns in visible light. The fruit is flat-fielded (divided by a heavy blur
    of itself, inside the mask) so the curved surface's shading cancels, then
    local dark patches are bruise candidates.
    Returns (features, overlay BGR image or None)."""
    f = {"ir_bruise_pct": NAN, "ir_bruise_count": NAN, "ir_bruise_area_cm2": NAN}
    if ir is None:
        return f, None
    g = ir.astype(np.float32)
    if g.ndim == 3:
        g = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if g.max() > 255:
        g = g / (g.max() / 255.0)
    if depth_mm is not None and depth_mm.shape == g.shape:
        d = depth_mm.astype(np.float32)
        m = ((d > 0) & (d < tray_depth_mm - min_height_mm)).astype(np.uint8)
    else:  # IR LEDs light the fruit; the matte-black chamber stays dark
        u8 = np.clip(g, 0, 255).astype(np.uint8)
        _, m = cv2.threshold(u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return f, None
    m = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    if m.sum() < 300:
        return f, None
    inner = cv2.erode(m, np.ones((9, 9), np.uint8))
    sig = max(6.0, 0.08 * np.sqrt(m.sum()))
    num = cv2.GaussianBlur(g * m, (0, 0), sig)
    den = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sig)
    flat = num / np.maximum(den, 1e-3)
    ratio = np.where(m > 0, g / np.maximum(flat, 1.0), 1.0)
    br = ((ratio < 0.84) & (inner > 0)).astype(np.uint8)
    br = cv2.morphologyEx(br, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(br)
    min_px = max(25, int(0.002 * m.sum()))
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]
    bmask = np.isin(lab, keep)
    z = float(np.median(depth_mm[m > 0])) if depth_mm is not None and depth_mm.shape == g.shape else tray_depth_mm - 30
    mm_per_px = z / fx
    f["ir_bruise_count"] = float(len(keep))
    f["ir_bruise_pct"] = float(bmask.sum() / max(1, m.sum()) * 100)
    f["ir_bruise_area_cm2"] = float(bmask.sum() * mm_per_px ** 2 / 100.0)
    vis = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    tint = vis.copy()
    tint[bmask] = (0.45 * tint[bmask] + 0.55 * np.array([40, 40, 255])).astype(np.uint8)
    cs, _ = cv2.findContours(bmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tint, cs, -1, (60, 60, 255), 2)
    oc, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tint, oc, -1, (200, 200, 200), 1)
    return f, tint


# --------------------------------------------------------------- spectral
def spectral_features(refl: dict | None) -> dict:
    f = {k: NAN for k in SPECTRAL_FEATURES}
    if not refl:
        return f
    for c in SPECTRAL_CHANNELS:
        f[f"r_{c}"] = float(refl.get(c, NAN))

    def ratio(a, b):
        va, vb = refl.get(a, NAN), refl.get(b, NAN)
        return float(va / vb) if vb and not math.isnan(vb) and abs(vb) > 1e-6 else NAN

    f["chl_index"] = ratio("f8_680", "nir")        # rises as chlorophyll fades
    f["ripen_index"] = ratio("f5_555", "f8_680")   # green vs red absorption
    f["carot_ratio"] = ratio("f3_480", "f5_555")   # carotenoid absorption at 480
    f["red_ratio"] = ratio("f7_630", "f4_515")
    nir, red = refl.get("nir", NAN), refl.get("f8_680", NAN)
    f["ndvi"] = float((nir - red) / (nir + red)) if (nir + red) and not math.isnan(nir + red) else NAN
    return f


# -------------------------------------------------------------------- gas
def enose_features(profile: dict | None) -> dict:
    """profile = {"baseline": [R per heater step], "end": [R per step]} (BME688
    gas resistance). Response per step = (R0 - R) / R0."""
    f = {k: NAN for k in ENOSE_FEATURES}
    if not profile or not profile.get("baseline") or not profile.get("end"):
        return f
    b, e = profile["baseline"], profile["end"]
    if len(b) != len(HEATER_STEPS) or len(e) != len(HEATER_STEPS):
        return f
    resp = [(bi - ei) / bi if bi else NAN for bi, ei in zip(b, e)]
    for t, r in zip(HEATER_STEPS, resp):
        f[f"enose_r{t}"] = float(r)
    lo, hi = resp[0], resp[-1]
    if lo == lo and hi == hi and abs(lo) > 1e-4:
        f["enose_shape"] = float(hi / lo)
    return f


def gas_features(series: list[tuple[float, float]] | None, volume_cm3: float = NAN,
                 temp: float | None = None, humidity: float | None = None,
                 profile: dict | None = None) -> dict:
    """series = [(seconds_since_start, response), ...] where response is the
    fractional rise above the empty-chamber baseline (see hw.gas.response)."""
    f = {k: NAN for k in GAS_FEATURES}
    f.update(enose_features(profile))
    if temp is not None:
        f["temp_c"] = float(temp)
    if humidity is not None:
        f["humidity_pct"] = float(humidity)
    if not series or len(series) < 5:
        return f
    t = np.array([s[0] for s in series], np.float64)
    r = np.array([s[1] for s in series], np.float64)
    ok = ~np.isnan(r)
    t, r = t[ok], r[ok]
    if len(t) < 5:
        return f
    tail = r[t >= t.max() - 10]
    f["gas_rise"] = float(np.median(tail))
    # Slope over the last 60 s, per minute.
    w = t >= t.max() - 60
    if w.sum() >= 5:
        f["gas_slope_per_min"] = float(np.polyfit(t[w], r[w], 1)[0] * 60)
    if volume_cm3 and not math.isnan(volume_cm3) and volume_cm3 > 1:
        f["gas_rise_per_cm3"] = float(f["gas_rise"] / volume_cm3 * 100)  # per 100 cm³
    return f


def vector(feats: dict) -> list[float]:
    return [float(feats.get(k, NAN)) for k in ALL_FEATURES]
