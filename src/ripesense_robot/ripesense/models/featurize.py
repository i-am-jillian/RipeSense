"""Turn a stored scan row (CSV/SQLite) into the feature dict. Training and live
inference both go through this, so features can never drift between them.

Returns (features, aux): aux carries the raw outputs of the optional
pre-trained models (fruit type, stage probabilities, spectral prior) that the
predictor shows on screen and uses in camera-first mode."""
from __future__ import annotations

import json
import math

import cv2

from .. import config, features
from ..sim.fruit import SPECTRAL_CHANNELS


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _json(v):
    if not v:
        return None
    try:
        return json.loads(v) if isinstance(v, str) else v
    except json.JSONDecodeError:
        return None


def row_features(row: dict, cfg: dict, models=None, frame=None) -> tuple[dict, dict]:
    """models: image.Models (or None). frame=(bgr, depth, fx, fy[, ir, ir_depth, ir_fx])
    skips reloading images during a live scan."""
    cam = cfg["camera"]
    ir = ir_depth = None
    ir_fx = 380.0
    if frame is None:
        bgr = depth = None
        dd = config.data_dir()
        if row.get("rgb_path"):
            bgr = cv2.imread(str(dd / row["rgb_path"]))
        if row.get("depth_path"):
            depth = cv2.imread(str(dd / row["depth_path"]), cv2.IMREAD_UNCHANGED)
        if row.get("ir_path"):
            ir = cv2.imread(str(dd / row["ir_path"]), cv2.IMREAD_GRAYSCALE)
        if row.get("ir_depth_path"):
            ir_depth = cv2.imread(str(dd / row["ir_depth_path"]), cv2.IMREAD_UNCHANGED)
        fx = fy = 615.0
    else:
        bgr, depth, fx, fy = frame[:4]
        if len(frame) > 4:
            ir, ir_depth, ir_fx = frame[4:7]

    aux: dict = {}
    cf, mask = features.camera_features(bgr, depth, fx, fy, cam["tray_depth_mm"], cam["mask_min_height_mm"])
    aux["mask"] = mask
    if ir is not None:
        irf, aux["ir_overlay"] = features.ir_bruises(ir, ir_depth, ir_fx, cam["tray_depth_mm"], cam["mask_min_height_mm"])
        cf.update(irf)
    if models is not None and bgr is not None:
        if models.image is not None:
            aux["image"], emb = models.image.predict(bgr, mask, with_embedding=True)
            cf["img_rot_prob"] = aux["image"]["rot_prob"]
            if emb is not None and models.fewshot is not None:
                aux["embedding"] = emb
                aux["fewshot"] = models.fewshot.match(emb)
        if models.stage is not None:
            aux["stage"] = models.stage.predict(bgr, mask)
            cf["img_stage_score"] = aux["stage"]["score"]

    refl = {c: _num(row.get(f"spec_{c}")) for c in SPECTRAL_CHANNELS}
    has_refl = not all(math.isnan(v) for v in refl.values())
    sf = features.spectral_features(refl if has_refl else None)
    fruit = row.get("fruit_type") or (aux.get("image") or {}).get("fruit_type", "")
    if models is not None and models.spec_prior is not None and has_refl:
        aux["spec_prior"] = models.spec_prior.predict(refl, fruit)
        if aux["spec_prior"]:
            sf["spec_prior_score"] = aux["spec_prior"]["score"]

    series = _json(row.get("gas_series"))
    gf = features.gas_features(series, cf.get("volume_cm3", float("nan")),
                               _num(row.get("temp")) if row.get("temp") not in (None, "") else None,
                               _num(row.get("humidity")) if row.get("humidity") not in (None, "") else None,
                               profile=_json(row.get("gas_profile")))
    return {**cf, **sf, **gf}, aux
