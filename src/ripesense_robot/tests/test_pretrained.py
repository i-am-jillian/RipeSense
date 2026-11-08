"""How the public-data models are combined before any cabinet scans exist."""
import numpy as np

from ripesense import features
from ripesense.models.infer import Predictor
from ripesense.models.spectral_prior import to_as7341

CFG = {"scan": {"confidence_threshold": 0.6}, "markdown": {}, "base_prices": {"default": 1}}
NAN_FEATS = {k: float("nan") for k in features.ALL_FEATURES}


def pred():
    p = Predictor.__new__(Predictor)
    p.cfg, p.art = CFG, None
    return p


def img(fruit, rot, conf=0.9):
    return {"fruit_type": fruit, "fruit_conf": conf, "rot_prob": rot, "fruit_top": [(fruit, conf)]}


def stage(u, r, o):
    return {"probs": {"unripe": u, "ripe": r, "overripe": o}, "score": r + 2 * o, "fruits": ["banana"]}


def test_banana_stage_model_drives_stage():
    r = pred().predict(NAN_FEATS, "banana", {"image": img("banana", 0.02), "stage": stage(0.9, 0.08, 0.02)})
    assert r["tier"] == "pretrained" and r["stage"] == "unripe"


def test_ripe_and_overripe_overlap_becomes_ripe_to_overripe():
    r = pred().predict(NAN_FEATS, "banana", {"image": img("banana", 0.02), "stage": stage(0.0, 0.5, 0.5)})
    assert r["fused_stage"] == "ripe_to_overripe"


def test_rot_dominates():
    r = pred().predict(NAN_FEATS, "banana", {"image": img("banana", 0.95), "stage": stage(0.1, 0.2, 0.7)})
    assert r["stage"] == "spoiled"


def test_stage_model_not_used_for_other_fruit():
    r = pred().predict(NAN_FEATS, "apple", {"image": img("apple", 0.05), "stage": stage(0.9, 0.05, 0.05)})
    assert r["stage"] == "ripe"            # fresh apple: edible, stage model ignored
    assert any("fresh vs rotten only" in b for b in r["basis"])


def test_rotten_apple():
    r = pred().predict(NAN_FEATS, "apple", {"image": img("apple", 0.9)})
    assert r["stage"] == "spoiled"


def test_as7341_band_integration():
    wl = np.arange(400, 1001, 2.5)
    spec = np.where(wl < 600, 0.2, 0.6)
    ch = to_as7341(wl, spec)
    assert abs(ch["f2_445"] - 0.2) < 1e-3 and abs(ch["f8_680"] - 0.6) < 1e-3 and abs(ch["nir"] - 0.6) < 1e-3
    short = to_as7341(np.arange(408, 902, 2.0), np.ones(247))
    assert np.isnan(short["nir"])        # camera stops at 901 nm: NIR channel not simulated


def test_no_fruit_in_view_is_uncertain():
    r = pred().predict(NAN_FEATS, "unknown", {"image": img("pomegranate", 0.06, conf=0.27)})
    assert r["stage"] == "uncertain" and any("no fruit recognised" in b for b in r["basis"])
