import math

import pytest

from ripesense import features, stages
from ripesense.hw import gas, spectral
from ripesense.models.image import parse_class
from ripesense.sim.fruit import SimFruit


def test_reflectance_formula():
    s = {c: 540.0 for c in spectral.SPECTRAL_CHANNELS}
    d = {c: 40.0 for c in spectral.SPECTRAL_CHANNELS}
    w = {c: 1040.0 for c in spectral.SPECTRAL_CHANNELS}
    r = spectral.reflectance(s, d, w)
    assert all(abs(v - 0.5) < 1e-9 for v in r.values())


def test_reflectance_uses_white_dark():
    s = {c: 550.0 for c in spectral.SPECTRAL_CHANNELS}
    d_now = {c: 50.0 for c in spectral.SPECTRAL_CHANNELS}
    w = {c: 1040.0 for c in spectral.SPECTRAL_CHANNELS}
    w_dark = {c: 40.0 for c in spectral.SPECTRAL_CHANNELS}
    r = spectral.reflectance(s, d_now, w, w_dark)
    assert abs(r["f1_415"] - 0.5) < 1e-9


def test_gas_response_sign():
    assert gas.response(1.1, 1.0, "mq") == pytest.approx(0.1)
    assert gas.response(90000, 100000, "bme680") == pytest.approx(0.1)


def test_camera_features_size_and_colour():
    f = SimFruit("banana", days_left=6.5, rotation=0)
    bgr, depth = f.render()
    feats, mask = features.camera_features(bgr, depth, 615.0, 615.0, 188.0, 8.0)
    assert 140 < feats["length_mm"] < 200          # sim banana is 180 mm
    assert feats["volume_cm3"] > 50
    assert feats["green_pct"] > 50                 # unripe = green


def test_brown_increases_with_ripeness():
    g = features.camera_features(*SimFruit("banana", 6.0, seed=1).render(), 615, 615)[0]
    b = features.camera_features(*SimFruit("banana", -1.0, seed=1).render(), 615, 615)[0]
    assert b["brown_pct"] + b["black_pct"] > g["brown_pct"] + g["black_pct"]
    assert b["spot_count"] > g["spot_count"]


def test_missing_sensors_give_nan():
    assert all(math.isnan(v) for v in features.spectral_features(None).values())
    assert all(math.isnan(v) for k, v in features.gas_features(None).items())


def test_gas_features_rise_and_slope():
    series = [(t, 0.001 * t) for t in range(121)]
    f = features.gas_features(series, volume_cm3=100.0)
    assert f["gas_rise"] == pytest.approx(0.115, abs=0.01)
    assert f["gas_slope_per_min"] == pytest.approx(0.06, abs=1e-6)
    assert f["gas_rise_per_cm3"] == pytest.approx(f["gas_rise"])


@pytest.mark.parametrize("name,fruit,rot", [
    ("rotten_banana", "banana", True), ("Fresh Banana", "banana", False),
    ("freshtomatoes", "tomato", False), ("Apple_Bad", "apple", True), ("orange", "orange", None)])
def test_parse_class(name, fruit, rot):
    assert parse_class(name) == (fruit, rot)


def test_stage_from_days_matches_doc_table():
    assert stages.stage_from_days(6) == "unripe"
    assert stages.stage_from_days(3) == "ripe"
    assert stages.stage_from_days(1) == "ripe_to_overripe"
    assert stages.stage_from_days(0) == "overripe"


def test_store_markdown_price():
    cfg = {"markdown": {"overripe": {"store_action": "Bake", "markdown_pct": [50, 70]}},
           "base_prices": {"banana": 1.0, "default": 1.0}}
    info = stages.store_info("overripe", "banana", cfg)
    assert info["price"] == pytest.approx(0.4)


def test_uncertain_rule():
    from ripesense.models.infer import Predictor
    p = Predictor.__new__(Predictor)
    p.cfg = {"scan": {"confidence_threshold": 0.6}, "markdown": {}, "base_prices": {"default": 1}}
    p.art = None
    # Heuristic path with no camera features → flat probs → uncertain.
    r = p.predict({k: float("nan") for k in features.ALL_FEATURES}, "banana")
    assert r["uncertain"] and r["stage"] == "uncertain"
