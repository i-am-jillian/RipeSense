"""Hackathon upgrades: forecast, few-shot, pricing, conformal, IR bruises,
e-nose, fruit profile, future preview, why-map, and their API endpoints."""
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from ripesense import features, forecast, fruit_info, pricing
from ripesense.sim.fruit import SimFruit

ART = Path(__file__).resolve().parents[1] / "models" / "artifacts"


# ------------------------------------------------------------------ forecast
def _hist(n, rate=0.4, q0=0.1, noise=0.12, seed=1):
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 11, 6, 9)
    return [{"t": t0 + timedelta(hours=12 * k), "q": q0 + rate * 0.5 * k * forecast.temp_factor(20) + rng.normal(0, noise),
             "conf": 0.85, "temp": 20.0} for k in range(n)]


def test_forecast_band_narrows_and_tracks_truth():
    one = forecast.forecast(_hist(1), "banana")
    six = forecast.forecast(_hist(6), "banana")
    w1 = one["days_left"][2] - one["days_left"][0]
    w6 = six["days_left"][2] - six["days_left"][0]
    assert w6 < 0.7 * w1                                   # more scans → tighter
    true_left = (2.5 - (0.1 + 0.4 * 2.5)) / 0.4            # after 6 scans (2.5 days)
    assert abs(six["days_left"][1] - true_left) < 1.0
    assert six["days_left"][0] <= true_left <= six["days_left"][2]


def test_forecast_whatifs_have_physical_signs():
    w = forecast.whatifs(_hist(3), "banana")
    assert w["fridge"]["delta_days"] > 0 > w["warm"]["delta_days"]
    assert w["bag"]["delta_days"] <= 0
    assert any("chill" in n for n in w["fridge"]["notes"])


def test_history_from_rows_skips_uncertain():
    rows = [{"pred_stage": "ripe", "pred_conf": "0.8", "timestamp": "2026-11-06T09:00:00", "temp": "", "pred_score": ""},
            {"pred_stage": "uncertain", "pred_conf": "0.3", "timestamp": "2026-11-06T10:00:00"}]
    h = forecast.history_from_rows(rows)
    assert len(h) == 1 and h[0]["q"] == 1.0 and h[0]["temp"] is None


# ------------------------------------------------------------------- few-shot
def test_fewshot_accepts_taught_and_rejects_others(tmp_path):
    from ripesense.models.fewshot import FewShot
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(4, 576))
    np.savez(tmp_path / "bank.npz", names=np.array(["a", "b", "c"]), protos=centres[:3])
    fs = FewShot(tmp_path / "fs.json", tmp_path / "bank.npz")
    for _ in range(5):
        fs.add("kiwi", centres[3] + rng.normal(scale=0.3, size=576))
    hit = fs.match(centres[3] + rng.normal(scale=0.3, size=576))
    miss = fs.match(centres[0] + rng.normal(scale=0.3, size=576))
    assert hit["accepted"] and hit["fruit"] == "kiwi"
    assert not miss["accepted"]
    fs.add("kiwi:ripe", centres[3])
    assert FewShot(tmp_path / "fs.json").summary() == {"kiwi": 5, "kiwi:ripe": 1}   # persisted
    fs.forget("kiwi")
    assert "kiwi" not in fs.summary()


# -------------------------------------------------------------------- pricing
def test_optimizer_never_worse_than_tier_and_policies():
    for stage, d, tier in [("ripe", {"p10": 2, "p50": 3, "p90": 4}, 0),
                           ("ripe_to_overripe", {"p10": 1, "p50": 1.5, "p90": 2}, 32.5),
                           ("overripe", {"p10": 0.3, "p50": 0.8, "p90": 1.5}, 60)]:
        o = pricing.optimise(stage, "banana", 0.29, d, tier_pct=tier)
        assert o["vs_tier"] >= -1e-9
    assert pricing.optimise("unripe", "banana", 0.29, {"p10": 5, "p50": 6, "p90": 8})["markdown_pct"] == 0
    assert pricing.optimise("overripe", "banana", 0.29, {"p10": 0, "p50": 0, "p90": 0.1})["action"] == "donate"
    assert pricing.optimise("spoiled", "banana", 0.29, {"p10": 0, "p50": 0, "p90": 0}) is None


# ------------------------------------------------------------------ conformal
def test_conformal_reaches_target_coverage():
    from ripesense.models.train_fusion import conformal
    rng = np.random.default_rng(3)
    y = rng.uniform(0, 8, 400)
    pred = y + rng.normal(0, 1.0, 400)
    c = conformal(pred - 0.5, pred + 0.5, y)            # too-narrow raw intervals
    assert c["raw_coverage"] < 0.5 and c["coverage"] >= 0.9
    y2 = rng.uniform(0, 8, 400)
    p2 = y2 + rng.normal(0, 1.0, 400)
    assert np.mean(np.abs(y2 - p2) <= 0.5 + c["q"]) > 0.85   # holds on fresh data


# ----------------------------------------------------------------- IR, e-nose
def test_ir_bruises_found_only_when_present():
    clean = SimFruit("banana", 3.0, hidden_bruises=0, seed=5)
    bruised = SimFruit("banana", 3.0, hidden_bruises=2, seed=5)
    f0, _ = features.ir_bruises(*clean.render_ir())
    f2, ov = features.ir_bruises(*bruised.render_ir())
    assert f0["ir_bruise_count"] == 0
    assert f2["ir_bruise_count"] >= 1 and f2["ir_bruise_area_cm2"] > 0 and ov is not None
    assert np.isnan(features.ir_bruises(None)[0]["ir_bruise_pct"])


def test_enose_profile_shape_tracks_ripeness():
    def feats(days):
        f = SimFruit("banana", days)
        w = f.enose_weights()
        b = [260000, 180000, 120000, 80000, 52000]
        r = f.gas_response(120)
        return features.enose_features({"baseline": b, "end": [bi * (1 - r * wi) for bi, wi in zip(b, w)]})
    ripe, rotten = feats(3.5), feats(-1.0)
    assert rotten["enose_shape"] > ripe["enose_shape"]   # ethanol: hotter plate responds more
    assert all(k in features.GAS_FEATURES for k in ripe)
    assert np.isnan(features.enose_features(None)["enose_r200"])


# ------------------------------------------------------------- fruit profile
def test_profile_banana_starch_to_sugar_and_weight():
    green = fruit_info.profile("banana", "unripe", 0.0, {"volume_cm3": 150.0, "length_mm": 180, "width_mm": 38})
    ripe = fruit_info.profile("banana", "ripe", 1.0, {"volume_cm3": 150.0, "length_mm": 180, "width_mm": 38})
    assert green["nutrition"]["sugar_g"] < ripe["nutrition"]["sugar_g"]
    assert green["starch_g_100g"] > ripe["starch_g_100g"]
    assert green["weight_source"] == "depth" and green["weight_g"] == round(150 * 0.94)
    assert fruit_info.profile("apple", "ripe", 1.0)["weight_source"] == "typical"
    assert fruit_info.stage_score({"ripe": 0.5, "overripe": 0.5}) == 1.5
    tips = fruit_info.ethylene_advice("banana", "overripe")
    assert "kiwi" in tips["keep_away_from"]


def test_future_preview_only_changes_the_fruit():
    from ripesense.future import preview
    f = SimFruit("banana", 6.0, seed=2)
    bgr, depth = f.render()
    mask = features.fruit_mask(bgr, depth, 188.0, 8.0)
    out = preview(bgr, "banana", 0.1, 2.2, mask=mask)
    diff = np.abs(out.astype(int) - bgr.astype(int)).sum(-1)
    assert diff[mask > 0].mean() > 20 and diff[mask == 0].max() == 0
    assert np.array_equal(preview(bgr, "banana", 1.0, 1.0, mask=mask), bgr)


@pytest.mark.skipif(not (ART / "stage_model.onnx").exists(), reason="trained image models not present")
def test_why_map_and_embedding():
    import cv2
    from ripesense.models.explain import explain
    from ripesense.models.image import Models
    m = Models(ART)
    bgr = cv2.cvtColor(np.full((240, 320, 3), (40, 200, 200), np.uint8), cv2.COLOR_HSV2BGR)
    ex = explain(m, bgr, None, "banana", "ripe")
    assert ex["image"].shape == (448, 448, 3) and ex["model"] == "stage model"
    e = m.image.embed(bgr)
    assert e.shape == (576,) and abs(np.linalg.norm(e) - 1) < 1e-4


# ---------------------------------------------------------------- store/app
def test_old_database_is_migrated(tmp_path, monkeypatch):
    monkeypatch.setenv("RIPESENSE_DATA_DIR", str(tmp_path))
    from ripesense import config, store
    config.load.cache_clear()
    con = sqlite3.connect(tmp_path / "ripesense.db")
    con.execute('CREATE TABLE scans (scan_id INTEGER PRIMARY KEY AUTOINCREMENT, "fruit_id" TEXT)')
    con.commit()
    con.close()
    sid = store.insert({"fruit_id": "X", "pred_score": 1.2, "gas_profile": {"baseline": [1], "end": [1]}})
    assert store.get(sid)["pred_score"] == "1.2"
    config.load.cache_clear()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RIPESENSE_MOCK", "1")
    monkeypatch.setenv("RIPESENSE_FAST", "1")
    monkeypatch.setenv("RIPESENSE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RIPESENSE_MODELS_DIR", str(tmp_path / "models"))
    from ripesense import config
    config.load.cache_clear()
    from ripesense.app import create_app
    app = create_app()
    yield app.test_client(), app.config["ctl"]
    config.load.cache_clear()


def _scan(c):
    assert c.post("/api/scan").status_code == 200
    t0 = time.time()
    while time.time() - t0 < 40:
        s = c.get("/api/status").get_json()
        if s["state"] in ("result", "error"):
            return s
        time.sleep(0.2)
    raise TimeoutError


def test_scan_has_forecast_profile_ir_and_unplug(client):
    c, ctl = client
    c.post("/api/sim", json={"fruit_type": "banana", "stage": "ripe"})
    c.post("/api/sim", json={"keep": True, "hidden_bruises": 2})
    c.post("/api/collect", json={"fruit_id": "B9"})
    s = _scan(c)
    r = s["result"]
    assert s["state"] == "result"
    assert r["ir"] and r["ir"]["ir_bruise_count"] >= 1
    assert r["features"]["enose_r400"] is not None
    if r["stage"] not in ("uncertain", "spoiled"):
        assert r["forecast"]["days_left"] and r["profile"]["weight_source"] == "depth"
        sid = r["scan_id"]
        assert c.get(f"/api/forecast/{sid}?scenario=fridge").status_code == 200
        assert c.get(f"/api/future/{sid}.jpg?days=2").mimetype == "image/jpeg"
        assert c.get(f"/api/recipe/{sid}").get_json()["title"]
    assert c.post("/api/unplug", json={"sensor": "gas", "off": True}).get_json()["disabled"] == ["gas"]
    r2 = _scan(c)["result"]
    assert r2["sensors"]["gas"] == "missing" and "gas" in r2["offline"]
    assert c.post("/api/unplug", json={"sensor": "nose"}).status_code == 400
    assert isinstance(c.get("/api/label_queue").get_json(), list)
    assert c.get("/api/train_history").get_json() == []
    assert "impact" in c.get("/api/store_summary").get_json()
    assert c.post("/api/teach", json={"label": "kiwi"}).status_code == 400   # no image model in tmp dir
