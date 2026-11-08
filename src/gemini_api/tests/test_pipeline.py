"""End-to-end tests in simulation. Run: python -m pytest -q"""
import math
import time

import numpy as np
import pytest

from ripesense import calibration, config
from ripesense.features import segment, image_features, spectral_features
from ripesense.hardware import build_devices
from ripesense.hardware.camera import MockCamera
from ripesense.hardware.sim import WORLD
from ripesense.hardware.spectral import MockSpectral, measure
from ripesense.model import HeuristicModel, Prediction, STAGES
from ripesense.scan import ScanController
from ripesense.storage import Store
from ripesense.train import simulate_dataset, train
from ripesense.verdict import verdict


@pytest.fixture
def cfg(tmp_path):
    return config.load(simulate=True, app={"data_dir": str(tmp_path / "data")},
                       model={"path": str(tmp_path / "model.joblib")}, gas={"scan_seconds": 2})


@pytest.fixture
def devices(cfg):
    d = build_devices(cfg)
    time.sleep(0.5)
    yield d
    d.close()


def _insert(fruit="banana", r=0.4):
    WORLD.open_door()
    time.sleep(1.2)
    WORLD.set_fruit(fruit, r)


def test_segmentation_and_size(cfg):
    WORLD.set_fruit("banana", 0.4)
    f = MockCamera(cfg["camera"], delay=0).capture()
    mask, method, floor = segment(f.color, f.depth)
    assert method == "depth" and mask.any()
    feats, _ = image_features(f, mask, floor)
    assert 60 < feats["i_length_mm"] < 160        # banana-sized
    assert 10 < feats["i_volume_cm3"] < 200
    assert feats["c_yellow_frac"] > 0.5


def test_spectral_ripening_index_falls_as_fruit_ripens(cfg):
    class IO:
        state = {}
        def set(self, k, v): self.state[k] = v
    io = IO()
    spec = MockSpectral(cfg["spectral"], io)
    WORLD.puck = True
    m = measure(spec, io, cfg["spectral"]); WORLD.puck = False
    calib = {"dark": m["dark"], "white": m["white"], "ir": m["ir"]}
    vals = []
    for r in (0.0, 0.35, 0.95):
        WORLD.set_fruit("banana", r)
        vals.append(spectral_features(measure(spec, io, cfg["spectral"]), calib)["s_ripen"])
    assert vals[0] > vals[1] > vals[2]


def test_heuristic_is_monotonic():
    h = HeuristicModel()
    order = []
    for g, d in [(0.9, 0.0), (0.0, 0.0), (0.0, 0.3), (0.0, 0.8)]:
        f = {"c_green_frac": g, "c_dark_frac": d, "c_brown_frac": 0.0, "c_spot_density": 0.0}
        order.append(STAGES.index(h.predict(f, "banana").stage))
    assert order == sorted(order) and order[0] == 0 and order[-1] == 3


def test_verdict_store_markdown(cfg):
    p = Prediction("ripe", {}, 0.5, 1.5, 2.5, 0.9, {}, False, "model")
    v = verdict(p, "banana", "store", cfg["store"])
    assert v["markdown"] == [25, 40] and v["price"] < v["list_price"]
    p.uncertain = True
    assert verdict(p, "banana", "home", cfg["store"])["headline"].startswith("Uncertain")


def test_full_scan_in_simulation(cfg, devices):
    WORLD.fruit_present = False
    calibration.capture_empty(cfg, devices)
    WORLD.puck = True
    calibration.calibrate_white(cfg, devices)
    WORLD.puck = False
    ctl = ScanController(cfg, devices)
    _insert("banana", 0.05)
    assert ctl.start()
    ctl.wait(30)
    s = ctl.snapshot()
    assert s["status"] == "result", s
    r = s["result"]
    assert r["prediction"]["stage"] == "unripe"
    assert r["segmentation"] == "depth"
    row = ctl.store.get(s["scan_id"])
    assert row["raw"]["gas_baseline_source"] == "door-open air"
    assert math.isfinite(row["features"]["g_rise"])
    assert (cfg.data_dir / "scans" / f"{s['scan_id']}_thumb.jpg").exists()


def test_scan_aborts_when_door_open(cfg, devices):
    ctl = ScanController(cfg, devices)
    WORLD.open_door()
    ctl.start(); ctl.wait(10)
    s = ctl.snapshot()
    assert s["status"] == "error" and "door" in s["message"].lower()
    WORLD.close_door()


def test_train_on_synthetic_data(cfg):
    store = Store(cfg.data_dir)
    n = simulate_dataset(cfg, store, n_fruits=24, scans_per_fruit=8)
    assert n == 192
    m = train(cfg, store, include_synthetic=True)
    assert m["stage_macro_f1"] > 0.6, m
    assert m["days_mae"] < 3.0, m
    from ripesense.model import load_predictor
    model = load_predictor(cfg)
    rows = store.training_rows(include_synthetic=True)
    pred = model.predict(rows[0]["features"], "banana")
    assert pred.mode == "model" and pred.stage in STAGES
    # one sensor missing still gives an answer
    f = dict(rows[-1]["features"]); [f.__setitem__(k, float("nan")) for k in list(f) if k.startswith("g_")]
    assert model.predict(f, "banana").stage in STAGES


def test_web_api(cfg, devices):
    from ripesense.web import create_app
    app = create_app(cfg, devices)
    c = app.test_client()
    assert c.get("/").status_code == 200 and c.get("/collect").status_code == 200
    assert c.get("/api/info").get_json()["simulate"] is True
    c.post("/api/sim/insert", json={"fruit": "banana", "ripeness": 0.45})
    assert c.post("/api/scan", json={"fruit_id": "B01"}).status_code == 202
    for _ in range(80):
        s = c.get("/api/state").get_json()
        if s["status"] in ("result", "error"):
            break
        time.sleep(0.25)
    assert s["status"] == "result", s
    sid = s["scan_id"]
    assert c.get(f"/api/scans/{sid}/thumb.jpg").status_code == 200
    assert s["live"]["preview"] and len(s["live"]["spectrum"]) == 8 and len(s["live"]["gas"]) > 2
    for _ in range(40):   # the recipe arrives after the result
        s = c.get("/api/state").get_json()
        if s["result"]["recipe_status"] != "pending":
            break
        time.sleep(0.25)
    assert s["result"]["recipe_status"] == "ready", s["result"]
    rec = c.get(f"/api/scans/{sid}/recipe").get_json()
    assert rec["recipe"]["title"] and rec["recipe"]["steps"]
    svg = c.get(f"/api/scans/{sid}/qr.svg")
    assert svg.status_code == 200 and b"<svg" in svg.data
    assert c.get(f"/r/{sid}").status_code == 200
    assert c.post("/api/label", json={"scan_id": sid, "stage": "ripe", "days": 3}).status_code == 200
    assert c.post("/api/mode", json={"mode": "store"}).get_json()["mode"] == "store"
    csv = c.get("/api/export.csv").get_data(as_text=True)
    assert csv.startswith("scan_id,") and "\n" in csv
    assert c.post("/api/train", json={}).status_code == 400   # not enough labels yet


def test_recipes_offline_and_links(cfg):
    from ripesense.recipes import get_recipe
    from ripesense.web import recipe_link
    r = get_recipe("banana", "overripe", 1, "home", cfg)     # no GEMINI_API_KEY in tests
    assert r["source"] == "built-in" and "banana" in r["title"].lower()
    assert get_recipe("banana", "spoiled", 0, "home", cfg) is None
    assert recipe_link(cfg, 7, r, "banana").endswith("/r/7")
    cfg["qr"]["target"] = "search"
    assert recipe_link(cfg, 7, r, "banana").startswith("https://www.google.com/search?q=One-bowl+banana+bread")


def test_gemini_recipe_with_model_fallback(cfg, monkeypatch):
    """Fake Gemini client: first model name is rejected, the fallback answers with JSON."""
    import json as _json
    from google import genai
    from ripesense import recipes
    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls.append((model, config.response_mime_type))
            if model == cfg["gemini"]["model"]:
                raise RuntimeError("404 model not found")
            return type("R", (), {"text": _json.dumps({
                "tip": "Bake it today.", "title": "Banana pancakes", "why": "Soft bananas mash easily.",
                "minutes": 15, "servings": 2, "ingredients": ["2 bananas", "2 eggs"], "steps": ["Mash", "Fry"]})})()

    class FakeClient:
        def __init__(self, *a, **k):
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setattr(genai, "Client", FakeClient)
    r = recipes.get_recipe("banana", "overripe", 1, "home", cfg)
    assert r["title"] == "Banana pancakes" and r["source"] == "Gemini (gemini-2.5-flash)"
    assert [c[0] for c in calls] == [cfg["gemini"]["model"], "gemini-2.5-flash"]
    assert calls[0][1] == "application/json"
