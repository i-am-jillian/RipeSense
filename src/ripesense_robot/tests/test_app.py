"""End-to-end: mock hardware → scan → stored row → result JSON, through Flask."""
import time

import pytest


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


def wait_result(c, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = c.get("/api/status").get_json()
        if s["state"] in ("result", "error"):
            return s
        time.sleep(0.2)
    raise TimeoutError


def test_full_mock_scan(client):
    c, ctl = client
    assert c.post("/api/calibrate").get_json()["ok"]
    c.post("/api/sim", json={"fruit_type": "banana", "stage": "overripe"})
    c.post("/api/collect", json={"fruit_id": "B01", "fruit_type": "banana", "stage_label": "overripe"})
    assert c.post("/api/scan").status_code == 200
    s = wait_result(c)
    assert s["state"] == "result", s["message"]
    r = s["result"]
    assert r["uncalibrated"] is True               # no trained model in tmp dir
    assert set(r["sensors"]) == {"camera", "spectral", "gas"}
    assert r["features"]["gas_rise"] is not None   # gas series was captured
    assert r["features"]["r_f8_680"] is not None   # spectral reflectance computed
    rows = c.get("/api/scans").get_json()
    assert rows[0]["fruit_id"] == "B01" and rows[0]["stage_label"] == "overripe"
    # Label back-fill + Store mode.
    assert c.post("/api/spoiled", json={"fruit_id": "B01"}).get_json()["updated"] == 1
    assert c.post("/api/mode", json={"mode": "store"}).get_json()["mode"] == "store"
    assert c.get("/api/store_summary").get_json()["source"] == "local"


def test_door_open_blocks_scan(client):
    c, ctl = client
    c.post("/api/door", json={"closed": False})
    assert c.post("/api/scan").status_code == 409
    assert c.get("/api/status").get_json()["state"] == "close_door"
