"""Camera-only input contracts, sensor independence, and real model smoke test."""
import io
import pytest
from PIL import Image

from ripesense.camera_only import CameraPredictor, ROOT, create_app


class FakePredictor:
    classes = ["overripe", "ripe", "unripe"]
    metrics = {}

    def predict(self, bgr):
        assert bgr.shape == (80, 120, 3)
        return {"stage": "ripe", "days_left": None}


def photo():
    out = io.BytesIO()
    Image.new("RGB", (120, 80), (240, 200, 20)).save(out, format="JPEG")
    out.seek(0)
    return out


def test_no_hardware_initialization(monkeypatch):
    # This route must work without importing or constructing cabinet controls.
    import ripesense.scan
    monkeypatch.setattr(ripesense.scan.Controller, "__init__",
                        lambda *_: pytest.fail("Cabinet controller was initialized"))
    c = create_app(predictor=FakePredictor()).test_client()
    assert c.get("/").status_code == 200
    assert c.get("/api/health").get_json()["mode"] == "camera_only"
    response = c.post("/api/predict", data={"image": (photo(), "banana.jpg")})
    assert response.status_code == 200
    assert response.get_json() == {"stage": "ripe", "days_left": None}


def test_bad_inputs():
    c = create_app(predictor=FakePredictor()).test_client()
    assert c.post("/api/predict").status_code == 400
    assert c.post("/api/predict", data={"image": (io.BytesIO(b"not a photo"), "bad.jpg")}).status_code == 400
    response = c.post("/api/predict", data={"image": (io.BytesIO(b"x" * (10*1024*1024)), "large.jpg")})
    assert response.status_code == 413
    assert "10 MB" in response.get_json()["error"]


def test_real_trained_model():
    model_dir = ROOT / "models/camera_only"
    assert (model_dir / "stage.onnx").exists(), "Train the actual camera model first"
    model = CameraPredictor(model_dir)
    sample = next((ROOT / "datasets/camera_banana/test/ripe").glob("*.jpg"))
    c = create_app(predictor=model).test_client()
    with sample.open("rb") as f:
        response = c.post("/api/predict", data={"image": (f, sample.name)})
    assert response.status_code == 200
    result = response.get_json()
    assert set(result["probabilities"]) == {"unripe", "ripe", "overripe"}
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["days_left"] is None
    assert result["sensors"]["gas"] == "not used"
    assert result["inference_ms"] > 0
