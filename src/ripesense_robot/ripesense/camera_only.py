"""Camera-only banana ripeness app; no GPIO, spectral or gas dependencies.

Run: .venv/bin/python -m ripesense.camera_only
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import threading
import time

import cv2
from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
GUIDANCE = {
    "unripe": ("Not ready yet", "Leave at room temperature and check daily.", "Often several days until ripe"),
    "ripe": ("Ready now", "Ready to eat; refrigerate to slow further ripening.", "Often 1–3 days at room temperature"),
    "overripe": ("Use soon", "If otherwise sound, use in baking or smoothies.", "Use today if otherwise sound"),
}


class CameraPredictor:
    def __init__(self, model_dir):
        import onnxruntime as ort
        self.model_dir = Path(model_dir)
        self.classes = json.loads((self.model_dir / "classes.json").read_text())
        if set(self.classes) != set(GUIDANCE):
            raise ValueError("Expected a three-stage banana model")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(self.model_dir / "stage.onnx"),
                                           sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        metrics_path = self.model_dir / "metrics.json"
        self.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    def predict(self, bgr):
        started = time.perf_counter()
        rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255
        x = ((x - np.array([.485, .456, .406], np.float32)) /
             np.array([.229, .224, .225], np.float32)).transpose(2, 0, 1)[None]
        logits = self.session.run(None, {self.input_name: x})[0][0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        index = int(probabilities.argmax())
        predicted, score = self.classes[index], float(probabilities[index])
        uncertain = score < .65
        headline, tip, guide = GUIDANCE[predicted]
        return {"fruit_type": "banana", "stage": "uncertain" if uncertain else predicted,
                "predicted_stage": predicted, "headline": "Uncertain — retake photo" if uncertain else headline,
                "score": score, "score_label": "Model score (not calibrated confidence)",
                "probabilities": dict(zip(self.classes, map(float, probabilities))),
                "tip": "Move closer, improve lighting, and try another angle." if uncertain else tip,
                "days_left": None, "shelf_life_guide": None if uncertain else guide,
                "shelf_life_basis": "General stage guidance; remaining shelf life is not measured",
                "sensors": {"camera": "used", "spectral": "not used", "gas": "not used"},
                "model": "MobileNetV3-Small / banana stages", "scope": "Bananas only",
                "inference_ms": round((time.perf_counter()-started)*1000, 1)}


def decode_photo(raw):
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("Photo is too large; use at most 16 megapixels")
            image = ImageOps.exif_transpose(image).convert("RGB")
            if min(image.size) < 32:
                raise ValueError("Photo is too small")
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Choose a valid JPEG, PNG or WebP photo") from exc


def create_app(model_dir=None, camera_index=0, predictor=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    model = predictor or CameraPredictor(model_dir or ROOT / "models/camera_only")
    camera_lock = threading.Lock()
    app.config["predictor"] = model

    def infer(bgr):
        if hasattr(model, "predict_described"):
            body = request.get_json(silent=True) or {}
            description = request.form.get("item", body.get("item", ""))
            return model.predict_described(bgr, str(description)[:200])
        return model.predict(bgr)

    @app.get("/")
    def index():
        return render_template("camera_only.html")

    @app.get("/api/health")
    def health():
        return jsonify({"ready": True, "mode": getattr(model, "mode", "camera_only"), "classes": model.classes,
                        "supported_items": getattr(model, "supported_items", ["banana"]),
                        "test": model.metrics.get("test"), "camera_index": camera_index})

    @app.post("/api/predict")
    def predict():
        photo = request.files.get("image")
        if photo is None:
            return jsonify(error="Provide a photo in the image field"), 400
        try:
            bgr = decode_photo(photo.read())
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(infer(bgr))

    @app.post("/api/capture")
    def capture():
        if not camera_lock.acquire(blocking=False):
            return jsonify(error="Camera is busy; try again"), 409
        try:
            from .hw.camera import WebcamCamera
            camera = WebcamCamera({"webcam_index": camera_index, "width": 640,
                                   "height": 480, "warmup_frames": 8})
            frame = camera.capture()
            result = infer(frame.bgr)
            ok, encoded = cv2.imencode(".jpg", frame.bgr)
            if ok:
                result["image"] = "data:image/jpeg;base64," + base64.b64encode(encoded).decode()
            result["source"] = f"attached camera {camera_index}"
            return jsonify(result)
        except (RuntimeError, cv2.error) as exc:
            return jsonify(error=f"Camera capture failed: {exc}. Try browser camera or upload a photo."), 503
        finally:
            camera_lock.release()

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="Photo exceeds 10 MB; choose a smaller photo"), 413

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--model-dir", default=str(ROOT / "models/camera_only"))
    args = parser.parse_args()
    try:
        app = create_app(args.model_dir, args.camera)
    except FileNotFoundError as exc:
        parser.exit(1, f"Missing trained camera model: {exc}\nRun python -m ripesense.models.train_camera_only first.\n")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
