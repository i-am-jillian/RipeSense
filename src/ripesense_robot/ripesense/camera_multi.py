"""Multi-fruit RGB assessment. Run python -m ripesense.camera_multi."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import cv2
from flask import render_template
import numpy as np
from PIL import Image

from .camera_only import CameraPredictor, ROOT, create_app as base_app


class MultiFruitPredictor:
    mode = "camera_multi"

    def __init__(self, quality_dir=None, banana_dir=None):
        import onnxruntime as ort
        root = Path(quality_dir or ROOT / "models/artifacts")
        self.classes = json.loads((root / "image_model_classes.json").read_text())
        self.labels = [label.split("_", 1) for label in self.classes]
        if any(len(pair) != 2 or pair[0] not in ("fresh", "rotten") for pair in self.labels):
            raise ValueError("Expected fresh_<fruit> / rotten_<fruit> labels")
        self.supported_items = sorted({fruit for _, fruit in self.labels})
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(root / "image_model.onnx"), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.banana = CameraPredictor(banana_dir or ROOT / "models/camera_only")
        metrics = root / "image_model_metrics.json"
        self.metrics = json.loads(metrics.read_text()) if metrics.exists() else {}

    @staticmethod
    def preprocess(bgr):
        # Match train_image's PIL Resize(240) + CenterCrop(224), including
        # aspect ratio and interpolation. Do not squash the full photo.
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        w, h = image.size
        size = (240, int(h*240/w)) if w < h else (int(w*240/h), 240)
        image = image.resize(size, Image.Resampling.BILINEAR)
        x, y = round((size[0]-224)/2), round((size[1]-224)/2)
        image = image.crop((x,y,x+224,y+224))
        a = np.asarray(image).astype(np.float32)/255
        a = (a-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        return a.transpose(2,0,1)[None]

    def _description(self, description):
        words = set(re.findall(r"[a-z]+", description.lower()))
        aliases = {"grapes":"grape", "strawberries":"strawberry", "tomatoes":"tomato",
                   "avocados":"avocado", "apples":"apple", "bananas":"banana",
                   "oranges":"orange", "limes":"lime", "jujubes":"jujube",
                   "pomegranates":"pomegranate", "guavas":"guava"}
        words = {aliases.get(w,w) for w in words}
        matches = sorted(words & set(self.supported_items))
        if len(matches) == 1:
            return matches[0], None
        if description.strip():
            return None, "Name one supported fruit, or leave the item field empty for automatic identification."
        return None, None

    def _unavailable(self, reason):
        return {"fruit_type": "unknown", "stage":"unsupported", "predicted_stage":"unsupported",
                "headline":"Item not supported", "score":0., "probabilities":{}, "tip":reason,
                "days_left":None, "shelf_life_guide":None,
                "shelf_life_basis":"No shelf-life prediction available", "inference_ms":0.,
                "supported_items":self.supported_items, "scope":"Fruit-specific trained visual assessment",
                "sensors":{"camera":"used", "spectral":"not used", "gas":"not used"},
                "model":"Multi-fruit MobileNetV3-Small", "score_label":"Uncalibrated model score"}

    def predict(self, bgr):
        return self.predict_described(bgr)

    def predict_described(self, bgr, description=""):
        hint, error = self._description(description)
        if error:
            return self._unavailable(error)
        started = time.perf_counter()
        logits = self.session.run(None, {self.input:self.preprocess(bgr)})[0][0]
        p = np.exp(logits-logits.max())
        p /= p.sum()
        fruits = {fruit:float(sum(p[i] for i,(_,f) in enumerate(self.labels) if f == fruit))
                  for fruit in self.supported_items}
        fruit = max(fruits,key=fruits.get)
        fruit_score = fruits[fruit]
        quality = {quality:float(sum(p[i] for i,(q,f) in enumerate(self.labels)
                                    if f == fruit and q == quality))/max(fruit_score,1e-9)
                   for quality in ("fresh","rotten")}
        condition = max(quality,key=quality.get)
        mismatch = hint is not None and hint != fruit
        uncertain = fruit_score < .65 or quality[condition] < .70 or mismatch
        predicted = "fresh" if condition == "fresh" else "spoiled"
        probabilities = {"fresh-looking":quality["fresh"],"visible deterioration":quality["rotten"]}
        headline = "Looks fresh" if condition == "fresh" else "Visible deterioration"
        tip = ("No obvious deterioration predicted from this photo. Check the fruit before use."
               if condition == "fresh" else "Visible condition suggests deterioration; inspect before deciding what to do.")
        stage_score = quality[condition]
        model_name = "MobileNetV3-Small / fruit type and visible quality"
        if fruit == "banana" and condition == "fresh" and not uncertain:
            detailed = self.banana.predict(bgr)
            predicted = detailed["predicted_stage"]
            probabilities = detailed["probabilities"]
            headline, tip = detailed["headline"], detailed["tip"]
            stage_score = detailed["score"]
            uncertain = detailed["stage"] == "uncertain"
            model_name += " + banana stages"
        if uncertain:
            headline = "Uncertain — retake photo"
            tip = (f"You entered {hint}, but the photo resembles {fruit}. Check the item and retake."
                   if mismatch else "Frame one fruit closely in even light, or try another angle.")
        return {"fruit_type":fruit, "fruit_score":fruit_score, "fruit_probabilities":fruits,
                "stage":"uncertain" if uncertain else predicted, "predicted_stage":predicted,
                "headline":headline, "score":float(stage_score), "probabilities":probabilities,
                "quality_probabilities":quality, "tip":tip, "days_left":None, "shelf_life_guide":None,
                "shelf_life_basis":"Remaining shelf life cannot be established from this photo",
                "supported_items":self.supported_items, "scope":"Fruit-specific trained visual assessment",
                "sensors":{"camera":"used", "spectral":"not used", "gas":"not used"},
                "model":model_name, "score_label":"Uncalibrated model score",
                "inference_ms":round((time.perf_counter()-started)*1000,1)}


def create_app(quality_dir=None, banana_dir=None, camera_index=0, predictor=None):
    model = predictor or MultiFruitPredictor(quality_dir,banana_dir)
    app = base_app(predictor=model,camera_index=camera_index)
    app.view_functions["index"] = lambda: render_template("camera_multi.html")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=5001)
    parser.add_argument("--camera",type=int,default=0)
    parser.add_argument("--quality-dir",default=str(ROOT/"models/artifacts"))
    parser.add_argument("--banana-dir",default=str(ROOT/"models/camera_only"))
    a = parser.parse_args()
    try:
        app = create_app(a.quality_dir,a.banana_dir,a.camera)
    except FileNotFoundError as exc:
        parser.exit(1,f"Missing trained model: {exc}\nTrain the multi-fruit model before starting.\n")
    app.run(host=a.host,port=a.port,threaded=True,use_reloader=False)


if __name__ == "__main__":
    main()
