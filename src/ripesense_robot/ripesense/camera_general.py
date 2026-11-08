"""Camera-based produce recognition with specialist models and text descriptions.

python -m ripesense.camera_general
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import time

from flask import jsonify,render_template

from .camera_only import CameraPredictor, ROOT, create_app as base_app
from .camera_multi import MultiFruitPredictor
from .models.openvision import ITEMS, OpenVision
from .models.camera_extras import ExtraFruitModels
from . import robot_view


class GeneralFruitPredictor:
    mode = "camera_general"
    classes = ["unripe","ripe","overripe","visible_deterioration","uncertain","unsupported"]
    supported_items = ITEMS

    def __init__(self,openvision_dir=None,quality_dir=None,banana_dir=None):
        self.vision = OpenVision(openvision_dir or ROOT/"models/openvision")
        self.banana = CameraPredictor(banana_dir or ROOT/"models/camera_only")
        self.quality_dirs = ([Path(quality_dir)] if quality_dir else
                             [ROOT/"models/camera_quality",ROOT/"models/artifacts"])
        self.banana_dir = banana_dir
        self.multi = None
        self.extras = None
        self.metrics = {}
        self._load_quality()

    def _load_quality(self):
        # Another training process may still be exporting the specialist.
        # Only load it once the final classes and metrics have been written.
        if self.multi is None:
            for directory in self.quality_dirs:
                if all((directory/name).exists() for name in
                       ["image_model.onnx","image_model_classes.json","image_model_metrics.json"]):
                    self.multi = MultiFruitPredictor(directory,self.banana_dir)
                    self.metrics = self.multi.metrics
                    break
        if self.extras is None and (ROOT/'models/camera_extras/models.json').exists():
            self.extras = ExtraFruitModels(ROOT/'models/camera_extras')

    def _base(self,item,headline,tip,state="uncertain"):
        return {"fruit_type":item,"stage":state,"predicted_stage":state,"headline":headline,
                "tip":tip,"score":0.,"probabilities":{},"days_left":None,"shelf_life_guide":None,
                "shelf_life_basis":"Remaining shelf life is not measured by this camera assessment",
                "sensors":{"camera":"used","gas":"not used","spectral":"not used"},
                "scope":"Visual assessment of fruits and vegetables",
                "model":"CLIP image-text matching", "assessment_basis":"Experimental description matching",
                "score_label":"Relative caption match, not calibrated confidence", "inference_ms":0.}

    def predict(self,bgr):
        return self.predict_described(bgr)

    def predict_described(self,bgr,description=""):
        started = time.perf_counter()
        self._load_quality()
        vector = self.vision.encode_image(bgr)
        identified = self.vision.identify(vector)
        item = identified["item"]
        specialist_identity = False
        if self.multi and identified["is_produce"]:
            candidate = self.multi.predict(bgr)
            name = candidate["fruit_type"]
            similarity = float(self.vision.encode_text((f"a photo of {name}",))[0] @ vector)
            if ((candidate["fruit_score"] >= .80 and similarity >= identified["similarity"]-.025) or
                (candidate["fruit_score"] >= .90 and identified['margin'] < .02 and
                 similarity >= identified['similarity']-.05)):
                item = name
                specialist_identity = True
        description = re.sub(r"[^\w\s-]"," ",description.lower()).strip()[:120]
        for plural,singular in {"strawberries":"strawberry","blueberries":"blueberry",
                                "raspberries":"raspberry","blackberries":"blackberry",
                                "cherries":"cherry"}.items():
            description = re.sub(r"\b"+plural+r"\b",singular,description)
        if not identified["is_produce"] or identified["similarity"] < .18:
            result = self._base("unknown","Show one fruit or vegetable",
                                "The photo does not clearly match a produce item. Move closer and try again.","unsupported")
        else:
            if not description and identified["margin"] < .01 and not specialist_identity:
                result = self._base("unknown","Fruit identity is unclear",
                                    "Several items look similar. Enter the fruit name or take a closer photo.")
                return self._finish(result,identified,started)
            matches = [candidate for candidate in ITEMS
                       if re.search(r"\b"+re.escape(candidate)+r"(?:s|es)?\b",description)]
            # Prefer longer names if the description contains overlapping terms.
            requested = max(matches,key=len) if matches else description
            if requested:
                requested_similarity = float(self.vision.encode_text((f"a photo of {requested}",))[0] @ vector)
                if requested_similarity < .18 or (requested != item and
                                                  identified["similarity"]-requested_similarity > .035):
                    result = self._base(item,"Item and photo may not match",
                                        f"You described {requested}; the image looks closer to {item}. Retake or correct the description.")
                    return self._finish(result,identified,started)
                item = requested
            result = None
            extras=getattr(self,'extras',None)
            if extras and item in extras.metrics:
                result=extras.predict(bgr,item)
            elif self.multi and item in self.multi.supported_items:
                specialist = self.multi.predict_described(bgr,item)
                if specialist["fruit_type"] == item:
                    result = specialist
                    result["assessment_basis"] = "Trained fruit type and visible-quality classifier"
                    result["score_label"] = "Uncalibrated specialist model score"
                else:
                    result = self._base(item,"Models disagree — retake photo",
                                        "The recognition and quality models disagree about the fruit. Try a closer photo.")
            elif item == "banana":
                result = self.banana.predict(bgr)
                result["assessment_basis"] = "Trained banana ripeness classifier"
            if result is None:
                assessment = self.vision.assess_description(vector,item)
                stage = assessment["stage"]
                headlines = {"unripe":"Looks unripe", "ripe":"Looks ripe",
                             "overripe":"Looks overripe", "visible_deterioration":"Possible visible deterioration"}
                result = self._base(item,headlines[stage]+" · experimental",
                                    f"Closest visual description: {stage.replace('_',' ')}. This comparison has not been validated for {item}.",stage)
                result.update(predicted_stage=stage,score=assessment["score"],
                              probabilities=assessment["probabilities"],caption_similarities=assessment["caption_similarities"])
                if assessment["margin"] < .015:
                    result.update(stage="uncertain",headline="Appearance is inconclusive",
                                  tip="The visual descriptions scored similarly. Try another view; a photo may not reveal ripeness.")
        return self._finish(result,identified,started)

    @staticmethod
    def _finish(result,identified,started):
        result["recognition"] = identified
        result["inference_ms"] = round((time.perf_counter()-started)*1000,1)
        return result


def create_app(predictor=None,camera_index=0,robot_host=robot_view.DEFAULT_ROBOT):
    model = predictor or GeneralFruitPredictor()
    app = base_app(predictor=model,camera_index=camera_index)
    app.view_functions["index"] = lambda: render_template("camera_general.html")
    def health():
        model._load_quality()
        trained=set(model.multi.supported_items if model.multi else ['banana'])
        if model.extras:trained.update(model.extras.metrics)
        return jsonify(ready=True,mode=model.mode,classes=model.classes,supported_items=ITEMS,
                       trained_items=sorted(trained),
                       open_vocabulary=True,description_assessment_validated=False,
                       test=model.metrics.get("test"),
                       specialist_tests={k:v['test'] for k,v in model.extras.metrics.items()} if model.extras else {},
                       camera_index=camera_index)
    app.view_functions["health"] = health
    robot_view.register(app,robot_host)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=5001)
    parser.add_argument("--camera",type=int,default=0)
    parser.add_argument("--robot",default=robot_view.DEFAULT_ROBOT,help="AiNex robot IP for the /robot live view")
    a = parser.parse_args()
    create_app(camera_index=a.camera,robot_host=a.robot).run(host=a.host,port=a.port,threaded=True,use_reloader=False)


if __name__ == "__main__":
    main()
