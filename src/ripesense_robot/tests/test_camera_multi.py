import numpy as np
import pytest

from ripesense.camera_multi import MultiFruitPredictor


class Session:
    def __init__(self, logits):
        self.logits = np.array([logits],dtype=np.float32)

    def run(self, _, inputs):
        image = inputs["image"]
        assert image.shape == (1,3,224,224)
        return [self.logits]


def predictor(logits):
    p = MultiFruitPredictor.__new__(MultiFruitPredictor)
    p.classes = ["fresh_apple","fresh_banana","rotten_apple","rotten_banana"]
    p.labels = [c.split("_",1) for c in p.classes]
    p.supported_items = ["apple","banana"]
    p.input = "image"
    p.session = Session(logits)
    p.banana = None
    return p


def test_apple_does_not_get_banana_stages():
    r = predictor([9,0,-2,0]).predict(np.zeros((120,180,3),np.uint8))
    assert r["fruit_type"] == "apple" and r["stage"] == "fresh"
    assert r["days_left"] is None and r["shelf_life_guide"] is None
    assert r["quality_probabilities"]["fresh"] > .99


def test_named_unsupported_item_has_no_invented_prediction():
    r = predictor([9,0,-2,0]).predict_described(None,"a ripe avocado")
    assert r["stage"] == "unsupported"
    assert r["fruit_type"] == "unknown" and r["probabilities"] == {}


def test_item_hint_does_not_force_visual_result():
    r = predictor([9,0,-2,0]).predict_described(np.zeros((120,180,3),np.uint8),"banana")
    assert r["stage"] == "uncertain" and r["fruit_type"] == "apple"
    assert "entered banana" in r["tip"]


def test_low_scores_ask_for_another_photo():
    r = predictor([0,0,0,0]).predict(np.zeros((120,180,3),np.uint8))
    assert r["stage"] == "uncertain"


def test_rotten_apple_is_assessed_from_apple_scores_only():
    r = predictor([0,-5,9,-5]).predict_described(np.zeros((120,180,3),np.uint8),"red apples")
    assert r["stage"] == "spoiled" and r["fruit_type"] == "apple"
    assert r["quality_probabilities"]["rotten"] > .99
    assert sum(r["fruit_probabilities"].values()) == pytest.approx(1)
