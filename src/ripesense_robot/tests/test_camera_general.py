import numpy as np

from ripesense.camera_general import GeneralFruitPredictor


class Vision:
    def __init__(self,item="avocado",produce=True,margin=.1):
        self.item,self.produce,self.margin=item,produce,margin

    def encode_image(self,image):
        return np.array([1.,0.])

    def identify(self,vector):
        return {"item":self.item,"is_produce":self.produce,"similarity":.5,
                "margin":self.margin,"candidates":[]}

    def encode_text(self,captions):
        return np.array([[.5,0.] for _ in captions])

    def assess_description(self,vector,item):
        return {"stage":"ripe","score":.4,"margin":.03,
                "probabilities":{"unripe":.2,"ripe":.4,"overripe":.2,"visible_deterioration":.2},
                "caption_similarities":{}}


def predictor(vision):
    p=GeneralFruitPredictor.__new__(GeneralFruitPredictor)
    p.vision=vision
    p.multi=None
    p._load_quality=lambda:None
    return p


def test_beyond_banana_uses_explicit_experimental_basis():
    p=predictor(Vision())
    r=p.predict_described(None,"avocado")
    assert r["fruit_type"] == "avocado" and r["stage"] == "ripe"
    assert "experimental" in r["headline"]
    assert r["assessment_basis"] == "Experimental description matching"
    assert r["days_left"] is None


def test_non_produce_is_not_given_ripeness():
    r=predictor(Vision("a computer keyboard",False)).predict(None)
    assert r["stage"] == "unsupported" and r["probabilities"] == {}
    assert r["days_left"] is None


def test_ambiguous_identity_requests_description():
    r=predictor(Vision(margin=.001)).predict(None)
    assert r["stage"] == "uncertain" and r["fruit_type"] == "unknown"
    assert "name" in r["tip"]


def test_description_mismatch_does_not_force_an_answer():
    p=predictor(Vision("apple"))
    p.vision.encode_text=lambda captions: np.array([[.1,0.]])
    r=p.predict_described(None,"avocado")
    assert r["stage"] == "uncertain" and r["fruit_type"] == "apple"
    assert "may not match" in r["headline"]


def test_unknown_description_can_be_compared_without_claiming_training():
    r=predictor(Vision()).predict_described(None,"a longan fruit")
    assert r["fruit_type"] == "a longan fruit"
    assert "Experimental" in r["assessment_basis"]
    assert "not been validated" in r["tip"]
