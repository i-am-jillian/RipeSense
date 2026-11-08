"""Lightweight inference heads for the additional labeled produce datasets."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..camera_multi import MultiFruitPredictor


class ExtraFruitModels:
    def __init__(self,root):
        root=Path(root)
        self.metrics=json.loads((root/'models.json').read_text())
        self.weights={}
        for item in self.metrics:
            with np.load(root/f'{item}.npz',allow_pickle=False) as model:
                self.weights[item]=(model['weights'].copy(),model['bias'].copy())
        options=ort.SessionOptions();options.intra_op_num_threads=2
        self.features=ort.InferenceSession(str(root/'features.onnx'),sess_options=options,providers=['CPUExecutionProvider'])

    def predict(self,bgr,item):
        vector=self.features.run(None,{'image':MultiFruitPredictor.preprocess(bgr)})[0][0]
        weights,bias=self.weights[item];logits=weights@vector+bias
        p=np.exp(logits-logits.max());p/=p.sum()
        classes=self.metrics[item]['classes'];index=int(p.argmax());stage=classes[index]
        uncertain=float(p[index])<.65
        headline={'fresh':'Looks fresh','spoiled':'Visible deterioration','unripe':'Looks unripe',
                  'breaking':'Beginning to ripen','ripe_first_stage':'Looks ripe · first stage',
                  'ripe_second_stage':'Looks ripe · second stage','overripe':'Looks overripe'}[stage]
        basis=('Trained Hass avocado stage classifier; cultivar-specific' if item=='avocado'
               else 'Trained tomato visible-quality classifier')
        return {'fruit_type':item,'stage':'uncertain' if uncertain else stage,'predicted_stage':stage,
                'headline':'Uncertain — retake photo' if uncertain else headline,
                'tip':('Try another view. Cultivar and lighting can change the result.' if uncertain else
                       'This is an appearance-based stage estimate for Hass avocados.' if item=='avocado' else
                       'This assesses visible condition, not ripeness or internal quality.'),
                'score':float(p[index]),'probabilities':dict(zip(classes,map(float,p))),
                'score_label':'Uncalibrated specialist model score','assessment_basis':basis,
                'model':'MobileNetV3-Small features + fruit-specific trained head',
                'days_left':None,'shelf_life_guide':None,'shelf_life_basis':'Not measured',
                'sensors':{'camera':'used','gas':'not used','spectral':'not used'}}
