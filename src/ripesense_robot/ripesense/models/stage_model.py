"""Stage classifier wrapper. In its own module so joblib pickles it under an
importable path (not __main__ when train_fusion runs as a script)."""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV

from ..stages import STAGES

LGB = dict(n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=5,
           subsample=0.9, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0, verbose=-1)

def make_stage_model(y, calibrate=True):
    base = LGBMClassifier(objective="multiclass", **LGB)
    counts = np.bincount(y, minlength=len(STAGES))
    present = counts[counts > 0]
    if calibrate and present.size >= 2 and present.min() >= 6:
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    return base


class StageModel:
    """Wraps a classifier so predict_proba always returns all 5 stage columns,
    even if some stages never appeared in training."""

    def __init__(self, calibrate=True):
        self.calibrate = calibrate
        self.m = None
        self.classes_ = None

    def fit(self, X, y):
        self.m = make_stage_model(y, self.calibrate)
        self.m.fit(X, y)
        self.classes_ = list(self.m.classes_)
        return self

    def predict_proba(self, X):
        p = self.m.predict_proba(X)
        out = np.zeros((p.shape[0], len(STAGES)))
        for j, c in enumerate(self.classes_):
            out[:, int(c)] = p[:, j]
        return out
