"""Fusion model: stage (unripe / ripe / overripe / spoiled), days-left range, confidence.

Before you have trained a model, HeuristicModel gives a rough answer from colour,
spectral indices and gas, so the demo works on day one. After `ripesense train`,
FusionModel (gradient-boosted trees on every sensor's features) takes over.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .features import FEATURES, FRUITS, MODALITIES, vector

log = logging.getLogger(__name__)
STAGES = ["unripe", "ripe", "overripe", "spoiled"]
NAN = float("nan")


@dataclass
class Prediction:
    stage: str
    probs: dict
    days_low: float
    days_mid: float
    days_high: float
    confidence: float
    agreement: dict = field(default_factory=dict)   # modality -> stage it votes for
    uncertain: bool = False
    mode: str = "heuristic"                         # "heuristic" or "model"
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _present(features: dict, names) -> bool:
    return any(isinstance(features.get(n), (int, float)) and math.isfinite(features.get(n)) for n in names)


def _stage_of(r: float) -> str:
    return "unripe" if r < 0.22 else "ripe" if r < 0.55 else "overripe" if r < 0.82 else "spoiled"


class HeuristicModel:
    """Each sensor gives a ripeness score 0..1 (0 = green, 1 = rotten); scores are averaged."""
    SHELF_DAYS = {"banana": 7, "avocado": 6, "tomato": 8, "apple": 20, "mango": 7, "other": 7}

    def votes(self, f: dict) -> dict:
        v = {}
        if _present(f, ["c_green_frac"]):
            v["camera"] = float(np.clip(0.35 * (1 - f["c_green_frac"]) +
                                        0.75 * min(1.0, 1.5 * f["c_dark_frac"] + f["c_brown_frac"]) +
                                        0.02 * f.get("c_spot_density", 0), 0, 1))
        if _present(f, ["s_ripen"]):
            r = 0.35 - 0.5 * f["s_ripen"]
            if math.isfinite(f.get("s_rclear", NAN)) and f.get("s_calibrated") == 1.0:
                r += 0.4 * max(0.0, 0.45 - f["s_rclear"])
            v["spectral"] = float(np.clip(r, 0, 1))
        if _present(f, ["g_rise"]):
            v["gas"] = float(np.clip(math.sqrt(max(f["g_rise"], 0) / 0.9), 0, 1))
        return v

    def predict(self, f: dict, fruit: str) -> Prediction:
        votes = self.votes(f)
        if not votes:
            return Prediction("ripe", {}, NAN, NAN, NAN, 0.0, {}, True, "heuristic", ["no sensor data"])
        w = {"camera": 0.45, "spectral": 0.35, "gas": 0.2}
        r = sum(w[k] * x for k, x in votes.items()) / sum(w[k] for k in votes)
        stage = _stage_of(r)
        spread = float(np.std(list(votes.values()))) if len(votes) > 1 else 0.15
        conf = float(np.clip(0.9 - 2.0 * spread, 0.3, 0.9))
        shelf = self.SHELF_DAYS.get(fruit, 7)
        mid = max(0.0, (0.85 - r) / 0.85 * shelf)
        agree = {k: _stage_of(x) for k, x in votes.items()}
        probs = {s: (conf if s == stage else (1 - conf) / 3) for s in STAGES}
        return Prediction(stage, probs, round(max(0.0, mid * 0.6), 1), round(mid, 1), round(mid * 1.4 + 0.5, 1),
                          conf, agree, conf < 0.6, "heuristic", ["Heuristic mode: train a model for better accuracy"])


class FusionModel:
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
        self.clf = None
        self.reg = {}
        self.modality_clf = {}
        self.metrics: dict = {}
        self.classes_: list[str] = []
        self.cols: list[int] = list(range(len(FEATURES)))   # features that had data at training time

    # ------------------------------------------------------------ training
    @staticmethod
    def _augment(X, y, d, g):
        """Modality dropout: copies of each row with one sensor blanked, so a missing sensor still works."""
        Xs, ys, ds, gs = [X], [y], [d], [g]
        for names in MODALITIES.values():
            Xc = X.copy()
            for n in names:
                Xc[:, FEATURES.index(n)] = np.nan
            Xs.append(Xc); ys.append(y); ds.append(d); gs.append(g)
        return np.vstack(Xs), np.concatenate(ys), np.concatenate(ds), np.concatenate(gs)

    @property
    def _cat(self):
        return np.array([FEATURES[i] == "fruit_code" for i in self.cols])

    def _base_clf(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                                              l2_regularization=0.1, categorical_features=self._cat, random_state=0)

    def _fit_core(self, X, y, d):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingRegressor
        counts = {s: int((y == s).sum()) for s in set(y)}
        clf = self._base_clf()
        if min(counts.values()) >= 6:
            clf = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
        clf.fit(X, y)
        reg = {}
        ok = np.isfinite(d)
        if ok.sum() >= 10:
            for q in (0.1, 0.5, 0.9):
                r = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=200, learning_rate=0.08,
                                                  max_leaf_nodes=15, categorical_features=self._cat, random_state=0)
                reg[q] = r.fit(X[ok], d[ok])
        return clf, reg

    def fit(self, rows: list[dict]) -> dict:
        from sklearn.metrics import f1_score, mean_absolute_error
        from sklearn.model_selection import GroupKFold
        X = np.array([vector(r["features"]) for r in rows])
        y = np.array([r["stage"] for r in rows])
        d = np.array([r.get("days_left", NAN) if r.get("days_left") is not None else NAN for r in rows], float)
        g = np.array([str(r.get("fruit_id") or i) for i, r in enumerate(rows)])
        if len(set(y)) < 2:
            raise ValueError("need at least two different stages to train")
        fc = FEATURES.index("fruit_code")
        self.cols = [i for i in range(len(FEATURES)) if i == fc or np.isfinite(X[:, i]).sum() >= 2]
        dropped = [FEATURES[i] for i in range(len(FEATURES)) if i not in self.cols]
        if dropped:
            self.metrics["unused_features"] = dropped

        # honest evaluation: whole fruits held out (never the same banana in train and test)
        n_groups = len(set(g))
        if n_groups >= 3:
            preds, truth, dp, dt = [], [], [], []
            for tr, te in GroupKFold(n_splits=min(5, n_groups)).split(X, y, g):
                if len(set(y[tr])) < 2:
                    continue
                Xa, ya, da, _ = self._augment(X[tr], y[tr], d[tr], g[tr])
                clf, reg = self._fit_core(Xa[:, self.cols], ya, da)
                Xte = X[te][:, self.cols]
                preds += list(clf.predict(Xte)); truth += list(y[te])
                if 0.5 in reg:
                    ok = np.isfinite(d[te])
                    dp += list(reg[0.5].predict(Xte[ok])); dt += list(d[te][ok])
            if truth:
                self.metrics["stage_macro_f1"] = round(float(f1_score(truth, preds, average="macro")), 3)
                self.metrics["stage_accuracy"] = round(float(np.mean(np.array(truth) == np.array(preds))), 3)
            if dt:
                self.metrics["days_mae"] = round(float(mean_absolute_error(dt, dp)), 2)
        self.metrics.update({"rows": len(rows), "fruits": n_groups,
                             "per_stage": {s: int((y == s).sum()) for s in STAGES}})

        Xa, ya, da, _ = self._augment(X, y, d, g)
        self.clf, self.reg = self._fit_core(Xa[:, self.cols], ya, da)
        self.classes_ = list(self.clf.classes_)
        for name, cols in MODALITIES.items():   # single-sensor models, used only to measure agreement
            idx = [fc] + [FEATURES.index(c) for c in cols if FEATURES.index(c) in self.cols]
            Xm = X[:, idx]
            if len(idx) > 1 and np.isfinite(Xm[:, 1:]).any(axis=1).sum() >= 8:
                from sklearn.ensemble import HistGradientBoostingClassifier
                m = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, random_state=0,
                                                   categorical_features=np.array([True] + [False] * (len(idx) - 1)))
                self.modality_clf[name] = (idx, m.fit(Xm, y))
        try:  # which features carry the decision (for the pitch slide)
            from sklearn.inspection import permutation_importance
            Xc = X[:, self.cols]
            base = self._base_clf().fit(Xc, y)
            imp = permutation_importance(base, Xc, y, n_repeats=5, random_state=0)
            order = np.argsort(-imp.importances_mean)[:10]
            self.metrics["top_features"] = [(FEATURES[self.cols[i]], round(float(imp.importances_mean[i]), 3))
                                            for i in order]
        except Exception as e:  # pragma: no cover
            log.info("feature importance skipped: %s", e)
        return self.metrics

    # ------------------------------------------------------------ inference
    def predict(self, f: dict, fruit: str) -> Prediction:
        x = vector(f)[None]
        xc = x[:, self.cols]
        p = self.clf.predict_proba(xc)[0]
        probs = {c: float(pi) for c, pi in zip(self.classes_, p)}
        stage = max(probs, key=probs.get)
        conf = probs[stage]
        if self.reg:
            q = sorted(float(max(0.0, self.reg[k].predict(xc)[0])) for k in (0.1, 0.5, 0.9))
            lo, mid, hi = q
        else:
            lo = mid = hi = NAN
        agree = {}
        for name, (idx, m) in self.modality_clf.items():
            if _present(f, MODALITIES[name]):
                agree[name] = str(m.predict(x[:, idx])[0])
        n_agree = sum(1 for s in agree.values() if s == stage)
        notes = []
        uncertain = conf < self.min_confidence
        if len(agree) >= 2 and n_agree < 2:
            uncertain = True
            notes.append("Sensors disagree")
        return Prediction(stage, probs, round(lo, 1), round(mid, 1), round(hi, 1), round(conf, 3), agree,
                          uncertain, "model", notes)

    def save(self, path: Path):
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"version": 1, "features": FEATURES, "model": self}, path)

    @staticmethod
    def load(path: Path) -> "FusionModel":
        import joblib
        blob = joblib.load(path)
        if blob["features"] != FEATURES:
            raise ValueError("model was trained on a different feature list; retrain it")
        return blob["model"]


def load_predictor(cfg):
    p = cfg.path(cfg["model"]["path"])
    if p.exists():
        try:
            m = FusionModel.load(p)
            m.min_confidence = cfg["model"]["min_confidence"]
            log.info("loaded trained model %s", p)
            return m
        except Exception as e:
            log.error("could not load model %s (%s); using heuristic", p, e)
    return HeuristicModel()


__all__ = ["STAGES", "FRUITS", "Prediction", "HeuristicModel", "FusionModel", "load_predictor"]
