"""Spectral prior pre-trained on DeepHS Fruit (doc p.14): hyperspectral scans of
fruit with measured ripeness, with each scan's mean spectrum averaged down to
the AS7341's channels so it looks like what our sensor would read.

  python -m ripesense.models.prep_deephs datasets/raw/deephs/Mango.zip ...  → datasets/deephs_as7341.csv
  python -m ripesense.models.spectral_prior --csv datasets/deephs_as7341.csv

Output models/artifacts/spectral_prior.joblib. At run time it turns a scan's
reflectance into P(unripe/ripe/overripe) and a 0–2 score, which becomes the
fusion feature `spec_prior_score` and the spectral vote in camera-first mode.

Caveat: DeepHS fruit (avocado, kiwi, mango, kaki, papaya) are not bananas,
and the Specim camera is not an AS7341 under our LEDs. Treat this as a prior
that your own scans will correct, not as a finished banana spectral model.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import warnings
from pathlib import Path

import numpy as np

from .. import config, features
from ..sim.fruit import SPECTRAL_CHANNELS

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

# AS7341 channel centres / FWHM (nm), approximated as Gaussians from the datasheet.
AS7341_BANDS = {"f1_415": (415, 26), "f2_445": (445, 30), "f3_480": (480, 36), "f4_515": (515, 39),
                "f5_555": (555, 39), "f6_590": (590, 40), "f7_630": (630, 50), "f8_680": (680, 52),
                "nir": (910, 40)}
PRIOR_CLASSES = ["unripe", "ripe", "overripe"]
MIN_GAIN = 0.05   # required CV accuracy gain over the majority-class baseline
SCORE = {"unripe": 0.0, "ripe": 1.0, "overripe": 2.0}
SPEC_FEATS = [f for f in features.SPECTRAL_FEATURES if f != "spec_prior_score"]


def to_as7341(wavelengths: np.ndarray, spectrum: np.ndarray) -> dict:
    """Average a reflectance spectrum into AS7341 channels with Gaussian responses."""
    out = {}
    for ch, (c, fwhm) in AS7341_BANDS.items():
        sd = fwhm / 2.3548
        w = np.exp(-0.5 * ((wavelengths - c) / sd) ** 2)
        if w.sum() < 1e-6 or wavelengths.max() < c + fwhm / 2 or wavelengths.min() > c - fwhm / 2:
            out[ch] = float("nan")
        else:
            out[ch] = float((w * spectrum).sum() / w.sum())
    vis = (wavelengths >= 400) & (wavelengths <= 950)
    out["clear"] = float(spectrum[vis].mean()) if vis.any() else float("nan")
    return out


class SpectralPrior:
    def __init__(self, art: dict):
        self.art = art
        self.fruits = art["fruits"]

    @classmethod
    def load(cls, models_dir: Path):
        p = models_dir / "spectral_prior.joblib"
        if not p.exists():
            return None
        try:
            import joblib
            art = joblib.load(p)
        except Exception as e:  # noqa: BLE001
            log.warning("spectral prior failed to load: %s", e)
            return None
        m = art.get("metrics", {})
        # Only let it vote if it beat "always predict the commonest class" in
        # fruit-grouped CV by a real margin. (DeepHS mango+kaki at AS7341
        # resolution did not: 51% vs a 51% baseline — see spectral_prior_metrics.json.)
        if m.get("cv_accuracy", 0) < m.get("majority_baseline", 1) + MIN_GAIN:
            log.warning("spectral prior not used: CV accuracy %.2f vs baseline %.2f",
                        m.get("cv_accuracy", float("nan")), m.get("majority_baseline", float("nan")))
            return None
        return cls(art)

    def predict(self, refl: dict | None, fruit_type: str = "") -> dict | None:
        if not refl or all(math.isnan(v) for v in refl.values() if isinstance(v, float)):
            return None
        f = features.spectral_features(refl)
        row = [f.get(k, float("nan")) for k in SPEC_FEATS]
        row += [1.0 if fruit_type == t else 0.0 for t in self.fruits]
        p = self.art["model"].predict_proba(np.array([row], float))[0]
        probs = {c: float(p[i]) for i, c in enumerate(self.art["classes"])}
        return {"probs": probs, "score": float(sum(SCORE[c] * v for c, v in probs.items())),
                "in_domain": fruit_type in self.fruits}


def train(csv_path: Path) -> dict:
    import joblib
    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold

    df = pd.read_csv(csv_path)
    df = df[df["label"].isin(PRIOR_CLASSES)].reset_index(drop=True)
    fruits = sorted(df["fruit"].unique())
    X = []
    for r in df.to_dict("records"):
        f = features.spectral_features({c: r.get(c, float("nan")) for c in SPECTRAL_CHANNELS})
        X.append([f.get(k, float("nan")) for k in SPEC_FEATS] + [1.0 if r["fruit"] == t else 0.0 for t in fruits])
    X = np.array(X, float)
    classes = [c for c in PRIOR_CLASSES if c in set(df["label"])]
    y = df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    groups = df["fruit_id"].astype(str).to_numpy()
    params = dict(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=4,
                  class_weight="balanced", reg_lambda=1.0, verbose=-1)

    n_groups = len(set(groups))
    metrics = {"n": int(len(y)), "fruits": fruits, "per_class": {c: int((y == i).sum()) for i, c in enumerate(classes)}}
    if n_groups >= 5:
        pred = np.zeros(len(y), int)
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            pred[te] = LGBMClassifier(**params).fit(X[tr], y[tr]).predict(X[te])
        metrics["cv_macro_f1"] = float(f1_score(y, pred, average="macro"))
        metrics["cv_accuracy"] = float((pred == y).mean())
        metrics["majority_baseline"] = float(np.bincount(y).max() / len(y))
        for t in fruits:
            m = df["fruit"].to_numpy() == t
            metrics[f"cv_accuracy_{t}"] = float((pred[m] == y[m]).mean())
    model = LGBMClassifier(**params).fit(X, y)
    art = {"model": model, "classes": classes, "fruits": fruits, "metrics": metrics, "source": str(csv_path)}
    out = config.models_dir()
    joblib.dump(art, out / "spectral_prior.joblib")
    (out / "spectral_prior_metrics.json").write_text(json.dumps(metrics, indent=1))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="datasets/deephs_as7341.csv")
    a = ap.parse_args()
    m = train(config.path(a.csv))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
