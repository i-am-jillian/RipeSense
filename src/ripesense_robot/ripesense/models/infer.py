"""Fused prediction + per-sensor agreement + the "Uncertain, rescan" rule (doc p.12).

Three tiers, best available first:
  1. fusion   — LightGBM trained on your own cabinet scans (models/train_fusion.py).
  2. pretrained — before you have scans: the public-data models (stage_model from
     banana photos, image_model fresh/rotten for 9 fruits, spectral_prior from DeepHS).
  3. heuristic — nothing trained at all: a colour rule, flagged `uncalibrated`.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from .. import config, features
from ..stages import HEADLINE, HOME_VERDICT, STAGE_LABEL, STAGES, store_info

log = logging.getLogger(__name__)

# Days-left range (p10, p50, p90) used when no days regressor is trained.
STAGE_DAYS = {"unripe": (5, 6, 8), "ripe": (3, 3.5, 4), "ripe_to_overripe": (1, 1.5, 2),
              "overripe": (0, 0, 1), "spoiled": (0, 0, 0)}
AGREE_MIN_PROB = 0.30
FRUIT_MIN_CONF = 0.40   # image model must be this sure what fruit it is, unless the user said   # a sensor "agrees" if fused stage is its top pick or ≥ this prob


def _missing(feats: dict, modality: str) -> bool:
    return all(math.isnan(feats.get(c, float("nan"))) for c in features.MODALITIES[modality])


class Predictor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.art = None
        self.reload()

    def reload(self):
        p = config.models_dir() / "fusion.joblib"
        self.art = None
        if p.exists():
            try:
                import joblib
                self.art = joblib.load(p)
                log.info("loaded fusion model (%s)", self.art.get("metrics"))
            except Exception as e:  # noqa: BLE001
                log.warning("fusion model failed to load: %s", e)

    @property
    def trained(self) -> bool:
        return self.art is not None

    # ------------------------------------------------------------------
    def predict(self, feats: dict, fruit_type: str, aux: dict | None = None) -> dict:
        thr = self.cfg["scan"]["confidence_threshold"]
        aux = aux or {}
        basis = []
        if self.art:
            probs, days, sensors = self._model(feats, fruit_type)
            uncalibrated, tier = False, "fusion"
        elif aux.get("image") or aux.get("stage") or aux.get("spec_prior"):
            probs, days, sensors, basis = self._pretrained(aux, fruit_type)
            uncalibrated, tier = False, "pretrained"
        else:
            probs, days, sensors = self._heuristic(feats)
            uncalibrated, tier = True, "heuristic"

        k = int(np.argmax(probs))
        fused = STAGES[k]
        conf = float(probs[k])
        disagree = sum(1 for s in sensors.values() if s == "disagree")
        uncertain = conf < thr or disagree >= 2
        stage = "uncertain" if uncertain else fused
        guarantee = None
        if days is None:
            days = STAGE_DAYS[fused]
        elif tier == "fusion" and (self.art or {}).get("conformal"):
            c = self.art["conformal"]
            days = sorted(days)
            days = [days[0] - c["q"], days[1], days[2] + c["q"]]
            guarantee = 1 - c["alpha"]
        p10, p50, p90 = (max(0.0, float(d)) for d in sorted(days))
        return {
            "stage": stage,
            "fused_stage": fused,
            "stage_label": STAGE_LABEL[stage],
            "headline": HEADLINE[stage],
            "home_verdict": HOME_VERDICT[stage],
            "confidence": conf,
            "probs": dict(zip(STAGES, map(float, probs))),
            "days": ({"p10": p10, "p50": p50, "p90": p90, "guarantee": guarantee}
                     if fused != "spoiled" else None),
            "sensors": sensors,
            "uncertain": uncertain,
            "uncalibrated": uncalibrated,
            "tier": tier,
            "basis": basis,
            "store": store_info(stage, fruit_type, self.cfg,
                                days={"p10": p10, "p50": p50, "p90": p90} if fused != "spoiled" else None),
        }

    # ------------------------------------------------------------------
    def _row(self, feats: dict, fruit_type: str, cols: list[str]) -> pd.DataFrame:
        row = {c: feats.get(c, float("nan")) for c in features.ALL_FEATURES}
        for t in self.art["fruit_types"]:
            row[f"is_{t}"] = 1.0 if t == fruit_type else 0.0
        return pd.DataFrame([row])[cols].astype(float)

    def _model(self, feats, fruit_type):
        a = self.art
        X = self._row(feats, fruit_type, a["columns"])
        probs = a["stage"].predict_proba(X)[0]
        days = None
        if a["days"]:
            days = [float(a["days"][q].predict(X)[0]) for q in sorted(a["days"])]
        fused = int(np.argmax(probs))
        sensors = {}
        for m in ("camera", "spectral", "gas"):
            if m not in a["per_sensor"] or _missing(feats, m):
                sensors[m] = "missing"
                continue
            model, cols = a["per_sensor"][m]
            p = model.predict_proba(self._row(feats, fruit_type, cols))[0]
            sensors[m] = "agree" if (int(np.argmax(p)) == fused or p[fused] >= AGREE_MIN_PROB) else "disagree"
        return probs, days, sensors

    def _pretrained(self, aux: dict, fruit_type: str):
        """Combine the public-data models into 5-stage probabilities.

        * image_model gives P(rotten) → the 'spoiled' share.
        * stage_model (banana-trained) splits the rest into unripe/ripe/overripe;
          when ripe and overripe are both likely, the overlap becomes
          'ripe_to_overripe'. It is only used for fruits it was trained on.
        * spectral_prior votes the same way for its fruits (DeepHS), or as a
          weak vote for others.
        * A fruit with no stage evidence at all is called ripe/edible vs spoiled.
        """
        basis: list[str] = []
        img = aux.get("image")
        rot = float(img["rot_prob"]) if img and not math.isnan(img["rot_prob"]) else None
        if img:
            basis.append(f"image model: {img['fruit_type']} {img['fruit_conf']:.0%}, rot {rot:.0%}" if rot is not None
                         else f"image model: {img['fruit_type']}")
        # No recognisable fruit in view (empty chamber, a hand, a face…): don't guess.
        if img and fruit_type in ("", "unknown") and img["fruit_conf"] < FRUIT_MIN_CONF:
            basis.append(f"no fruit recognised (best guess {img['fruit_type']} {img['fruit_conf']:.0%} "
                         f"< {FRUIT_MIN_CONF:.0%}): fill the frame with one fruit and rescan")
            sensors = {"camera": "disagree", "spectral": "missing", "gas": "missing"}
            return np.full(len(STAGES), 1 / len(STAGES)), None, sensors, basis

        votes = []  # (weight, {unripe, ripe, overripe})
        fs = aux.get("fewshot")
        if fs and fs.get("accepted"):
            basis.append(f"taught on this cabinet: {fs['label']} (similarity {fs['sim']:.2f}, "
                         f"{fs['n_examples']} examples)")
            if fs.get("stage") in ("unripe", "ripe", "overripe"):
                q = {"unripe": 0.05, "ripe": 0.05, "overripe": 0.05}
                q[fs["stage"]] = 0.9
                votes.append((1.5, q))
        st = aux.get("stage")
        if st and fruit_type in st["fruits"]:
            votes.append((1.0, st["probs"]))
            basis.append("stage model: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(st["probs"].items(), key=lambda kv: -kv[1])))
        sp = aux.get("spec_prior")
        if sp:
            w = 0.6 if sp["in_domain"] else 0.25
            votes.append((w, sp["probs"]))
            basis.append(f"spectral prior ({'trained on ' + fruit_type if sp['in_domain'] else 'other fruits'}): "
                         + ", ".join(f"{k} {v:.0%}" for k, v in sp["probs"].items()))

        if votes:
            tw = sum(w for w, _ in votes)
            q = {k: sum(w * v.get(k, 0.0) for w, v in votes) / tw for k in ("unripe", "ripe", "overripe")}
        else:
            q = {"unripe": 0.0, "ripe": 1.0, "overripe": 0.0}   # only fresh/rotten known
            basis.append(f"no ripeness-stage model for {fruit_type or 'this fruit'}: fresh vs rotten only")
        n = sum(q.values()) or 1.0
        pu, pr, po = q["unripe"] / n, q["ripe"] / n, q["overripe"] / n
        m = min(pr, po)
        p_rto = 2 * m if m > 0.15 else 0.0
        if p_rto:
            pr, po = pr - m, po - m
        ps = rot if rot is not None else 0.0
        # A clearly fresh fruit can't be spoiled; a clearly rotten one dominates.
        probs = np.array([pu, pr, p_rto, po, 0.0]) * (1 - ps)
        probs[4] = ps
        probs = probs / probs.sum()

        sensors = {"camera": "agree" if (img or st) else "missing", "spectral": "missing", "gas": "missing"}
        if sp:
            fused = STAGES[int(np.argmax(probs))]
            sp_top = max(sp["probs"], key=sp["probs"].get)
            near = {"ripe_to_overripe": ("ripe", "overripe"), "spoiled": ("overripe",)}.get(fused, (fused,))
            sensors["spectral"] = "agree" if sp_top in near else "disagree"
        return probs, None, sensors, basis

    def _heuristic(self, feats):
        """Camera-only colour heuristic (banana-ish). Only for before training."""
        sensors = {"camera": "missing", "spectral": "missing", "gas": "missing"}
        if _missing(feats, "camera"):
            return np.full(len(STAGES), 1 / len(STAGES)), None, sensors
        g = feats.get("green_pct", 0) / 100
        br = feats.get("brown_pct", 0)
        bl = feats.get("black_pct", 0)
        s = 0.45 * (1 - min(1.0, g * 1.5)) + 0.55 * min(1.0, (br + 1.5 * bl) / 35)
        if bl > 25:
            s = max(s, 0.9)
        centers = np.array([0.08, 0.35, 0.55, 0.75, 0.95])
        logits = -((s - centers) ** 2) / (2 * 0.08 ** 2)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        p = 0.75 * p + 0.25 / len(STAGES)   # cap confidence: it's a guess
        sensors["camera"] = "agree"
        return p, None, sensors
