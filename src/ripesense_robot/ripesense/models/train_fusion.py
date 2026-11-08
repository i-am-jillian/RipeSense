"""Train the fusion models from data/scans.csv (doc p.12-13).

  python -m ripesense.models.train_fusion [--csv data/scans.csv]

* Stage classifier: LightGBM on all features → 5 stages, probability-calibrated.
* Days-left: LightGBM quantile regressors at p10/p50/p90 (needs days_left_label).
* Per-sensor checks: one small classifier per modality, used only for agreement.
* Modality dropout: every training row is duplicated with each sensor blanked.
* Evaluation is grouped by fruit_id — scans of one fruit never sit in both
  train and test. Reports macro-F1 and MAE in days, plus a feature-importance
  chart for the pitch (proves all three sensors contribute).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import GroupKFold

from .. import config, features
from ..stages import STAGE_INDEX, STAGES
from .featurize import row_features
from .image import Models
from .stage_model import LGB, StageModel

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

QUANTILES = (0.1, 0.5, 0.9)


def build_table(csv_path, cfg) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df = df[df["stage_label"].isin(STAGES)].copy()
    if df.empty:
        raise SystemExit("No rows with a stage_label. Label scans in the /collect page first.")
    models = Models(config.models_dir())
    log.info("pre-trained models used as features: %s", models.summary())
    feats = [row_features(r, cfg, models)[0] for r in df.to_dict("records")]
    X = pd.DataFrame(feats, index=df.index)[features.ALL_FEATURES].astype(float)
    out = pd.concat([df[["scan_id", "fruit_id", "fruit_type", "stage_label", "days_left_label"]], X], axis=1)
    return out


def fruit_onehot(df: pd.DataFrame, fruit_types: list[str]) -> pd.DataFrame:
    return pd.DataFrame({f"is_{t}": (df["fruit_type"] == t).astype(float) for t in fruit_types}, index=df.index)


def dropout_augment(X: pd.DataFrame, y, groups, extra=None):
    """Original rows + one copy per modality with that modality's features NaN."""
    parts_X, parts_y, parts_g, parts_e = [X], [y], [groups], [extra] if extra is not None else []
    for cols in features.MODALITIES.values():
        Xc = X.copy()
        Xc[[c for c in cols if c in Xc]] = np.nan
        parts_X.append(Xc)
        parts_y.append(y)
        parts_g.append(groups)
        if extra is not None:
            parts_e.append(extra)
    Xa = pd.concat(parts_X)
    ya = np.concatenate(parts_y)
    ga = np.concatenate(parts_g)
    ea = np.concatenate(parts_e) if extra is not None else None
    return Xa, ya, ga, ea



CONFORMAL_ALPHA = 0.10   # target: true days-left inside the range 90 % of the time


def conformal(lo, hi, y, alpha=CONFORMAL_ALPHA) -> dict:
    """Split-conformal (CQR, Romano et al. 2019) on out-of-fold quantile predictions:
    widen [p10, p90] by q so that, for a new fruit from the same distribution, the
    true days-left lands inside with probability ≥ 1 - alpha (finite-sample guarantee)."""
    lo, hi, y = map(np.asarray, (lo, hi, y))
    s = np.maximum(lo - y, y - hi)
    n = len(s)
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    q = float(np.sort(s)[k - 1])
    return {"alpha": alpha, "q": q, "n": int(n),
            "raw_coverage": float(np.mean((y >= lo) & (y <= hi))),
            "coverage": float(np.mean((y >= lo - q) & (y <= hi + q)))}


def evaluate(T: pd.DataFrame, Xfull: pd.DataFrame, y, days, groups) -> dict:
    n_groups = len(set(groups))
    if n_groups < 3:
        log.warning("Only %d fruits: skipping grouped CV", n_groups)
        return {}
    gkf = GroupKFold(n_splits=min(5, n_groups))
    pred_s, true_s, pred_d, true_d, lo_d, hi_d = [], [], [], [], [], []
    for tr, te in gkf.split(Xfull, y, groups):
        Xa, ya, _, da = dropout_augment(Xfull.iloc[tr], y[tr], groups[tr], days[tr])
        sm = StageModel(calibrate=False).fit(Xa, ya)
        pred_s += list(sm.predict_proba(Xfull.iloc[te]).argmax(1))
        true_s += list(y[te])
        ok = ~np.isnan(da)
        if ok.sum() >= 15:
            te_ok = ~np.isnan(days[te])
            if te_ok.any():
                Xt = Xfull.iloc[te][te_ok]
                r = {q: LGBMRegressor(objective="quantile", alpha=q, **LGB).fit(Xa[ok], da[ok]) for q in QUANTILES}
                pred_d += list(r[0.5].predict(Xt))
                lo_d += list(r[0.1].predict(Xt))
                hi_d += list(r[0.9].predict(Xt))
                true_d += list(days[te][te_ok])
    m = {"stage_macro_f1": float(f1_score(true_s, pred_s, average="macro")),
         "stage_accuracy": float(np.mean(np.array(true_s) == np.array(pred_s))),
         "n_scans": int(len(y)), "n_fruits": int(n_groups)}
    if pred_d:
        m["days_mae"] = float(mean_absolute_error(true_d, pred_d))
        lo = np.minimum(lo_d, hi_d)
        hi = np.maximum(lo_d, hi_d)
        m["conformal"] = conformal(lo, hi, true_d)
    return m


def log_history(metrics: dict, path) -> list:
    """Append this training run to train_history.json (the 'it learned during the
    event' chart on the store dashboard)."""
    import datetime as _dt
    hist = []
    if path.exists():
        try:
            hist = json.loads(path.read_text())
        except json.JSONDecodeError:
            hist = []
    hist.append({"at": _dt.datetime.now().isoformat(timespec="seconds"),
                 **{k: metrics.get(k) for k in ("n_scans", "n_fruits", "stage_macro_f1", "stage_accuracy", "days_mae")},
                 "coverage": (metrics.get("conformal") or {}).get("coverage")})
    path.write_text(json.dumps(hist, indent=1))
    return hist


def importance_by_modality(model: StageModel, cols: list[str]) -> dict:
    est = model.m
    if isinstance(est, CalibratedClassifierCV):
        imps = np.mean([c.estimator.booster_.feature_importance("gain") for c in est.calibrated_classifiers_], axis=0)
    else:
        imps = est.booster_.feature_importance("gain")
    by_feat = dict(zip(cols, map(float, imps)))
    tot = sum(by_feat.values()) or 1.0
    by_mod = {m: sum(by_feat.get(c, 0) for c in fs) / tot for m, fs in features.MODALITIES.items()}
    by_mod["fruit_type"] = sum(v for k, v in by_feat.items() if k.startswith("is_")) / tot
    return {"by_modality": by_mod, "by_feature": dict(sorted(by_feat.items(), key=lambda kv: -kv[1]))}


def plot_importance(imp: dict, metrics: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    top = list(imp["by_feature"].items())[:15][::-1]
    tot = sum(imp["by_feature"].values()) or 1
    colors = {"camera": "#3B82F6", "spectral": "#10B981", "gas": "#F59E0B"}

    def mod_of(f):
        return next((m for m, fs in features.MODALITIES.items() if f in fs), "fruit_type")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1, 2.2]})
    mods = ["camera", "spectral", "gas"]
    a1.bar(mods, [imp["by_modality"][m] * 100 for m in mods], color=[colors[m] for m in mods])
    a1.set_ylabel("% of total gain")
    a1.set_title("Contribution by sensor")
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    a2.barh([f for f, _ in top], [v / tot * 100 for _, v in top],
            color=[colors.get(mod_of(f), "#9CA3AF") for f, _ in top])
    a2.set_xlabel("% of total gain")
    a2.set_title("Top features")
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)
    sub = (f"macro-F1 {metrics['stage_macro_f1']:.2f} · {metrics['n_scans']} scans · "
           f"{metrics['n_fruits']} fruits (held out by fruit)") if metrics else "no CV (too few fruits)"
    fig.suptitle(f"RipeSense fusion model — {sub}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(csv_path, cfg) -> dict:
    T = build_table(csv_path, cfg)
    fruit_types = sorted(t for t in T["fruit_type"].unique() if t)
    X = pd.concat([T[features.ALL_FEATURES], fruit_onehot(T, fruit_types)], axis=1)
    cols = list(X.columns)
    y = T["stage_label"].map(STAGE_INDEX).to_numpy()
    days = pd.to_numeric(T["days_left_label"], errors="coerce").to_numpy(dtype=float)
    groups = T["fruit_id"].replace("", "unknown").to_numpy()

    metrics = evaluate(T, X, y, days, groups)

    Xa, ya, _, da = dropout_augment(X, y, groups, days)
    stage = StageModel().fit(Xa, ya)

    days_models = {}
    ok = ~np.isnan(da)
    if ok.sum() >= 15 * 4:  # ×4 because of dropout copies
        for q in QUANTILES:
            days_models[q] = LGBMRegressor(objective="quantile", alpha=q, **LGB).fit(Xa[ok], da[ok])
    else:
        log.warning("Too few days_left labels (%d); days range will come from the stage", int(ok.sum() / 4))

    per_sensor = {}
    for m, fcols in features.MODALITIES.items():
        sub = X[fcols].copy()
        if sub.notna().any(axis=1).sum() >= 10:
            sub_cols = fcols + [c for c in cols if c.startswith("is_")]
            per_sensor[m] = (StageModel(calibrate=False).fit(X[sub_cols], y), sub_cols)

    imp = importance_by_modality(stage, cols)
    art = {"version": 1, "columns": cols, "fruit_types": fruit_types, "stage": stage,
           "days": days_models, "per_sensor": per_sensor, "metrics": metrics, "importance": imp,
           "conformal": metrics.get("conformal") if days_models else None}
    out = config.models_dir()
    joblib.dump(art, out / "fusion.joblib")
    (out / "metrics.json").write_text(json.dumps({"metrics": metrics, "importance_by_modality": imp["by_modality"]}, indent=2))
    plot_importance(imp, metrics, out / "feature_importance.png")
    if metrics:
        log_history(metrics, out / "train_history.json")
    return {"metrics": metrics, "importance": imp["by_modality"], "out": str(out)}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    cfg = config.load()
    csv_path = a.csv or (config.data_dir() / "scans.csv")
    r = train(csv_path, cfg)
    m = r["metrics"]
    print("\n=== RipeSense fusion model ===")
    if m:
        print(f"scans {m['n_scans']}  fruits {m['n_fruits']}  (CV grouped by fruit)")
        print(f"stage macro-F1  {m['stage_macro_f1']:.3f}   accuracy {m['stage_accuracy']:.3f}")
        if "days_mae" in m:
            print(f"days-left MAE   {m['days_mae']:.2f} days")
        if m.get("conformal"):
            c = m["conformal"]
            print(f"days range      raw coverage {c['raw_coverage']:.0%} → conformal ±{c['q']:.2f} d → "
                  f"{c['coverage']:.0%} (target {1 - c['alpha']:.0%})")
    print("contribution: " + "  ".join(f"{k} {v*100:.0f}%" for k, v in r["importance"].items()
                                        if not math.isnan(v)))
    print(f"saved → {r['out']}/fusion.joblib, feature_importance.png")


if __name__ == "__main__":
    main()
