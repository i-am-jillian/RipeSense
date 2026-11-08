"""Ripening forecast ('perfect on Saturday ~3 pm'): a small physics-informed
digital twin per fruit, updated with every scan (Bayesian).

Model. Latent ripeness q (0 unripe · 1 ripe · 1.5 ripe→overripe · 2 overripe ·
3 spoiled) rises at a rate r per "20 °C-equivalent day". Temperature enters
through Q10 = 2.3 (ripening rate roughly doubles per +10 °C, the standard
postharvest rule): effective time τ = Σ Δt · Q10^((T-20)/10).

    q(τ) = q_now + r · τ

Prior on r per fruit type (typical counter ripening times); prior on q_now from
the latest scan. Each scan of the same fruit_id is a noisy observation (noise
grows when the model was unsure). Posterior is Gaussian in (q_now, r): closed
form, instant on a Pi. Sampling the posterior gives the shaded band and the
p10/p50/p90 time of each milestone. One scan → wide band; 3-4 scans over a
day or two → tight band. That narrowing is the demo.

What-ifs change the future temperature or the ethylene exposure.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from .fruit_info import ETHYLENE, STAGE_SCORE

Q10 = 2.3
# Prior ripening rate at 20 °C, score units/day (unripe→spoiled ≈ 3 units).
RATE = {"banana": 0.33, "avocado": 0.38, "mango": 0.30, "tomato": 0.25, "pear": 0.28,
        "kiwi": 0.20, "apple": 0.07, "strawberry": 0.70, "grape": 0.15, "guava": 0.35,
        "orange": 0.06, "lime": 0.07, "pomegranate": 0.05, "jujube": 0.20, "default": 0.20}
RATE_LOG_SD = 0.35                       # ±35 % prior uncertainty on the rate
CHILL_SENSITIVE = {"banana", "avocado", "mango", "tomato", "guava"}
MILESTONES = {"ripe": 0.6, "peak_end": 1.75, "spoiled": 2.5}   # ripe ≈ where "ripe" overtakes "unripe"
SCENARIOS = {
    "counter": {"temp": None, "label": "On the counter"},
    "fridge": {"temp": 4.0, "label": "In the fridge (4 °C)"},
    "warm": {"temp": 27.0, "label": "Warm spot (27 °C)"},
    "bag": {"temp": None, "ethylene": 1.4, "label": "Paper bag with a banana"},
}
DEFAULT_TEMP = 21.0


def temp_factor(t_c: float) -> float:
    return Q10 ** ((t_c - 20.0) / 10.0)


def _obs_sd(conf: float) -> float:
    return 0.20 + 0.45 * (1.0 - max(0.0, min(1.0, conf)))


def posterior(history: list[dict], fruit: str):
    """history: [{"t": datetime, "q": score, "conf": 0-1, "temp": °C|None}] (any order).
    Returns (mean[q_now, r], cov 2x2, t_now)."""
    h = sorted(history, key=lambda x: x["t"])
    t_now = h[-1]["t"]
    # Effective (temperature-weighted) days of each scan relative to the latest one.
    taus, tau = [0.0] * len(h), 0.0
    for i in range(len(h) - 2, -1, -1):
        dt = (h[i + 1]["t"] - h[i]["t"]).total_seconds() / 86400.0
        temp = h[i + 1].get("temp") or h[i].get("temp") or DEFAULT_TEMP
        tau -= dt * temp_factor(temp)
        taus[i] = tau
    r0 = RATE.get(fruit, RATE["default"])
    mu0 = np.array([h[-1]["q"], r0])
    cov0 = np.diag([0.6 ** 2, (RATE_LOG_SD * r0) ** 2])
    A = np.linalg.inv(cov0)
    b = A @ mu0
    for o, ta in zip(h, taus):
        x = np.array([1.0, ta])
        w = 1.0 / _obs_sd(o.get("conf", 0.7)) ** 2
        A += w * np.outer(x, x)
        b += w * x * o["q"]
    cov = np.linalg.inv(A)
    return cov @ b, cov, t_now


def forecast(history: list[dict], fruit: str, scenario: str = "counter", now: datetime | None = None,
             temp_now: float | None = None, n: int = 600, horizon_days: float = 12.0) -> dict:
    fruit = (fruit or "default").lower()
    mean, cov, t_last = posterior(history, fruit)
    now = now or t_last
    sc = SCENARIOS.get(scenario, SCENARIOS["counter"])
    last_temp = next((o.get("temp") for o in sorted(history, key=lambda x: x["t"], reverse=True) if o.get("temp")), None)
    T = sc["temp"] if sc.get("temp") is not None else (temp_now or last_temp or DEFAULT_TEMP)
    eth = sc.get("ethylene", 1.0) if ETHYLENE.get(fruit, (1, 1))[1] >= 2 else 1.0
    rng = np.random.default_rng(0)
    S = rng.multivariate_normal(mean, cov, n)
    q_now = S[:, 0] + S[:, 1] * (now - t_last).total_seconds() / 86400.0 * temp_factor(last_temp or DEFAULT_TEMP)
    tf = temp_factor(T)
    if fruit in CHILL_SENSITIVE and T < 12:
        tf = max(tf, 0.5)   # chilling injury: below ~12 °C these fruit decline instead of keeping
    r = np.clip(S[:, 1], 0.02 * RATE.get(fruit, 0.2), None) * tf
    r_eth = r * eth   # ethylene speeds ripening, not rotting, so apply below ripe_end only

    def days_to(level):
        d1 = np.clip((np.minimum(level, 1.75) - q_now) / r_eth, 0, None)
        q_mid = np.maximum(q_now, np.minimum(level, 1.75))
        return np.where(level <= 1.75, d1, d1 + np.clip((level - q_mid) / r, 0, None))

    out = {"fruit": fruit, "scenario": scenario, "scenario_label": sc["label"], "temp_c": round(T, 1),
           "n_scans": len(history), "now": now.isoformat(timespec="minutes"),
           "q_now": round(float(np.median(q_now)), 2), "rate_per_day": round(float(np.median(r)), 3),
           "milestones": {}}
    for name, lvl in MILESTONES.items():
        d = days_to(lvl)
        p10, p50, p90 = (float(np.percentile(d, p)) for p in (10, 50, 90))
        out["milestones"][name] = {"days": [round(p10, 2), round(p50, 2), round(p90, 2)],
                                   "at": (now + timedelta(days=p50)).isoformat(timespec="minutes"),
                                   "reached": p90 <= 0.01}
    rp, pe = out["milestones"]["ripe"], out["milestones"]["peak_end"]
    out["peak_window"] = {"start": rp["at"], "end": pe["at"],
                          "start_days": rp["days"][1], "end_days": pe["days"][1]}
    out["days_left"] = out["milestones"]["spoiled"]["days"]
    # Band for the chart: q(t) percentiles at 0.25-day steps.
    ts = np.arange(0, horizon_days + 1e-9, 0.25)
    qs = []
    for d in ts:
        ripen = np.minimum(q_now + r_eth * d, np.maximum(q_now, 1.75))
        t_at_175 = np.clip((1.75 - q_now) / r_eth, 0, None)
        q = np.where(q_now + r_eth * d <= 1.75, ripen, np.maximum(q_now, 1.75) + r * np.clip(d - t_at_175, 0, None))
        qs.append(np.percentile(np.clip(q, 0, 3.2), [10, 50, 90]))
    qs = np.array(qs)
    out["curve"] = {"days": ts.round(2).tolist(), "p10": qs[:, 0].round(3).tolist(),
                    "p50": qs[:, 1].round(3).tolist(), "p90": qs[:, 2].round(3).tolist()}
    notes = []
    if scenario == "fridge" and fruit in CHILL_SENSITIVE:
        notes.append(f"{fruit.capitalize()} is chill-sensitive: the skin may darken in the fridge even though the flesh keeps.")
    if eth > 1 and scenario == "bag":
        notes.append("Ethylene from the banana speeds ripening; it does not slow rotting afterwards.")
    if len(history) == 1:
        notes.append("One scan so far: band is wide. Scan the same fruit again tomorrow to tighten it.")
    out["notes"] = notes
    return out


def whatifs(history, fruit, now=None, temp_now=None) -> dict:
    """Days-left (p50) and peak window under each scenario, for the what-if buttons."""
    res = {}
    for k in SCENARIOS:
        f = forecast(history, fruit, k, now=now, temp_now=temp_now, n=300, horizon_days=0)
        res[k] = {"label": f["scenario_label"], "days_left": f["days_left"],
                  "peak_window": f["peak_window"], "notes": f["notes"]}
    base = res["counter"]["days_left"][1]
    for k in res:
        res[k]["delta_days"] = round(res[k]["days_left"][1] - base, 1)
    return res


def history_from_rows(rows: list[dict]) -> list[dict]:
    """Store rows (one fruit_id) → forecast observations. Uncertain scans are skipped."""
    hist = []
    for r in rows:
        st = r.get("pred_stage") or ""
        if st not in STAGE_SCORE:
            continue
        try:
            q = float(r["pred_score"]) if r.get("pred_score") not in (None, "") else STAGE_SCORE[st]
            conf = float(r.get("pred_conf") or 0.6)
            t = datetime.fromisoformat(r["timestamp"])
        except (ValueError, KeyError, TypeError):
            continue
        temp = None
        try:
            temp = float(r["temp"]) if r.get("temp") not in (None, "") else None
        except ValueError:
            pass
        if temp is not None and math.isnan(temp):
            temp = None
        hist.append({"t": t, "q": q, "conf": conf, "temp": temp})
    return hist
