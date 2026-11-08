"""Markdown optimizer: pick the discount that maximises what the store gets back
for this item, given *its* predicted days left, instead of a fixed tier.

For each candidate discount d (0-70 %):
    P(sell) = E_D[ 1 - exp(-λ(d) · D) ]      D ~ predicted days-left distribution
    λ(d)    = λ_stage · exp(k · d)           buyers per day (price elasticity)
    value   = P(sell) · price · (1 - d) - P(waste) · disposal
plus a "donate today" option (enhanced tax deduction, no waste), used only when
even the best markdown is unlikely to sell the item before it spoils. The fixed tier
from config.yaml is always one of the candidates, so the optimizer can never
do worse than the tier in expectation.

λ_stage and k are demo assumptions (editable in config.yaml `pricing:`); a
store would fit them from its own sales data. Everything shown is an expected
value from this model, not an observed sale.
"""
from __future__ import annotations

import math

import numpy as np

from .fruit_info import BODY, co2e_kg

DEFAULTS = {
    "daily_buy_rate": {"unripe": 0.30, "ripe": 0.45, "ripe_to_overripe": 0.30, "overripe": 0.12},
    "elasticity": 3.0,            # 50 % off → ×4.5 buyers/day
    "disposal_cost": 0.03,        # $ per item binned
    "donation_value_pct": 0.30,   # enhanced deduction as a share of the full price
    "max_markdown_pct": 70,
    "donate_below_p_sell": 0.6,   # donate only if the best price sells it with < 60 % chance
}


def days_samples(days: dict | None, n: int = 400, seed: int = 0) -> np.ndarray:
    if not days:
        return np.zeros(n)
    p10, p50, p90 = days.get("p10", 0), days.get("p50", 0), days.get("p90", 0)
    z = np.random.default_rng(seed).standard_normal(n)
    d = np.where(z > 0, p50 + (p90 - p50) * z / 1.2816, p50 + (p50 - p10) * z / 1.2816)
    return np.clip(d, 0, None)


def optimise(stage: str, fruit: str, base_price: float, days: dict | None, cfg: dict | None = None,
             tier_pct: float | None = None) -> dict | None:
    if stage in ("spoiled", "uncertain") or not days:
        return None
    p = {**DEFAULTS, **((cfg or {}).get("pricing") or {})}
    lam0 = p["daily_buy_rate"].get(stage, 0.2)
    D = days_samples(days)
    grid = sorted(set(list(range(0, p["max_markdown_pct"] + 1, 5)) + ([round(tier_pct)] if tier_pct is not None else [])))
    if stage == "unripe":   # policy: never discount fruit that hasn't ripened yet
        grid = [0]

    def evaluate(pct):
        lam = lam0 * math.exp(p["elasticity"] * pct / 100.0)
        ps = float(np.mean(1 - np.exp(-lam * D)))
        val = ps * base_price * (1 - pct / 100.0) - (1 - ps) * p["disposal_cost"]
        return {"pct": pct, "price": round(base_price * (1 - pct / 100.0), 2), "p_sell": ps,
                "p_waste": 1 - ps, "value": val}

    opts = [evaluate(x) for x in grid]
    best = max(opts, key=lambda o: o["value"])
    donate = {"pct": None, "price": 0.0, "p_sell": 0.0, "p_waste": 0.0,
              "value": base_price * p["donation_value_pct"]}
    action = "sell"
    # Donate only when even the best markdown probably won't sell it in time.
    if donate["value"] > best["value"] and best["p_sell"] < p["donate_below_p_sell"]:
        best, action = donate, "donate"
    full = evaluate(0)
    tier = evaluate(tier_pct) if tier_pct is not None else full
    grams = BODY.get(fruit, (150, 1, 1))[0]
    return {
        "action": action,
        "markdown_pct": best["pct"], "price": best["price"],
        "p_sell": round(best["p_sell"], 3), "p_waste": round(best["p_waste"], 3),
        "expected_value": round(best["value"], 3),
        "vs_tier": round(best["value"] - tier["value"], 3),
        "vs_no_markdown": round(best["value"] - full["value"], 3),
        "waste_avoided_kg": round(max(0.0, full["p_waste"] - best["p_waste"]) * grams / 1000.0, 4),
        "co2e_avoided_kg": round(co2e_kg(fruit, max(0.0, full["p_waste"] - best["p_waste"]) * grams), 4),
        "curve": [{"pct": o["pct"], "value": round(o["value"], 3), "p_sell": round(o["p_sell"], 3)} for o in opts],
    }
