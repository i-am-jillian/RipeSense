"""Ripeness stages, verdict text and retail markdown lookups (doc p.15)."""
from __future__ import annotations

STAGES = ["unripe", "ripe", "ripe_to_overripe", "overripe", "spoiled"]
STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}

STAGE_LABEL = {
    "unripe": "Unripe",
    "ripe": "Ripe",
    "ripe_to_overripe": "Ripe to overripe",
    "overripe": "Overripe",
    "spoiled": "Spoiled",
    "uncertain": "Uncertain",
}

HOME_VERDICT = {
    "unripe": "Not ready yet, check again in 2–3 days",
    "ripe": "Ready now, good for 3–4 days",
    "ripe_to_overripe": "Eat today or tomorrow",
    "overripe": "Use today: smoothies, baking",
    "spoiled": "Compost it",
    "uncertain": "Rescan (turn the fruit)",
}

# Short headline for the large verdict text on the 480x320 screen.
HEADLINE = {
    "unripe": "Not ready yet",
    "ripe": "Ready now",
    "ripe_to_overripe": "Eat within 2 days",
    "overripe": "Use today",
    "spoiled": "Compost it",
    "uncertain": "Uncertain — rescan",
}


def stage_from_days(days: float) -> str:
    """Days-left (median) → stage, per the doc's verdict table."""
    if days >= 5:
        return "unripe"
    if days >= 3:
        return "ripe"
    if days >= 1:
        return "ripe_to_overripe"
    return "overripe"


def store_info(stage: str, fruit_type: str, cfg: dict, days: dict | None = None) -> dict:
    """Store action + suggested price. The fixed tier from config is the
    fallback; with a days-left range the markdown optimizer (pricing.py) picks
    the price that maximises expected recovery for this item."""
    if stage == "uncertain":
        return {"store_action": "Rescan", "markdown_pct": None, "price": None, "base_price": None}
    tier = cfg["markdown"].get(stage, {})
    prices = cfg.get("base_prices", {})
    base = float(prices.get(fruit_type, prices.get("default", 1.0)))
    pct = tier.get("markdown_pct")
    price = None
    mid = None
    if pct is not None:
        mid = (pct[0] + pct[1]) / 2.0
        price = round(base * (1 - mid / 100.0), 2)
    out = {
        "store_action": tier.get("store_action", ""),
        "markdown_pct": pct,
        "price": price,
        "base_price": base,
        "tier_price": price,
    }
    if days and stage != "spoiled" and (cfg.get("pricing") or {}).get("optimise", True):
        from .pricing import optimise
        opt = optimise(stage, fruit_type, base, days, cfg, tier_pct=mid)
        if opt:
            out["optimiser"] = opt
            if opt["action"] == "donate":
                out.update(store_action="Donate today (food bank)", markdown_pct=None, price=None)
            else:
                out.update(markdown_pct=[opt["markdown_pct"], opt["markdown_pct"]], price=opt["price"])
                if opt["markdown_pct"] and not tier.get("markdown_pct", [0, 0])[1]:
                    out["store_action"] = f"Mark down {opt['markdown_pct']}%"
    return out
