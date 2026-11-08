"""Turn a prediction into what the screen says: home advice or a store markdown."""
from __future__ import annotations

import math

# (min days-left, stage, home verdict, store action, markdown low %, markdown high %)
TIERS = [
    (5, "unripe", "Not ready yet: check again in 2-3 days", "Keep in back stock", 0, 0),
    (3, "ripe", "Ready now: good for 3-4 days", "Front of display", 0, 0),
    (1, "ripe", "Eat today or tomorrow", "'Ripe now' sticker", 25, 40),
    (0, "overripe", "Use today: smoothies or baking", "Bake / juice bin", 50, 70),
]

DEFAULT_TIPS = {
    "unripe": "Store it at room temperature; a paper bag speeds ripening.",
    "ripe": "Perfect to eat now.",
    "overripe": "Great for smoothies or banana bread.",
    "spoiled": "Compost it.",
}


def verdict(pred, fruit: str, mode: str, store_cfg: dict) -> dict:
    out = {"fruit": fruit, "mode": mode, "stage": pred.stage}
    if pred.uncertain:
        out.update(headline="Uncertain: turn the fruit and rescan", action="Rescan", markdown=None, price=None,
                   tip="The sensors did not agree well enough to be sure.")
        return out
    if pred.stage == "spoiled":
        out.update(headline="Spoiled: compost it", action="Pull from shelf", markdown=None, price=None,
                   tip=DEFAULT_TIPS["spoiled"])
        return out
    days = pred.days_mid if math.isfinite(pred.days_mid) else {"unripe": 6, "ripe": 3, "overripe": 0}[pred.stage]
    if pred.stage == "unripe":
        tier = TIERS[0]
    elif pred.stage == "overripe":
        tier = TIERS[3]
    else:
        tier = TIERS[1] if days >= 3 else TIERS[2]
    _, _, home, store, lo, hi = tier
    out["days"] = round(days)
    price = store_cfg["prices"].get(fruit, store_cfg["prices"].get("other", 1.0))
    md = (lo + hi) / 2
    out.update(headline=home, action=store, markdown=[lo, hi] if hi else None,
               price=round(price * (1 - md / 100), 2), list_price=price, currency=store_cfg["currency"],
               tip=DEFAULT_TIPS[pred.stage])
    return out
