"""Fruit profile card: what the fruit *is* right now, not just its stage.

* Measured: length/width from the camera, volume from depth → estimated weight.
* Estimated: sweetness (°Brix) and sugar vs starch from the ripeness score;
  calories, sugar, fibre, vitamin C, potassium scaled to the estimated edible weight.
* Advice: best uses for this stage, storage, and ethylene (which fruit not to
  keep it next to).

Nutrition per 100 g edible portion is from USDA FoodData Central (SR Legacy,
raw), rounded. Brix ranges, densities and edible fractions are typical
literature values. All of it is an estimate and is labelled as such on screen.
"""
from __future__ import annotations

import math

# kcal, sugar g, fibre g, vitamin C mg, potassium mg  (per 100 g edible)
NUTRITION = {
    "banana": (89, 12.2, 2.6, 8.7, 358), "apple": (52, 10.4, 2.4, 4.6, 107),
    "orange": (47, 9.4, 2.4, 53.2, 181), "lime": (30, 1.7, 2.8, 29.1, 102),
    "guava": (68, 8.9, 5.4, 228.0, 417), "pomegranate": (83, 13.7, 4.0, 10.2, 236),
    "grape": (69, 15.5, 0.9, 3.2, 191), "strawberry": (32, 4.9, 2.0, 58.8, 153),
    "jujube": (79, None, None, 69.0, 250), "mango": (60, 13.7, 1.6, 36.4, 168),
    "avocado": (160, 0.7, 6.7, 10.0, 485), "tomato": (18, 2.6, 1.2, 13.7, 237),
    "kiwi": (61, 9.0, 3.0, 92.7, 312), "pear": (57, 9.8, 3.1, 4.3, 116),
}
# typical whole weight g, edible fraction, density g/cm³
BODY = {
    "banana": (180, 0.65, 0.94), "apple": (182, 0.90, 0.82), "orange": (180, 0.73, 0.90),
    "lime": (67, 0.84, 0.95), "guava": (55, 1.0, 1.0), "pomegranate": (282, 0.52, 0.95),
    "grape": (5, 1.0, 1.05), "strawberry": (12, 0.94, 0.95), "jujube": (15, 0.90, 1.0),
    "mango": (336, 0.69, 1.02), "avocado": (200, 0.74, 1.0), "tomato": (123, 1.0, 1.0),
    "kiwi": (75, 0.86, 1.05), "pear": (178, 0.90, 1.0),
}
# °Brix at unripe / ripe / overripe (fruit that sweeten as they ripen)
BRIX = {
    "banana": (3, 19, 22), "mango": (8, 15, 18), "tomato": (3.5, 5, 5.5), "kiwi": (6, 14, 16),
    "pear": (10, 13, 14), "apple": (11, 13, 14), "orange": (9, 11, 12), "lime": (7, 8, 8),
    "guava": (7, 10, 11), "pomegranate": (14, 16, 17), "grape": (14, 18, 20),
    "strawberry": (6, 8, 9), "jujube": (18, 25, 28),
}
# Sugar g/100 g at unripe / ripe / overripe where starch converts (banana, mango).
SUGAR_CURVE = {"banana": (2.0, 12.2, 15.0), "mango": (6.0, 13.7, 15.0)}
STARCH_CURVE = {"banana": (20.0, 2.0, 1.0), "mango": (8.0, 1.0, 0.5)}
# Ethylene: (production 0-3, sensitivity 0-3), after UC Davis Postharvest tables.
ETHYLENE = {
    "apple": (3, 2), "banana": (2, 3), "avocado": (3, 3), "tomato": (2, 2), "mango": (2, 2),
    "pear": (3, 3), "kiwi": (1, 3), "guava": (2, 2), "orange": (0, 1), "lime": (0, 1),
    "grape": (0, 1), "strawberry": (0, 1), "pomegranate": (0, 1), "jujube": (1, 1),
}
# kg CO2e per kg of food (Poore & Nemecek 2018, via Our World in Data).
CO2E_PER_KG = {"banana": 0.86, "apple": 0.43, "orange": 0.39, "lime": 0.39, "tomato": 2.09,
               "grape": 1.53, "strawberry": 1.53, "default": 1.05}

USES = {
    "banana": {"unripe": ["Cook it: green-banana curry or chips", "High in resistant starch (gut-friendly, low GI)"],
               "ripe": ["Eat now", "Slice onto cereal or toast"],
               "ripe_to_overripe": ["Smoothies", "Freeze in slices for later"],
               "overripe": ["Banana bread or pancakes", "Sweetest now: no added sugar needed"],
               "spoiled": ["Compost it"]},
    "avocado": {"unripe": ["Wait 2–4 days", "Paper bag with a banana to speed it up"],
                "ripe": ["Guacamole or toast today", "Fridge to hold 2 more days"],
                "overripe": ["Cut away brown parts: smoothies or dressings"], "spoiled": ["Compost it"]},
    "mango": {"unripe": ["Green-mango salad or pickle"], "ripe": ["Eat fresh, lassi"],
              "overripe": ["Smoothies, sorbet, chutney"], "spoiled": ["Compost it"]},
    "tomato": {"unripe": ["Fried green tomatoes", "Leave stem-up on the counter to ripen"],
               "ripe": ["Salads, sandwiches"], "overripe": ["Cook down into sauce or soup"],
               "spoiled": ["Compost it"]},
    "apple": {"ripe": ["Eat fresh, salads"], "overripe": ["Apple sauce, crumble, baking"], "spoiled": ["Compost it"]},
    "pear": {"unripe": ["Poach it"], "ripe": ["Eat now, with cheese"], "overripe": ["Smoothies, baking"],
             "spoiled": ["Compost it"]},
}
GENERIC_USES = {"unripe": ["Not ready: leave at room temperature"], "ripe": ["Eat fresh now"],
                "ripe_to_overripe": ["Eat today or tomorrow"], "overripe": ["Cook, blend or bake today"],
                "spoiled": ["Compost it"], "uncertain": ["Rescan"]}
STORAGE = {
    "banana": "Room temperature, hang it; separate from other fruit to slow ripening. Fridge only once ripe (peel darkens, flesh keeps).",
    "avocado": "Counter until ripe, then fridge for up to 3 days.",
    "tomato": "Counter, stem-side down. Fridge dulls flavour.",
    "apple": "Fridge crisper: keeps weeks. Keep away from other produce.",
    "mango": "Counter until ripe, then fridge.", "pear": "Counter to ripen, then fridge.",
    "orange": "Cool or fridge, loose (not sealed).", "lime": "Fridge in a sealed bag: 4 weeks.",
    "grape": "Fridge, unwashed.", "strawberry": "Fridge, unwashed, single layer. Eat within 2–3 days.",
    "kiwi": "Counter to ripen, fridge to hold. Very sensitive to ethylene.",
}
STAGE_SCORE = {"unripe": 0.0, "ripe": 1.0, "ripe_to_overripe": 1.5, "overripe": 2.0, "spoiled": 3.0}


def stage_score(probs: dict) -> float:
    """Expected ripeness on a 0 (unripe) … 3 (spoiled) scale from stage probabilities."""
    tot = sum(probs.get(s, 0) for s in STAGE_SCORE) or 1.0
    return sum(STAGE_SCORE[s] * probs.get(s, 0) for s in STAGE_SCORE) / tot


def _interp3(vals, s):
    """Interpolate a (unripe, ripe, overripe) triple at score s∈[0, 2]."""
    s = min(2.0, max(0.0, s))
    a, b, c = vals
    return a + (b - a) * s if s <= 1 else b + (c - b) * (s - 1)


def profile(fruit: str, stage: str, score: float, feats: dict | None = None) -> dict:
    feats = feats or {}
    fruit = (fruit or "").lower()
    out: dict = {"fruit": fruit, "estimated": True}
    body = BODY.get(fruit)
    vol = feats.get("volume_cm3")
    L, W = feats.get("length_mm"), feats.get("width_mm")
    has_depth = vol is not None and not (isinstance(vol, float) and math.isnan(vol)) and vol > 1
    if body:
        weight, wsrc = body[0], "typical"
        if has_depth:
            weight, wsrc = vol * body[2], "depth"
        out.update(weight_g=round(weight), weight_source=wsrc, edible_g=round(weight * body[1]))
        if has_depth:
            out["volume_cm3"] = round(vol)
    if L and W and not math.isnan(L) and has_depth:
        out["size_cm"] = [round(L / 10, 1), round(W / 10, 1)]

    if fruit in BRIX:
        out["brix"] = round(_interp3(BRIX[fruit], score), 1)
    if fruit in SUGAR_CURVE:
        out["starch_g_100g"] = round(_interp3(STARCH_CURVE[fruit], score), 1)
    n = NUTRITION.get(fruit)
    if n and body:
        g = out["edible_g"] / 100.0
        kcal, sugar, fibre, vitc, k = n
        if fruit in SUGAR_CURVE:
            sugar = _interp3(SUGAR_CURVE[fruit], score)
        out["nutrition"] = {k_: (round(v * g, 1) if v is not None else None) for k_, v in
                            (("kcal", kcal), ("sugar_g", sugar), ("fibre_g", fibre),
                             ("vitamin_c_mg", vitc), ("potassium_mg", k))}
        if fruit == "banana":
            # Resistant starch makes green bananas low-GI; sugar makes ripe ones quick energy.
            out["note"] = ("Mostly starch now: slow energy, low glycaemic index" if score < 0.5 else
                           "Starch has turned to sugar: quick energy, sweetest" if score > 1.4 else
                           "Starch converting to sugar")
    out["uses"] = USES.get(fruit, {}).get(stage) or GENERIC_USES.get(stage, [])
    out["storage"] = STORAGE.get(fruit, "")
    out["ethylene"] = ethylene_advice(fruit, stage)
    return out


def ethylene_advice(fruit: str, stage: str) -> dict:
    """Which fruit to keep it away from, and whether it can speed others up."""
    prod, sens = ETHYLENE.get(fruit, (1, 1))
    ripe_late = stage in ("overripe", "ripe_to_overripe", "spoiled")
    p_eff = min(3, prod + (1 if ripe_late else 0))   # ripening/rotting fruit emit more
    keep_from = sorted(f for f, (p, _) in ETHYLENE.items() if p >= 2 and f != fruit) if sens >= 2 and stage == "unripe" else []
    harms = sorted(f for f, (_, s) in ETHYLENE.items() if s >= 3 and f != fruit) if p_eff >= 2 else []
    tips = []
    if harms:
        tips.append(f"Keep away from {', '.join(harms)}: its ethylene will ripen them faster.")
    if stage == "unripe" and sens >= 2:
        tips.append("To ripen faster: paper bag with a banana or apple.")
    if stage in ("overripe", "spoiled") and p_eff >= 2:
        tips.append("Isolate it: it is speeding up everything around it.")
    return {"production": p_eff, "sensitivity": sens, "keep_away_from": harms,
            "ripens_with": keep_from, "tips": tips}


def co2e_kg(fruit: str, grams: float) -> float:
    return CO2E_PER_KG.get(fruit, CO2E_PER_KG["default"]) * grams / 1000.0
