"""Gemini turns the model's structured output into one friendly sentence.
It never makes the decision. Without GEMINI_API_KEY or network, a canned tip
is used so scans still work offline."""
from __future__ import annotations

import json
import logging
import os
import urllib.request

log = logging.getLogger(__name__)

FALLBACK = {
    "unripe": "Leave it on the counter; check again in a couple of days.",
    "ripe": "Perfect for eating fresh over the next few days.",
    "ripe_to_overripe": "Eat it today or tomorrow — great sliced on cereal or toast.",
    "overripe": "Ideal for smoothies or baking today.",
    "spoiled": "Past its best — compost it.",
    "uncertain": "Turn the fruit a quarter and scan again.",
}
RECIPE_FALLBACK = {
    ("banana", "overripe"): "Perfect for banana bread today.",
    ("banana", "ripe_to_overripe"): "Freeze slices now for smoothies, or bake banana bread tomorrow.",
    ("avocado", "ripe"): "Ready for guacamole or avocado toast.",
    ("tomato", "overripe"): "Cook it down into a quick pasta sauce today.",
}


def fallback_tip(fruit: str, stage: str) -> str:
    return RECIPE_FALLBACK.get((fruit, stage), FALLBACK.get(stage, ""))


def tip(fruit: str, result: dict, cfg: dict) -> tuple[str, str]:
    """Returns (tip, source) where source is 'gemini' or 'offline'."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    stage = result["stage"]
    if not key or stage == "uncertain":
        return fallback_tip(fruit, stage), "offline"
    days = result.get("days") or {}
    facts = {
        "fruit": fruit,
        "stage": result["stage_label"],
        "days_left_range": [round(days.get("p10", 0)), round(days.get("p90", 0))] if days else None,
        "confidence": round(result["confidence"], 2),
    }
    prompt = (
        "You write one short tip (max 18 words) for a kitchen fruit-ripeness scanner screen. "
        "Do not change or second-guess the verdict; just say how to use the fruit, with one concrete "
        "recipe or storage idea that suits the days left. No emoji, no quotes.\n"
        f"Scanner result: {json.dumps(facts)}"
    )
    model = cfg["cloud"].get("gemini_model", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"maxOutputTokens": 60, "temperature": 0.7}}).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=cfg["cloud"].get("gemini_timeout_s", 6)) as r:
            data = json.loads(r.read())
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip().strip('"')
        return (text or fallback_tip(fruit, stage)), "gemini"
    except Exception as e:  # noqa: BLE001
        log.warning("Gemini tip failed (%s); using offline tip", e)
        return fallback_tip(fruit, stage), "offline"


# ------------------------------------------------------------ "use it tonight"
OFFLINE_RECIPES = {
    ("banana", "overripe"): {"title": "3-ingredient banana pancakes",
                             "ingredients": ["1 overripe banana", "2 eggs", "pinch of cinnamon"],
                             "steps": ["Mash the banana", "Whisk in eggs and cinnamon",
                                       "Fry small spoonfuls 1–2 min a side"], "minutes": 10},
    ("banana", "ripe_to_overripe"): {"title": "Banana–oat smoothie",
                                     "ingredients": ["1 banana", "250 ml milk", "3 tbsp oats", "1 tsp honey"],
                                     "steps": ["Blend everything 30 s", "Serve cold"], "minutes": 5},
    ("banana", "unripe"): {"title": "Green banana chips",
                           "ingredients": ["2 green bananas", "oil", "salt", "chilli powder"],
                           "steps": ["Peel and slice thinly", "Fry until crisp", "Season while hot"], "minutes": 20},
    ("avocado", "ripe"): {"title": "5-minute guacamole",
                          "ingredients": ["1 avocado", "½ lime", "salt", "chopped onion & coriander"],
                          "steps": ["Mash avocado with lime and salt", "Fold in onion and coriander"], "minutes": 5},
    ("tomato", "overripe"): {"title": "Quick tomato pasta sauce",
                             "ingredients": ["3 soft tomatoes", "1 garlic clove", "olive oil", "basil"],
                             "steps": ["Chop tomatoes", "Fry garlic in oil 1 min", "Add tomatoes, simmer 10 min",
                                       "Tear in basil"], "minutes": 15},
    ("apple", "overripe"): {"title": "Microwave apple crumble",
                            "ingredients": ["1 apple", "2 tbsp oats", "1 tbsp butter", "1 tbsp sugar", "cinnamon"],
                            "steps": ["Dice apple into a mug", "Rub oats, butter, sugar together, scatter on top",
                                      "Microwave 2–3 min"], "minutes": 8},
    ("mango", "overripe"): {"title": "Mango lassi",
                            "ingredients": ["1 mango", "200 g yoghurt", "100 ml milk", "pinch of cardamom"],
                            "steps": ["Blend everything until smooth"], "minutes": 5},
}
GENERIC_RECIPES = {
    "overripe": {"title": "Fruit smoothie", "ingredients": ["the fruit", "yoghurt or milk", "ice"],
                 "steps": ["Cut away any bad parts", "Blend 30 s"], "minutes": 5},
    "ripe_to_overripe": {"title": "Fruit and yoghurt bowl", "ingredients": ["the fruit", "yoghurt", "oats or granola"],
                         "steps": ["Slice the fruit", "Layer with yoghurt and granola"], "minutes": 5},
    "ripe": {"title": "Eat it fresh", "ingredients": ["the fruit"], "steps": ["Wash, slice, enjoy"], "minutes": 2},
}


def recipe(fruit: str, stage: str, profile: dict | None, cfg: dict) -> dict:
    """One quick recipe for the fruit as it is now. Gemini when a key is set,
    otherwise a built-in recipe. Returns {title, ingredients, steps, minutes, source}."""
    base = OFFLINE_RECIPES.get((fruit, stage)) or GENERIC_RECIPES.get(stage)
    if stage in ("spoiled", "uncertain") or not base:
        return {"title": "Compost it" if stage == "spoiled" else "Rescan first", "ingredients": [], "steps": [],
                "minutes": 0, "source": "offline"}
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return {**base, "source": "offline"}
    facts = {"fruit": fruit, "stage": stage,
             "estimated_weight_g": (profile or {}).get("weight_g"),
             "estimated_brix": (profile or {}).get("brix")}
    prompt = ("Suggest ONE quick home recipe (under 20 minutes) that suits this fruit at this ripeness. "
              "Reply as compact JSON with keys title, ingredients (list of short strings), steps (list, max 4), "
              f"minutes (int). No markdown.\nFruit: {json.dumps(facts)}")
    model = cfg["cloud"].get("gemini_model", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"maxOutputTokens": 300, "temperature": 0.7,
                                            "responseMimeType": "application/json"}}).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=cfg["cloud"].get("gemini_timeout_s", 6) + 4) as r:
            data = json.loads(r.read())
        r = json.loads(data["candidates"][0]["content"]["parts"][0]["text"])
        return {"title": str(r["title"]), "ingredients": [str(x) for x in r.get("ingredients", [])][:8],
                "steps": [str(x) for x in r.get("steps", [])][:5], "minutes": int(r.get("minutes", 0) or 0),
                "source": "gemini"}
    except Exception as e:  # noqa: BLE001
        log.warning("Gemini recipe failed (%s); using built-in recipe", e)
        return {**base, "source": "offline"}
