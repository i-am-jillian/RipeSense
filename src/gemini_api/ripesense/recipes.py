"""A recipe (and a one-line tip) matched to the fruit's ripeness.

Gemini writes it as structured JSON when GEMINI_API_KEY is set; otherwise a built-in recipe is used,
so the QR code works offline too. Gemini only words the answer: the model decides the ripeness.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import os

log = logging.getLogger(__name__)
_pool = cf.ThreadPoolExecutor(max_workers=2)

SCHEMA = {
    "type": "object",
    "properties": {
        "tip": {"type": "string", "description": "One sentence for the screen, max 18 words"},
        "title": {"type": "string"},
        "why": {"type": "string", "description": "Why this recipe suits this ripeness, one sentence"},
        "minutes": {"type": "integer"},
        "servings": {"type": "integer"},
        "ingredients": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["tip", "title", "why", "minutes", "servings", "ingredients", "steps"],
}

PROMPT = """You are the recipe helper inside RipeSense, a fruit-ripeness scanner.
The scanner measured: fruit = {fruit}, stage = {stage}, about {days} day(s) left before it spoils.
{audience}
Suggest ONE easy home recipe that suits a {fruit} at exactly this stage (for example, very ripe bananas
suit baking; unripe ones suit cooking). Keep it to at most 8 ingredients and 6 short steps, using
common kitchen equipment. Metric and US units together where useful. No emoji."""

# Built-in recipes: (fruit, stage) -> recipe. Used offline or when Gemini fails.
LIBRARY = {
    ("banana", "unripe"): dict(
        title="Savory green banana fry", minutes=25, servings=2,
        why="Green bananas are starchy, not sweet, so they cook like potatoes.",
        ingredients=["2 green bananas", "1 tbsp oil", "1/2 tsp cumin seeds", "1/2 tsp turmeric",
                     "1/2 tsp chili powder", "Salt", "Squeeze of lemon"],
        steps=["Peel the bananas with a knife and slice into 1 cm rounds.",
               "Boil 5 minutes in salted water, then drain.",
               "Heat oil, add cumin seeds until they sizzle.",
               "Add bananas, turmeric, chili and salt; fry 8-10 minutes until golden.",
               "Finish with lemon and serve warm."]),
    ("banana", "ripe"): dict(
        title="Banana oat breakfast bowl", minutes=10, servings=1,
        why="A ripe banana is sweet enough that you need no added sugar.",
        ingredients=["1 ripe banana", "1/2 cup (45 g) rolled oats", "3/4 cup (180 ml) milk",
                     "1 tbsp peanut butter", "Pinch of cinnamon"],
        steps=["Mash half the banana in a small pot.", "Add oats and milk; simmer 4-5 minutes, stirring.",
               "Stir in cinnamon.", "Top with the rest of the banana, sliced, and the peanut butter."]),
    ("banana", "overripe"): dict(
        title="One-bowl banana bread", minutes=65, servings=8,
        why="Brown-spotted bananas are at peak sugar, perfect for baking.",
        ingredients=["3 very ripe bananas", "1/3 cup (75 g) melted butter", "1/2 cup (100 g) sugar",
                     "1 egg", "1 tsp vanilla", "1 tsp baking soda", "Pinch of salt", "1.5 cups (190 g) flour"],
        steps=["Heat the oven to 175 C / 350 F and grease a loaf pan.", "Mash the bananas in a bowl.",
               "Stir in butter, sugar, egg and vanilla.", "Mix in baking soda and salt, then fold in flour.",
               "Pour into the pan and bake 55-60 minutes, until a skewer comes out clean.",
               "Cool 10 minutes before slicing."]),
    ("avocado", "unripe"): dict(
        title="Ripen it first: paper-bag trick", minutes=5, servings=1,
        why="Hard avocados are bitter; they ripen in 1-3 days at room temperature.",
        ingredients=["1 firm avocado", "1 paper bag", "1 banana or apple"],
        steps=["Put the avocado in the paper bag with the banana or apple.",
               "Fold the bag shut and leave it at room temperature.",
               "Scan it again tomorrow."]),
    ("avocado", "ripe"): dict(
        title="Quick guacamole", minutes=10, servings=3,
        why="A just-ripe avocado mashes smoothly but still has a fresh taste.",
        ingredients=["2 ripe avocados", "1/2 small onion, finely chopped", "1 small tomato, diced",
                     "1 lime", "Handful of cilantro", "Salt"],
        steps=["Halve the avocados, remove the pits, scoop into a bowl.", "Mash with a fork, leaving some chunks.",
               "Stir in onion, tomato and chopped cilantro.", "Season with lime juice and salt to taste."]),
    ("avocado", "overripe"): dict(
        title="Chocolate avocado mousse", minutes=10, servings=2,
        why="Very soft avocado blends silky-smooth; cocoa hides any browning.",
        ingredients=["1 very ripe avocado", "3 tbsp cocoa powder", "3 tbsp honey or maple syrup",
                     "3 tbsp milk", "1/2 tsp vanilla", "Pinch of salt"],
        steps=["Scoop the avocado into a blender, cutting away any dark brown parts.",
               "Add everything else and blend until smooth.", "Chill 15 minutes and serve."]),
    ("tomato", "unripe"): dict(
        title="Fried green tomatoes", minutes=20, servings=2,
        why="Firm green tomatoes hold their shape and stay tangy when fried.",
        ingredients=["2 green tomatoes", "1/2 cup (60 g) flour", "1 egg", "1/2 cup (70 g) cornmeal or breadcrumbs",
                     "Salt and pepper", "Oil for frying"],
        steps=["Slice tomatoes 1 cm thick and season with salt and pepper.",
               "Dip each slice in flour, then beaten egg, then cornmeal.",
               "Fry in hot oil 2-3 minutes per side until golden.", "Drain on paper towel and serve hot."]),
    ("tomato", "ripe"): dict(
        title="Fresh tomato salsa", minutes=10, servings=4,
        why="Ripe tomatoes are juicy and sweet, best eaten raw.",
        ingredients=["3 ripe tomatoes", "1/2 onion", "1 small chili", "1 lime", "Handful of cilantro", "Salt"],
        steps=["Dice tomatoes, onion and chili finely.", "Chop the cilantro.",
               "Mix everything with lime juice and salt.", "Rest 5 minutes before serving."]),
    ("tomato", "overripe"): dict(
        title="Quick roasted tomato pasta sauce", minutes=35, servings=2,
        why="Soft tomatoes roast down into a rich sauce in minutes.",
        ingredients=["4 soft tomatoes", "3 garlic cloves", "2 tbsp olive oil", "Salt and pepper",
                     "Handful of basil", "200 g pasta"],
        steps=["Heat the oven to 220 C / 425 F.", "Halve tomatoes, toss with garlic, oil, salt and pepper.",
               "Roast 25 minutes.", "Cook the pasta meanwhile.",
               "Mash the roasted tomatoes into a sauce, toss with pasta and basil."]),
    ("mango", "unripe"): dict(
        title="Green mango salad", minutes=15, servings=2,
        why="Unripe mango is crisp and sour, like a crunchy vegetable.",
        ingredients=["1 green mango", "1 small carrot", "1 tbsp lime juice", "1 tsp sugar", "Pinch of chili",
                     "Handful of peanuts", "Salt"],
        steps=["Peel and shred the mango and carrot.", "Whisk lime juice, sugar, chili and salt.",
               "Toss everything together and top with peanuts."]),
    ("mango", "ripe"): dict(
        title="Mango lassi", minutes=5, servings=2,
        why="A ripe mango is sweet and fragrant, ideal for blending.",
        ingredients=["1 ripe mango", "1 cup (240 g) plain yogurt", "1/2 cup (120 ml) milk", "1 tsp sugar (optional)",
                     "Pinch of cardamom"],
        steps=["Peel and chop the mango.", "Blend with everything else until smooth.", "Serve cold."]),
    ("mango", "overripe"): dict(
        title="Mango smoothie popsicles", minutes=10, servings=6,
        why="Very soft mango is too mushy to slice but perfect to freeze.",
        ingredients=["1 very ripe mango", "1 banana", "1/2 cup (120 g) yogurt", "1 tbsp honey"],
        steps=["Blend everything until smooth.", "Pour into popsicle molds or small cups.",
               "Freeze at least 4 hours."]),
    ("apple", "unripe"): dict(
        title="Tangy apple slaw", minutes=10, servings=2,
        why="Tart, firm apples add crunch and sharpness to a slaw.",
        ingredients=["1 tart apple", "2 cups shredded cabbage", "2 tbsp yogurt", "1 tsp mustard", "1 tsp honey",
                     "Salt and pepper"],
        steps=["Cut the apple into matchsticks.", "Whisk yogurt, mustard, honey, salt and pepper.",
               "Toss with apple and cabbage."]),
    ("apple", "ripe"): dict(
        title="Apple and peanut butter slices", minutes=5, servings=1,
        why="A ripe apple is crisp and sweet, best fresh.",
        ingredients=["1 apple", "2 tbsp peanut butter", "1 tbsp granola", "Pinch of cinnamon"],
        steps=["Core the apple and slice into rings.", "Spread with peanut butter.", "Sprinkle granola and cinnamon."]),
    ("apple", "overripe"): dict(
        title="Quick apple crumble", minutes=40, servings=4,
        why="Soft, mealy apples bake into a jammy filling.",
        ingredients=["3 soft apples", "1 tbsp sugar", "1 tsp cinnamon", "1/2 cup (45 g) oats",
                     "1/3 cup (40 g) flour", "1/4 cup (50 g) brown sugar", "1/4 cup (55 g) cold butter"],
        steps=["Heat the oven to 190 C / 375 F.", "Slice apples, toss with sugar and cinnamon, put in a dish.",
               "Rub oats, flour, brown sugar and butter into crumbs.", "Scatter over the apples and bake 25-30 minutes."]),
}
TIPS = {"unripe": "Not ready to eat raw yet: try it cooked, or wait a couple of days.",
        "ripe": "Perfect to eat now.", "overripe": "Best used today in cooking or baking.",
        "spoiled": "Past its best: compost it."}


def builtin(fruit: str, stage: str) -> dict | None:
    if stage == "spoiled":
        return None
    r = LIBRARY.get((fruit, stage))
    if r is None:
        return None
    return {**r, "tip": TIPS[stage], "source": "built-in"}


def _gemini(fruit, stage, days, mode, cfg) -> dict | None:
    g = cfg["gemini"]
    if not g.get("enabled") or not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        log.info("google-genai not installed; using built-in recipes")
        return None
    audience = ("The reader is a grocery shopper deciding whether to buy it today."
                if mode == "store" else "The reader is cooking at home.")
    prompt = PROMPT.format(fruit=fruit, stage=stage, days="unknown" if days is None else days, audience=audience)
    conf = types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=SCHEMA,
                                       temperature=0.7)
    client = genai.Client()
    last = None
    for model in [g["model"]] + [m for m in g.get("fallback_models", []) if m != g["model"]]:
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=conf)
            data = json.loads(resp.text)
            data["ingredients"] = [str(x) for x in data["ingredients"]][:10]
            data["steps"] = [str(x) for x in data["steps"]][:8]
            data["source"] = f"Gemini ({model})"
            return data
        except Exception as e:  # wrong model name, quota, network, bad JSON
            last = e
            log.warning("Gemini %s failed: %s", model, e)
    log.warning("all Gemini models failed (%s); using built-in recipe", last)
    return None


def get_recipe(fruit: str, stage: str, days, mode: str, cfg) -> dict | None:
    """Gemini first (with a timeout), then the built-in library. None for spoiled fruit."""
    if stage == "spoiled":
        return None
    try:
        data = _pool.submit(_gemini, fruit, stage, days, mode, cfg).result(timeout=cfg["gemini"]["timeout_s"])
    except cf.TimeoutError:
        log.warning("Gemini timed out; using built-in recipe")
        data = None
    return data or builtin(fruit, stage)
