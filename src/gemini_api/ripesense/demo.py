"""Repeatable touchscreen demo. No sensor or model claims are made by this mode."""
from __future__ import annotations

import io
import json
import logging
import os
from pathlib import Path
import socket
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from flask import Flask, Response, abort, jsonify, request, send_from_directory
import segno

from .recipes import SCHEMA

STATIC = Path(__file__).parent / "static"
SCENARIO = dict(fruit="Banana", stage="Slightly green", ideal_days="1–2",
                texture="Firm", sweetness="Mild", simulated=True,
                note="Demo estimate at room temperature; not a measurement of this fruit.")
FALLBACK = dict(title="Warm cinnamon banana yogurt bowl", minutes=10, servings=1,
    why="A firm, slightly green banana holds its shape when warmed; a little honey balances its mild sweetness.",
    tip="Enjoy it firm today, or let it sweeten for another 1–2 days.",
    ingredients=["1 slightly green banana", "1 tsp butter or neutral oil", "¼ tsp cinnamon",
                 "½ cup (120 g) plain yogurt", "1 tsp honey, optional", "2 tbsp toasted oats"],
    steps=["Peel the banana and slice into thick rounds.",
           "Warm the butter or oil in a pan over medium heat.",
           "Cook the banana 2–3 minutes on each side until golden and softened.",
           "Sprinkle with cinnamon. Spoon over yogurt, add oats, and sweeten to taste."],
    source="Built-in fallback")
PROMPT = """Write one easy recipe for a hackathon simulation of a slightly green banana.
The scenario is a yellow banana with some green peel, firm flesh, and mild sweetness.
It is edible now in this scripted scenario, and ideal for sweeter fresh eating in an estimated
1–2 days at room temperature. This is NOT days until spoilage. No real sensors assessed it.
Choose a recipe to use it NOW at this exact ripeness: gentle cooking is appropriate.
Do not assume a fully green cooking banana, a sweet overripe banana, or use banana bread.
No safety guarantees or claims of actual measurement. At most 8 ingredients and 6 short steps.
Use common equipment and include quantities. Provide minutes, servings, a short title,
one sentence explaining why the recipe suits this ripeness, and a short storage tip.
Return only the requested JSON."""


def read_key():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    path = Path(os.getenv("RIPESENSE_KEY_FILE", str(Path.home() / ".config/ripesense/gemini.key")))
    return path.read_text().strip() if path.exists() else ""


def validate_recipe(data):
    for name in ("title", "why", "tip"):
        if not isinstance(data.get(name), str) or not data[name].strip() or len(data[name]) > 600:
            raise ValueError("Invalid recipe text")
    for name, maximum in (("ingredients", 8), ("steps", 6)):
        values = data.get(name)
        if not isinstance(values, list) or not 1 <= len(values) <= maximum:
            raise ValueError("Invalid recipe list")
        if any(not isinstance(x, str) or not x.strip() or len(x) > 800 for x in values):
            raise ValueError("Invalid recipe item")
    for name in ("minutes", "servings"):
        if type(data.get(name)) is not int or not 1 <= data[name] <= 240:
            raise ValueError("Invalid recipe quantity")
    return {key: data[key] for key in SCHEMA["required"]}


def generate_recipe():
    key = read_key()
    if not key:
        return dict(FALLBACK), "Gemini key not configured"
    models = os.getenv("RIPESENSE_GEMINI_MODELS", "gemini-2.5-flash").split(",")
    failure = "Gemini unavailable"
    for model in models:
        model = model.strip()
        payload = {"contents": [{"parts": [{"text": PROMPT}]}],
                   "generationConfig": {"responseMimeType": "application/json",
                       "responseJsonSchema": SCHEMA, "temperature": 0.6}}
        req = Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                      data=json.dumps(payload).encode(), headers={"Content-Type": "application/json",
                      "x-goog-api-key": key}, method="POST")
        try:
            with urlopen(req, timeout=22) as response:
                body = json.load(response)
            parts = body["candidates"][0]["content"]["parts"]
            data = validate_recipe(json.loads("".join(p.get("text", "") for p in parts if not p.get("thought"))))
            return {**data, "source": f"Gemini · {model}"}, None
        except HTTPError as exc:
            failure = f"Gemini returned HTTP {exc.code}"
        except Exception as exc:
            # Do not log request headers, credentials or response bodies.
            failure = f"Gemini unavailable ({type(exc).__name__})"
    return dict(FALLBACK), failure


def lan_url():
    override = os.getenv("RIPESENSE_PUBLIC_URL", "").rstrip("/")
    if override:
        return override
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("10.255.255.255", 1))
            address = sock.getsockname()[0]
        except OSError:
            address = "127.0.0.1"
    return f"http://{address}:8000"


def create_app(data_dir=None, recipe_fn=generate_recipe, duration=16):
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
    folder = Path(data_dir or os.getenv("RIPESENSE_DEMO_DATA", "demo-data"))
    folder.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    state = {"scan_id": None, "started": None, "recipe": None, "recipe_status": "idle", "recipe_error": None}

    def save_recipe(scan_id):
        try:
            recipe, error = recipe_fn()
        except Exception:
            recipe, error = dict(FALLBACK), "Recipe service unavailable"
        record = dict(scan_id=scan_id, scenario=SCENARIO, recipe=recipe, recipe_error=error)
        temp = folder / f"{scan_id}.tmp"
        temp.write_text(json.dumps(record, ensure_ascii=False))
        temp.replace(folder / f"{scan_id}.json")
        with lock:
            if state["scan_id"] == scan_id:
                state.update(recipe=recipe, recipe_status="ready", recipe_error=error)

    def record(scan_id):
        if len(scan_id) != 16 or any(c not in "0123456789abcdef" for c in scan_id):
            abort(404)
        path = folder / f"{scan_id}.json"
        if not path.exists():
            abort(404)
        return json.loads(path.read_text())

    @app.get("/")
    def kiosk():
        return send_from_directory(STATIC, "demo.html")

    @app.get("/api/demo/state")
    def snapshot():
        with lock:
            snap = dict(state)
        elapsed = max(0, time.monotonic() - snap["started"]) if snap["started"] else 0
        snap.update(status="idle" if not snap["scan_id"] else "scanning" if elapsed < duration else "result",
                    progress=min(1, elapsed / duration), scenario=SCENARIO, gemini_configured=bool(read_key()))
        if snap["scan_id"]:
            snap["recipe_url"] = f"{lan_url()}/r/{snap['scan_id']}"
        return jsonify(snap)

    @app.post("/api/demo/scan")
    def scan():
        # Only the local touchscreen (or an SSH localhost request) can spend API quota.
        if request.remote_addr not in ("127.0.0.1", "::1"):
            abort(403)
        if request.headers.get("Origin") and request.headers["Origin"] != request.host_url.rstrip("/"):
            abort(403)
        with lock:
            if state["started"] and (time.monotonic() - state["started"] < duration or state["recipe_status"] == "pending"):
                return jsonify(error="Scan or recipe still running"), 409
            scan_id = uuid4().hex[:16]
            state.update(scan_id=scan_id, started=time.monotonic(), recipe=None, recipe_status="pending", recipe_error=None)
        threading.Thread(target=save_recipe, args=(scan_id,), daemon=True).start()
        return jsonify(scan_id=scan_id), 202

    @app.post("/api/demo/reset")
    def reset():
        if request.remote_addr not in ("127.0.0.1", "::1"):
            abort(403)
        with lock:
            state.update(scan_id=None, started=None, recipe=None, recipe_status="idle", recipe_error=None)
        return jsonify(ok=True)

    @app.get("/api/demo/recipe/<scan_id>")
    def recipe(scan_id):
        return jsonify(record(scan_id))

    @app.get("/api/demo/qr/<scan_id>.svg")
    def qr(scan_id):
        record(scan_id)
        buf = io.BytesIO()
        segno.make(f"{lan_url()}/r/{scan_id}", error="m", micro=False).save(
            buf, kind="svg", scale=6, border=4, dark="#101710", light="#ffffff", xmldecl=False)
        return Response(buf.getvalue(), mimetype="image/svg+xml")

    @app.get("/r/<scan_id>")
    def phone(scan_id):
        record(scan_id)
        return send_from_directory(STATIC, "demo-recipe.html")

    @app.after_request
    def headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    create_app().run(host="0.0.0.0", port=8000, threaded=True, use_reloader=False)
