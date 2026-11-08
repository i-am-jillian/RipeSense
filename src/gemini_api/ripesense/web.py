"""Flask app: the kiosk screen (/), the data-collection page (/collect) and a small JSON API."""
from __future__ import annotations

import io
import logging
import socket
import threading
import time
from pathlib import Path
from urllib.parse import quote_plus

from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

from . import calibration
from .hardware import build_devices
from .hardware.sim import WORLD
from .model import STAGES
from .features import FRUITS
from .scan import ScanController
from .sinks import make_sink
from .storage import Store, _clean
from .train import train

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


def lan_address() -> str:
    """The Pi's address on the local network (no packet is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def recipe_link(cfg, scan_id: int, recipe: dict | None, fruit: str) -> str:
    q = cfg["qr"]
    if q.get("target") == "search":
        title = recipe["title"] if recipe else f"{fruit} recipe"
        return "https://www.google.com/search?q=" + quote_plus(f"{title} recipe")
    base = (q.get("public_url") or f"http://{lan_address()}:{cfg['app']['port']}").rstrip("/")
    return f"{base}/r/{scan_id}"


def create_app(cfg, devices=None) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
    devices = devices or build_devices(cfg)
    store = Store(cfg.data_dir)
    ctl = ScanController(cfg, devices, store, make_sink(cfg))
    app.config.update(ctl=ctl, store=store, devices=devices, rs_cfg=cfg)
    sim = cfg["app"]["simulate"]
    busy_lock = threading.Lock()

    @app.get("/")
    def kiosk():
        return send_from_directory(STATIC, "kiosk.html")

    @app.get("/collect")
    def collect():
        return send_from_directory(STATIC, "collect.html")

    @app.get("/api/state")
    def state():
        return jsonify(_clean(ctl.snapshot()))

    @app.get("/api/info")
    def info():
        cal = calibration.load(cfg)
        return jsonify(simulate=sim, fruits=FRUITS, stages=STAGES, mode=ctl.mode,
                       calibrated_white="white" in cal, calibrated_empty="empty" in cal,
                       devices={k: type(getattr(devices, k)).__name__ for k in ("camera", "spectral", "io")}
                       | {"gas": type(devices.gas.sensor).__name__},
                       predictor=ctl.snapshot()["predictor"],
                       model_metrics=_clean(getattr(ctl.predictor, "metrics", None)))

    @app.post("/api/scan")
    def scan():
        body = request.get_json(silent=True) or {}
        fruit = body.get("fruit") or None
        if fruit and fruit not in FRUITS:
            abort(400, "unknown fruit")
        ok = ctl.start(fruit_hint=fruit, fruit_id=body.get("fruit_id") or None)
        return jsonify(started=ok), (202 if ok else 409)

    @app.post("/api/mode")
    def mode():
        m = (request.get_json(silent=True) or {}).get("mode")
        if m not in ("home", "store"):
            abort(400, "mode must be home or store")
        ctl.mode = m
        return jsonify(mode=m)

    @app.get("/api/scans")
    def scans():
        return jsonify(_clean(store.recent(int(request.args.get("limit", 50)))))

    @app.get("/api/scans/<int:scan_id>")
    def scan_detail(scan_id):
        row = store.get(scan_id)
        if not row:
            abort(404)
        row.pop("raw", None)
        return jsonify(_clean(row))

    @app.get("/api/scans/<int:scan_id>/thumb.jpg")
    def thumb(scan_id):
        p = store.thumb_path(scan_id)
        if not p.exists():
            abort(404)
        return send_file(p, mimetype="image/jpeg", max_age=3600)

    @app.get("/api/live/preview.jpg")
    def live_preview():
        p = cfg.data_dir / "live_preview.jpg"
        if not p.exists():
            abort(404)
        return send_file(p, mimetype="image/jpeg", max_age=0)

    @app.get("/api/scans/<int:scan_id>/recipe")
    def recipe_json(scan_id):
        row = store.get(scan_id)
        if not row:
            abort(404)
        res = row["result"] or {}
        return jsonify(_clean({"scan_id": scan_id, "fruit": row["fruit_type"], "stage": row["stage"],
                               "days_low": row["days_low"], "days_high": row["days_high"],
                               "headline": res.get("headline"), "recipe": row["recipe"], "ts": row["ts"]}))

    @app.get("/api/scans/<int:scan_id>/qr.svg")
    def qr(scan_id):
        import segno
        row = store.get(scan_id)
        if not row or not row["recipe"]:
            abort(404)
        buf = io.BytesIO()
        segno.make(recipe_link(cfg, scan_id, row["recipe"], row["fruit_type"]), error="m").save(
            buf, kind="svg", scale=4, border=2, dark="#111111", light="#ffffff", xmldecl=False)
        return Response(buf.getvalue(), mimetype="image/svg+xml")

    @app.get("/r/<int:scan_id>")
    def recipe_page(scan_id):
        return send_from_directory(STATIC, "recipe.html")

    @app.post("/api/label")
    def label():
        b = request.get_json(silent=True) or {}
        stage = b.get("stage") or None
        if stage is not None and stage not in STAGES:
            abort(400, "bad stage")
        days = b.get("days")
        days = float(days) if days not in (None, "") else None
        store.label(int(b["scan_id"]), stage, days, b.get("fruit_id") or None, b.get("fruit_type") or None)
        return jsonify(ok=True)

    @app.post("/api/train")
    def do_train():
        b = request.get_json(silent=True) or {}
        if not busy_lock.acquire(blocking=False):
            abort(409, "already training")
        try:
            metrics = train(cfg, store, include_synthetic=bool(b.get("include_synthetic")))
            ctl.reload_model()
            return jsonify(ok=True, metrics=_clean(metrics))
        except ValueError as e:
            return jsonify(ok=False, error=str(e)), 400
        finally:
            busy_lock.release()

    @app.post("/api/calibrate/<what>")
    def calibrate(what):
        if ctl.busy():
            abort(409, "scan running")
        if what == "white":
            if sim:
                WORLD.puck = True
            try:
                res = calibration.calibrate_white(cfg, devices)
            finally:
                WORLD.puck = False
            return jsonify(ok=True, saturated=res["saturated"])
        if what == "empty":
            if sim:
                present = WORLD.fruit_present
                WORLD.fruit_present = False
            try:
                res = calibration.capture_empty(cfg, devices)
            finally:
                if sim:
                    WORLD.fruit_present = present
            return jsonify(ok=True, floor_m=res["floor_m"])
        abort(404)

    @app.get("/api/export.csv")
    def export():
        return Response(store.export_csv(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=ripesense_scans.csv"})

    # ---- simulator controls (only when simulate = true)
    @app.post("/api/sim/insert")
    def sim_insert():
        if not sim:
            abort(404)
        b = request.get_json(silent=True) or {}
        WORLD.open_door()
        time.sleep(1.5)       # door stands open: room air for the gas baseline
        r = b.get("ripeness")
        WORLD.set_fruit(b.get("fruit") or "banana", None if r in (None, "") else float(r))
        return jsonify(fruit=WORLD.fruit_type, ripeness=round(WORLD.ripeness, 2))

    @app.post("/api/sim/touch")
    def sim_touch():
        if not sim:
            abort(404)
        devices.io.press()
        return jsonify(ok=True)

    return app
