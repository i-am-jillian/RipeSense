"""Flask app: kiosk screen (/), data collection (/collect), store dashboard
(/store) and sim controls (/dev, mock only).

  python -m ripesense.app            # real hardware (on the Pi)
  RIPESENSE_MOCK=1 RIPESENSE_FAST=1 python -m ripesense.app   # laptop
"""
from __future__ import annotations

import logging
import threading
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory

from . import config, store
from .cloud import tiger
from .scan import Controller
from .stages import STAGE_LABEL, STAGES

log = logging.getLogger("ripesense")


def create_app(controller: Controller | None = None) -> Flask:
    app = Flask(__name__)
    ctl = controller or Controller()
    app.config["ctl"] = ctl
    cfg = ctl.cfg
    train_state = {"running": False, "last": None}

    # ----------------------------------------------------------------- pages
    @app.get("/")
    def kiosk():
        return render_template("kiosk.html")

    @app.get("/collect")
    def collect():
        return render_template("collect.html", stages=STAGES, stage_label=STAGE_LABEL)

    @app.get("/store")
    def store_page():
        return render_template("store.html")

    @app.get("/camera")
    def camera_page():
        return render_template("camera.html")

    @app.get("/api/preview.jpg")
    def preview():
        jpg = ctl.preview_jpeg()
        if jpg is None:
            abort(404)
        return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/reload_models")
    def reload_models():
        ctl.reload_models()
        return jsonify(ctl.models.summary())

    @app.get("/dev")
    def dev():
        if not ctl.mock:
            abort(404)
        return render_template("dev.html", stages=STAGES)

    @app.get("/data/<path:p>")
    def data_file(p):
        return send_from_directory(config.data_dir(), p)

    @app.get("/model/<path:p>")
    def model_file(p):
        return send_from_directory(config.models_dir(), p)

    # ------------------------------------------------------------------- api
    @app.get("/api/status")
    def status():
        return jsonify(ctl.snapshot())

    @app.post("/api/scan")
    def scan():
        meta = request.get_json(silent=True) or {}
        ok = ctl.start_scan(meta)
        return jsonify({"started": ok, **ctl.snapshot()}), (200 if ok else 409)

    @app.post("/api/reset")
    def reset():
        if ctl.status["state"] != "scanning":
            ctl.result = None
            ctl._set("idle", 0, "Place fruit, close door, tap")
        return jsonify(ctl.snapshot())

    @app.post("/api/mode")
    def mode():
        m = (request.get_json(silent=True) or {}).get("mode")
        ctl.mode = m if m in ("home", "store") else ("store" if ctl.mode == "home" else "home")
        return jsonify({"mode": ctl.mode})

    @app.post("/api/save")
    def save():
        r = ctl.result
        if not r:
            return jsonify({"ok": False}), 400
        store.update(r["scan_id"], saved="1")
        r["saved"] = True
        return jsonify({"ok": True})

    @app.post("/api/collect")
    def set_collect():
        body = request.get_json(silent=True) or {}
        ctl.collect = {k: str(body.get(k, "")).strip() for k in ("fruit_id", "fruit_type", "stage_label")}
        return jsonify(ctl.collect)

    @app.get("/api/scans")
    def scans():
        limit = int(request.args.get("limit", 100))
        return jsonify(store.rows(limit))

    @app.post("/api/label/<int:scan_id>")
    def label(scan_id):
        body = request.get_json(silent=True) or {}
        allowed = {k: v for k, v in body.items() if k in ("stage_label", "days_left_label", "fruit_id", "fruit_type")}
        if "stage_label" in allowed and allowed["stage_label"] not in STAGES + [""]:
            return jsonify({"error": "bad stage"}), 400
        store.update(scan_id, **allowed)
        return jsonify(store.get(scan_id))

    @app.post("/api/spoiled")
    def spoiled():
        body = request.get_json(silent=True) or {}
        fid = body.get("fruit_id", "").strip()
        day = date.fromisoformat(body["date"]) if body.get("date") else date.today()
        if not fid:
            return jsonify({"error": "fruit_id required"}), 400
        return jsonify({"updated": store.mark_spoiled(fid, day)})

    @app.post("/api/calibrate")
    def calibrate():
        try:
            return jsonify({"ok": True, "ref": ctl.calibrate_white()})
        except Exception as e:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.post("/api/train")
    def train():
        if train_state["running"]:
            return jsonify({"running": True}), 409

        def work():
            from .models import train_fusion
            train_state["running"] = True
            try:
                train_state["last"] = {"ok": True, **train_fusion.train(config.data_dir() / "scans.csv", cfg),
                                       "at": datetime.now().isoformat(timespec="seconds")}
                ctl.reload_models()
            except BaseException as e:  # noqa: BLE001  (SystemExit on no labels)
                train_state["last"] = {"ok": False, "error": str(e)}
            finally:
                train_state["running"] = False

        threading.Thread(target=work, daemon=True).start()
        return jsonify({"running": True})

    @app.get("/api/train")
    def train_status():
        return jsonify(train_state)

    @app.get("/api/train_history")
    def train_history():
        import json as _json
        p = config.models_dir() / "train_history.json"
        return jsonify(_json.loads(p.read_text()) if p.exists() else [])

    @app.get("/api/label_queue")
    def label_queue():
        """Active learning: unlabelled scans the model was least sure about first."""
        rows = [r for r in store.rows(300) if not r.get("stage_label")]

        def conf(r):
            try:
                return float(r.get("pred_conf") or 0)
            except ValueError:
                return 0.0
        rows.sort(key=conf)
        keep = ("scan_id", "fruit_id", "fruit_type", "timestamp", "pred_stage", "pred_conf", "rgb_path")
        return jsonify([{k: r.get(k) for k in keep} for r in rows[: int(request.args.get("limit", 12))]])

    # ---------------------------------------------------- upgrades: teach etc.
    @app.post("/api/teach")
    def teach():
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(ctl.teach(str(body.get("label", "")), int(body.get("n", 5)),
                                     bool(body.get("from_last"))))
        except (RuntimeError, ValueError) as e:
            return jsonify({"error": str(e)}), 400

    @app.post("/api/forget")
    def forget():
        body = request.get_json(silent=True) or {}
        if ctl.models.fewshot is None:
            return jsonify({"error": "few-shot unavailable"}), 400
        ctl.models.fewshot.forget(body.get("label") or None)
        return jsonify({"taught": ctl.models.fewshot.summary()})

    def _scan_row(scan_id):
        r = store.get(scan_id)
        if not r:
            abort(404)
        return r

    def _history_until(r):
        from . import forecast as fc
        rows = store.rows(where="WHERE fruit_id=? AND scan_id<=?", params=(r["fruit_id"], r["scan_id"])) \
            if r.get("fruit_id") else [r]
        return fc.history_from_rows(rows)

    @app.get("/api/forecast/<int:scan_id>")
    def forecast_api(scan_id):
        from . import forecast as fc
        r = _scan_row(scan_id)
        hist = _history_until(r)
        if not hist:
            return jsonify({"error": "no usable scans for this fruit"}), 400
        return jsonify({"forecast": fc.forecast(hist, r["fruit_type"], request.args.get("scenario", "counter")),
                        "whatif": fc.whatifs(hist, r["fruit_type"])})

    @app.get("/api/future/<int:scan_id>.jpg")
    def future_api(scan_id):
        import cv2
        import numpy as np

        from . import forecast as fc
        from .future import preview
        r = _scan_row(scan_id)
        days = float(request.args.get("days", 3))
        bgr = cv2.imread(str(config.data_dir() / r["rgb_path"])) if r.get("rgb_path") else None
        hist = _history_until(r)
        if bgr is None or not hist:
            abort(404)
        f = fc.forecast(hist, r["fruit_type"], request.args.get("scenario", "counter"), horizon_days=max(days, 0.25))
        i = int(np.argmin(np.abs(np.array(f["curve"]["days"]) - days)))
        q_now, q_fut = f["curve"]["p50"][0], f["curve"]["p50"][i]
        img = preview(bgr, r["fruit_type"], q_now, q_fut, seed=scan_id)
        h = 360 / img.shape[0]
        img = cv2.resize(img, None, fx=h, fy=h, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return Response(buf.tobytes(), mimetype="image/jpeg", headers={"X-Ripeness": f"{q_now:.2f}->{q_fut:.2f}"})

    @app.get("/api/recipe/<int:scan_id>")
    def recipe_api(scan_id):
        from . import fruit_info
        from .cloud import gemini
        r = _scan_row(scan_id)
        stage = r.get("pred_stage") or "uncertain"
        score = float(r["pred_score"]) if r.get("pred_score") else fruit_info.STAGE_SCORE.get(stage, 1.0)
        prof = fruit_info.profile(r["fruit_type"], stage, score)
        return jsonify(gemini.recipe(r["fruit_type"], stage, prof, cfg))

    @app.post("/api/unplug")
    def unplug():
        """'Pull the cable' demo: switch a sensor off/on for the next scans."""
        body = request.get_json(silent=True) or {}
        sensor = body.get("sensor")
        if sensor not in ("camera", "spectral", "gas"):
            return jsonify({"error": "sensor must be camera|spectral|gas"}), 400
        if body.get("off", sensor not in ctl.disabled):
            ctl.disabled.add(sensor)
        else:
            ctl.disabled.discard(sensor)
        return jsonify({"disabled": sorted(ctl.disabled)})

    @app.get("/api/store_summary")
    def store_summary():
        remote = tiger.week_by_tier(cfg)
        if remote is not None:
            return jsonify({"source": "tiger", "rows": [
                {"day": r["day"].date().isoformat(), "stage": r["stage"], "n": r["n"],
                 "markdown_dollars": float(r["markdown_dollars"] or 0)} for r in remote]})
        since = (datetime.now() - timedelta(days=7)).isoformat()
        agg: dict = defaultdict(Counter)
        dollars: dict = defaultdict(float)
        impact = {"recovered": 0.0, "vs_tier": 0.0, "waste_kg": 0.0, "co2e_kg": 0.0, "donate": 0, "priced": 0}
        from .stages import store_info

        def num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        for r in store.rows(where="WHERE timestamp >= ?", params=(since,)):
            day = r["timestamp"][:10]
            st = r["pred_stage"] or "uncertain"
            agg[day][st] += 1
            dd = [num(r.get(k)) for k in ("pred_days_p10", "pred_days_p50", "pred_days_p90")]
            days = {"p10": dd[0], "p50": dd[1], "p90": dd[2]} if None not in dd else None
            info = store_info(st, r["fruit_type"], cfg, days=days)
            if info["price"] is not None and info["base_price"] is not None:
                dollars[(day, st)] += info["base_price"] - info["price"]
            o = info.get("optimiser")
            if o:
                impact["priced"] += 1
                impact["recovered"] += o["vs_no_markdown"]
                impact["vs_tier"] += o["vs_tier"]
                impact["waste_kg"] += o["waste_avoided_kg"]
                impact["co2e_kg"] += o["co2e_avoided_kg"]
                impact["donate"] += o["action"] == "donate"
        rows = [{"day": d, "stage": s, "n": n, "markdown_dollars": round(dollars[(d, s)], 2)}
                for d in sorted(agg) for s, n in agg[d].items()]
        return jsonify({"source": "local", "rows": rows,
                        "impact": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in impact.items()}})

    # --------------------------------------------------------- sim (mock only)
    @app.post("/api/sim")
    def sim():
        if not ctl.mock:
            abort(404)
        from .sim import fruit as simf
        body = request.get_json(silent=True) or {}
        if not body.get("keep"):
            simf.current = simf.random_fruit(body.get("fruit_type") or None, body.get("stage") or None,
                                             body.get("fruit_id") or "sim-demo")
        if "hidden_bruises" in body:
            simf.current.hidden_bruises = int(body["hidden_bruises"])
        return jsonify({"fruit_type": simf.current.fruit_type, "stage": simf.current.stage,
                        "days_left": round(simf.current.days_left, 2),
                        "hidden_bruises": simf.current.hidden_bruises})

    @app.post("/api/door")
    def door():
        if not ctl.mock:
            abort(404)
        closed = bool((request.get_json(silent=True) or {}).get("closed", True))
        ctl.io.set_door(closed)
        return jsonify({"door_closed": ctl.io.door_closed()})

    return app


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = config.load()
    app = create_app()
    log.info("RipeSense on http://%s:%s  (mock=%s)", cfg["ui"]["host"], cfg["ui"]["port"], cfg["mock"])
    app.run(host=cfg["ui"]["host"], port=cfg["ui"]["port"], threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
