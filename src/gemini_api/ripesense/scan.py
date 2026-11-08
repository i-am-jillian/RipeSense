"""The scan sequence. One tap: camera -> spectral -> (gas, running the whole time) -> model -> screen.

The controller exposes a small state dict that the UI polls:
  status: idle | scanning | result | error
  step:   door | camera | spectral | gas | model
"""
from __future__ import annotations

import logging
import math
import threading
import time
import traceback

import cv2
import numpy as np

from . import calibration
from .features import (FEATURES, ImageClassifier, fruit_code, gas_features, image_features, segment,
                       spectral_features)
from .hardware.spectral import WAVELENGTHS, measure
from .model import load_predictor
from .storage import Store
from .recipes import get_recipe
from .verdict import verdict

log = logging.getLogger(__name__)
STEPS = ["door", "camera", "spectral", "gas", "model"]


class Aborted(Exception):
    pass


class ScanController:
    def __init__(self, cfg, devices, store: Store | None = None, sink=None):
        self.cfg = cfg
        self.dev = devices
        self.store = store or Store(cfg.data_dir)
        self.sink = sink
        self.predictor = load_predictor(cfg)
        self.classifier = None
        p = cfg.path(cfg["model"]["image_model"])
        if p.exists():
            try:
                self.classifier = ImageClassifier(p)
            except Exception as e:  # pragma: no cover
                log.warning("image classifier not loaded: %s", e)
        self.mode = cfg["app"]["mode"]
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state = {"status": "idle", "step": None, "progress": 0.0, "message": "", "result": None, "scan_id": None}
        devices.io.on_touch(lambda: self.start())

    # ------------------------------------------------------------------ public
    def snapshot(self) -> dict:
        with self._lock:
            # opening the door after a result or error means "next fruit": back to the idle prompt
            opened = self.dev.io.door_opened_at
            if self.state["status"] in ("result", "error") and opened and opened > self.state.get("finished_at", 0):
                self.state.update(status="idle", result=None, message="", progress=0.0)
            s = dict(self.state)
        s["mode"] = self.mode
        s["predictor"] = "model" if not type(self.predictor).__name__.startswith("Heuristic") else "heuristic"
        s["gas_warmup_left_s"] = round(self.dev.gas.warmup_left_s)
        s["door_closed"] = self.dev.io.door_closed()
        return s

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, fruit_hint: str | None = None, fruit_id: str | None = None) -> bool:
        if self.busy():
            return False
        self._thread = threading.Thread(target=self._run, args=(fruit_hint, fruit_id), daemon=True, name="scan")
        self._thread.start()
        return True

    def wait(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)

    def reload_model(self):
        self.predictor = load_predictor(self.cfg)

    # ------------------------------------------------------------------ internals
    def _set(self, **kw):
        with self._lock:
            self.state.update(kw)

    def _step(self, name, message, progress):
        self._set(status="scanning", step=name, message=message, progress=round(progress, 3))

    def _live(self, **kw):
        with self._lock:
            self.state["live"] = {**self.state.get("live", {}), **kw}

    def _save_preview(self, frame, mask):
        x, y, w, h = cv2.boundingRect(mask)
        pad = max(w, h) // 4
        crop = frame.color[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        cv2.imwrite(str(self.cfg.data_dir / "live_preview.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def _gas_curve(self, t0, baseline):
        s = self.dev.gas.window(t0, time.time())
        if not s:
            return [], None
        base = baseline if baseline is not None else s[0].raw
        r = [self.dev.gas.response(x.raw, base) for x in s]
        r = [v for v in r if math.isfinite(v)]
        if len(r) > 60:
            idx = np.linspace(0, len(r) - 1, 60).astype(int)
            r = [r[i] for i in idx]
        return [round(v, 4) for v in r], (round(r[-1] - 1, 4) if r else None)

    def _recipe(self, scan_id, fruit, pred, days, mode):
        try:
            recipe = get_recipe(fruit, pred.stage, days, mode, self.cfg)
        except Exception as e:  # pragma: no cover
            log.warning("recipe failed: %s", e)
            recipe = None
        self.store.set_recipe(scan_id, recipe)
        with self._lock:
            if self.state.get("scan_id") == scan_id and self.state.get("result"):
                upd = {"recipe_status": "ready" if recipe else "none"}
                if recipe:
                    upd.update(recipe_title=recipe["title"], recipe_source=recipe["source"])
                    if recipe["source"].startswith("Gemini") and not pred.uncertain:
                        upd.update(tip=recipe["tip"], tip_source="gemini")
                self.state["result"] = {**self.state["result"], **upd}

    def _check_door(self):
        if self.dev.io.door_closed() is False:
            raise Aborted("Door opened: close it and tap to scan again")

    def _run(self, fruit_hint, fruit_id):
        io, gas, cfg = self.dev.io, self.dev.gas, self.cfg
        t_start = time.time()
        self._set(result=None, scan_id=None,
                  live={"token": int(t_start * 1000), "preview": False, "spectrum": None, "gas": [], "gas_rise": None})
        try:
            self._step("door", "Checking the door", 0.02)
            if io.door_closed() is False:
                raise Aborted("Close the door, then tap to scan")

            # gas: baseline from the empty chamber just before the door last opened
            opened, closed = io.door_opened_at, io.door_closed_at
            if opened and t_start - opened < 1800:
                baseline, how = gas.baseline(opened, closed)
            else:
                baseline, how = None, "none"
            gas_ok = gas.warm
            io.set("fan", 1)

            self._step("camera", "Photographing the fruit", 0.08)
            io.set("led_strip", 1)
            try:
                frame = self.dev.camera.capture()
            finally:
                io.set("led_strip", 0)
            self._check_door()
            empty_c, empty_d = calibration.empty_reference(cfg)
            mask, seg_method, floor = segment(frame.color, frame.depth, empty_c, empty_d)
            img_f, fruit_seen = image_features(frame, mask, floor, self.classifier)
            if not mask.any():
                raise Aborted("No fruit found: place it in the cradle and rescan")
            self._save_preview(frame, mask)
            self._live(preview=True)

            self._step("spectral", "Reading the colour spectrum", 0.25)
            meas = measure(self.dev.spectral, io, cfg["spectral"])
            calib = calibration.load(cfg).get("white")
            spec_f = spectral_features(meas, calib)
            self._live(spectrum=[round(float(spec_f[f"s_r{nm}"]), 4) for nm in WAVELENGTHS])
            self._check_door()

            # gas keeps accumulating; wait out the rest of the window with live progress
            total = cfg["gas"]["scan_seconds"]
            while (left := total - (time.time() - t_start)) > 0:
                self._check_door()
                self._step("gas", f"Sniffing for ripening gases: {math.ceil(left)} s", 0.35 + 0.55 * (1 - left / total))
                curve, rise = self._gas_curve(t_start, baseline)
                self._live(gas=curve, gas_rise=rise, seconds_left=math.ceil(left))
                time.sleep(min(0.5, left))
            io.set("fan", 0)
            samples = gas.window(t_start, time.time())
            if baseline is None and samples:
                baseline, how = samples[0].raw, "scan-start"
            gas_f = gas_features(samples, baseline, gas, img_f.get("i_volume_cm3", float("nan")))
            if not gas_ok:
                gas_f = {k: float("nan") for k in gas_f}

            self._step("model", "Thinking", 0.93)
            fruit = fruit_hint or fruit_seen or cfg["app"]["default_fruit"]
            features = {"fruit_code": float(fruit_code(fruit)), **img_f, **spec_f, **gas_f}
            pred = self.predictor.predict(features, fruit)
            if not gas_ok:
                pred.notes.append(f"Gas sensor warming up ({round(gas.warmup_left_s / 60)} min): not used")
            if calib is None:
                pred.notes.append("Spectral sensor not calibrated: run 'ripesense calibrate'")
            if meas["saturated"]:
                pred.notes.append("Spectral reading saturated: lower the gain")
            if how in ("scan-start", "before door opened"):
                pred.notes.append("No door-open baseline: gas reading is approximate")
            v = verdict(pred, fruit, self.mode, cfg["store"])
            gauge = sum(i * pred.probs.get(st, 0) for i, st in enumerate(["unripe", "ripe", "overripe", "spoiled"])) / 3
            result = {**v, "prediction": pred.to_dict(), "segmentation": seg_method, "gauge": round(gauge, 3),
                      "elapsed_s": round(time.time() - t_start, 1), "fruit_id": fruit_id,
                      "recipe_status": "none" if pred.stage == "spoiled" else "pending",
                      "readings": {"length_mm": img_f.get("i_length_mm"),
                                   "brown_pct": 100 * (img_f.get("c_brown_frac", 0) + img_f.get("c_dark_frac", 0)),
                                   "ripen_idx": spec_f.get("s_ripen"),
                                   "gas_pct": 100 * gas_f["g_rise"] if math.isfinite(gas_f.get("g_rise", float("nan"))) else None}}
            raw = {"spectral": meas, "gas_baseline": baseline, "gas_baseline_source": how,
                   "gas": [(s.t, s.raw) for s in samples], "floor_m": floor}
            scan_id = self.store.add(fruit_type=fruit, fruit_id=fruit_id, pred=pred, mode=self.mode,
                                     features={k: features.get(k) for k in FEATURES}, raw=raw, result=result)
            self.store.save_images(scan_id, frame, mask)
            if self.sink:
                try:
                    self.sink.write(scan_id, v, pred)
                except Exception as e:  # pragma: no cover
                    log.warning("sink write failed: %s", e)
            self._set(status="result", step=None, progress=1.0, message="", result=result, scan_id=scan_id,
                      finished_at=time.time())
            log.info("scan %s: %s %s (%.0f%%) in %.1fs", scan_id, fruit, pred.stage, 100 * pred.confidence,
                     result["elapsed_s"])
            if pred.stage != "spoiled":   # recipe + QR arrive a few seconds after the result
                threading.Thread(target=self._recipe, daemon=True, name="recipe",
                                 args=(scan_id, fruit, pred, v.get("days"), self.mode)).start()
        except Aborted as e:
            self._set(status="error", step=None, message=str(e), progress=0.0, finished_at=time.time())
        except Exception as e:  # pragma: no cover
            log.error("scan failed: %s\n%s", e, traceback.format_exc())
            self._set(status="error", step=None, message=f"Scan failed: {e}", progress=0.0, finished_at=time.time())
        finally:
            io.all_off()
