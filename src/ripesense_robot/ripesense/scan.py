"""Scan controller: one tap runs camera → spectral while gas accumulates in
parallel, then features → fusion model → result (doc p.3-4 flowchart)."""
from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from datetime import datetime

import numpy as np

from . import config, fruit_info, store
from . import forecast as fc
from .cloud import gemini, tiger
from .hw import camera as cam_mod
from .hw import gas as gas_mod
from .hw import gpio as gpio_mod
from .hw import spectral as spec_mod
from .models.featurize import row_features
from .models.image import Models
from .models.infer import Predictor
from .sim.fruit import SPECTRAL_CHANNELS

log = logging.getLogger(__name__)
BASELINE_MAX_AGE_S = 30 * 60


class Controller:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or config.load()
        self.mock = bool(self.cfg["mock"])
        self.fast = os.environ.get("RIPESENSE_FAST", "") in ("1", "true")
        # Camera-only: just a USB webcam (laptop demo, or the RealSense-failed fallback).
        # No GPIO, spectral or gas; scans start from the on-screen button.
        self.camera_only = bool(self.cfg.get("camera_only"))
        if self.camera_only:
            self.cfg["gpio"]["require_door"] = False
            self.io = gpio_mod.MockIO(self.cfg)
            self.camera = cam_mod.make_camera(self.cfg, False)
            self.spectral = self.gas = None
        else:
            self.io = gpio_mod.make_io(self.cfg, self.mock)
            self.camera = cam_mod.make_camera(self.cfg, self.mock)
            self.spectral = self._try(lambda: spec_mod.make_spectral(self.cfg, self.mock), "spectral")
            self.gas = self._try(lambda: gas_mod.make_gas(self.cfg, self.mock), "gas")
        if self.gas is not None and self.mock and self.fast:
            self.gas.time_scale = 15.0
        self.monitor = gas_mod.GasMonitor(self.gas, hz=self.cfg["gas"]["sample_hz"]).start() if self.gas else None
        self.models = Models(config.models_dir())
        log.info("pre-trained models: %s", self.models.summary())
        self.predictor = Predictor(self.cfg)
        self.white_ref_path = config.data_dir() / self.cfg["spectral"]["white_ref_file"]

        self.lock = threading.Lock()
        self.status = {"state": "idle", "progress": 0.0, "message": "Place fruit, close door, tap"}
        self.result: dict | None = None
        self.mode = "home"
        self.collect: dict = {}       # fruit_id / fruit_type / stage_label for data collection
        self.baseline_ts = 0.0
        # Sensors switched off from the dev page ("pull the cable" demo). A sensor
        # that really fails mid-scan is handled the same way.
        self.disabled: set[str] = set()
        self.last_embedding = None    # image embedding of the last scan (for "teach")
        if hasattr(self.camera, "ir_light"):
            self.camera.ir_light = lambda on: self.io.set("spec_ir", on)

        self.io.on_touch(lambda: self.start_scan())
        self.io.on_door_open(self._door_opened)
        self.io.all_off()

    def _try(self, fn, name):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            log.error("%s sensor unavailable: %s — scans continue without it", name, e)
            return None

    # ------------------------------------------------------------- door / purge
    def _door_opened(self):
        # The chamber is empty only if no fruit result is on screen.
        if self.monitor and self.status["state"] in ("idle", "close_door"):
            self.monitor.mark_door_open()
            self.baseline_ts = time.time()
        if self.status["state"] == "result":
            self._set("idle", 0, "Place fruit, close door, tap")
        threading.Thread(target=self._purge, daemon=True).start()

    def _purge(self):
        if self.status["state"] == "scanning":
            return
        self.io.set("fan", True)
        time.sleep(self.cfg["scan"]["fan_purge_seconds"])
        if self.status["state"] != "scanning":
            self.io.set("fan", False)

    # ------------------------------------------------------------------ status
    def _set(self, state, progress, message):
        self.status = {"state": state, "progress": round(progress, 3), "message": message}

    def snapshot(self) -> dict:
        return {**self.status, "result": self.result, "mode": self.mode, "mock": self.mock,
                "camera_only": self.camera_only, "models": self.models.summary(),
                "trained": self.predictor.trained, "collect": self.collect,
                "door_closed": self.io.door_closed(),
                "calibrated": self.white_ref_path.exists(),
                "sensors_present": {"camera": "camera" not in self.disabled,
                                    "spectral": self.spectral is not None and "spectral" not in self.disabled,
                                    "gas": self.gas is not None and "gas" not in self.disabled},
                "disabled": sorted(self.disabled)}

    # -------------------------------------------------------------------- scan
    def start_scan(self, meta: dict | None = None) -> bool:
        if not self.lock.acquire(blocking=False):
            return False
        if self.io.require_door and not self.io.door_closed():
            self.lock.release()
            self._set("close_door", 0, "Close the door")
            return False
        threading.Thread(target=self._run, args=(meta or {},), daemon=True).start()
        return True

    def _run(self, meta: dict):
        try:
            self.result = None
            self._scan(meta)
        except Exception as e:  # noqa: BLE001
            log.error("scan failed: %s\n%s", e, traceback.format_exc())
            self._set("error", 0, f"Scan failed: {e}")
        finally:
            self.io.all_off()
            if self.mock and self.gas is not None:
                self.gas.seal(False)
            self.lock.release()

    def _illum(self, mode: str):
        self.io.set("spec_white", mode == "white")
        self.io.set("spec_ir", mode == "ir")
        if hasattr(self.spectral, "set_illumination"):
            self.spectral.set_illumination(mode)
        time.sleep(0.05 if self.mock else 0.3)  # LED settle

    def _scan(self, meta: dict):
        cfg = self.cfg
        read_s = 8 if self.fast else cfg["gas"]["read_seconds"]
        t0 = time.time()
        ts = datetime.now()
        self._set("scanning", 0.02, "Starting…")

        use_gas = self.monitor is not None and "gas" not in self.disabled
        use_spec = self.spectral is not None and "spectral" not in self.disabled
        offline = [s for s in ("spectral", "gas") if s in self.disabled]
        baseline = base_profile = None
        if self.monitor and self.monitor.empty_baseline and time.time() - self.baseline_ts < BASELINE_MAX_AGE_S:
            baseline = self.monitor.empty_baseline
            base_profile = self.monitor.empty_profile
        if use_gas:
            self.io.set("fan", True)
            if self.mock:
                self.gas.seal(True)

        # 1 · Camera, LED strip on.
        self._set("scanning", 0.05, "Camera…")
        self.io.set("led_strip", True)
        time.sleep(0.1 if self.mock else 0.4)
        frame = None
        try:
            if "camera" not in self.disabled:
                frame = self.camera.capture()
        except Exception as e:  # noqa: BLE001
            log.error("camera failed: %s", e)
            offline.append("camera")
        self.io.set("led_strip", False)

        # 2 · Spectral in the dark: dark → white LEDs → IR LED.
        self._set("scanning", 0.15, "Spectral…")
        spec = {}
        if use_spec:
            try:
                self._illum("off")
                dark = self.spectral.read()
                self._illum("white")
                white_lit = self.spectral.read()
                self._illum("ir")
                ir_lit = self.spectral.read()
                self._illum("off")
                sample = spec_mod.combine_illuminations(white_lit, ir_lit)
                ref = spec_mod.load_white_ref(self.white_ref_path)
                refl = (spec_mod.reflectance(sample, dark, ref["white"], ref["dark"]) if ref
                        else spec_mod.reflectance(sample, dark, {k: dark[k] + 1000.0 for k in dark}))
                spec = {"refl": refl, "dark": dark, "white_lit": white_lit, "ir_lit": ir_lit,
                        "white_ref": ref["white"] if ref else None}
            except Exception as e:  # noqa: BLE001
                log.error("spectral failed: %s", e)
                offline.append("spectral")

        # 3 · Gas: wait until read_s has elapsed since t0.
        series, gas_raw_end, temp, hum, gas_profile = [], None, None, None, None
        if use_gas:
            while (el := time.time() - t0) < read_s:
                self._set("scanning", 0.2 + 0.7 * el / read_s, f"Reading gas… {int(read_s - el)} s")
                time.sleep(0.25)
            pts = self.monitor.window(t0, time.time())
            if pts:
                if baseline is None:  # no empty-chamber baseline: first 3 s
                    first = sorted(r["raw"] for t, r in pts if t - t0 <= 3) or [pts[0][1]["raw"]]
                    baseline = first[len(first) // 2]
                kind = self.gas.kind
                ts_scale = getattr(self.gas, "time_scale", 1.0)  # fast mock compresses time
                series = [(round((t - t0) * ts_scale, 2), gas_mod.response(r["raw"], baseline, kind))
                          for t, r in pts]
                gas_raw_end = pts[-1][1]["raw"]
                temp, hum = pts[-1][1].get("temp"), pts[-1][1].get("humidity")
                # E-nose profile: empty-chamber (or first 3 s) vs the last 10 s.
                if base_profile is None:
                    base_profile = gas_mod.median_profile([r for t, r in pts if t - t0 <= 3])
                end_profile = gas_mod.median_profile([r for t, r in pts if t >= pts[-1][0] - 10])
                if base_profile and end_profile:
                    gas_profile = {"baseline": base_profile, "end": end_profile}
            else:
                offline.append("gas")
            self.io.set("fan", False)

        # 4 · Features + models.
        self._set("scanning", 0.93, "Thinking…")
        key = f"{ts:%Y%m%d_%H%M%S}"
        rgb_p, depth_p = store.save_images(key, frame.bgr if frame else None, frame.depth_mm if frame else None)
        ir_p, ir_dp = store.save_ir(key, getattr(frame, "ir", None), getattr(frame, "ir_depth_mm", None))
        refl = spec.get("refl") or {}
        given_type = meta.get("fruit_type") or self.collect.get("fruit_type") or ""
        if self.mock and not given_type:   # the sim's drawn ellipse is no photo: use its known type
            from .sim import fruit as simf
            given_type = simf.current.fruit_type
        row = {
            "fruit_id": meta.get("fruit_id") or self.collect.get("fruit_id", ""),
            "fruit_type": given_type,
            "timestamp": ts.isoformat(timespec="seconds"),
            "session": store.session_name(ts),
            "mode": self.mode,
            "rgb_path": rgb_p, "depth_path": depth_p,
            **{f"spec_{c}": refl.get(c) for c in SPECTRAL_CHANNELS},
            "spec_white_lit": spec.get("white_lit"), "spec_ir_lit": spec.get("ir_lit"),
            "white_ref": spec.get("white_ref"), "dark_ref": spec.get("dark"),
            "gas_kind": self.gas.kind if self.gas else "", "gas_profile": gas_profile,
            "ir_path": ir_p, "ir_depth_path": ir_dp,
            "gas_baseline": baseline, "gas_120s": gas_raw_end, "gas_series": series,
            "temp": temp, "humidity": hum,
            "stage_label": meta.get("stage_label") or self.collect.get("stage_label", ""),
            "days_left_label": "",
            "source": frame.source if frame else "",
        }
        fr = ((frame.bgr, frame.depth_mm, frame.fx, frame.fy, frame.ir, frame.ir_depth_mm, frame.ir_fx)
              if frame else None)
        feats, aux = row_features(row, cfg, self.models, frame=fr)
        self.last_embedding = aux.get("embedding")
        img_pred = aux.get("image")
        fs = aux.get("fewshot")
        taught = fs["fruit"] if fs and fs.get("accepted") else ""
        fruit_type = given_type or taught or (img_pred["fruit_type"] if img_pred and img_pred["fruit_conf"] >= 0.4 else "")
        fruit_type = fruit_type or ("unknown" if img_pred else "banana")
        row["fruit_type"] = fruit_type
        res = self.predictor.predict(feats, fruit_type, aux)
        score = fruit_info.stage_score(res["probs"])
        fused = res["fused_stage"]

        # Forecast from this fruit's scan history (+ this scan). In the pretrained
        # tier it also replaces the fixed stage→days lookup with a personal range.
        fcast = whatif = None
        if res["stage"] not in ("uncertain", "spoiled") and fruit_type != "unknown":
            hist = fc.history_from_rows(store.rows(where="WHERE fruit_id=?", params=(row["fruit_id"],))) \
                if row["fruit_id"] else []
            hist.append({"t": ts, "q": score, "conf": res["confidence"], "temp": temp})
            try:
                fcast = fc.forecast(hist, fruit_type, temp_now=temp)
                whatif = fc.whatifs(hist, fruit_type, temp_now=temp)
                if res["tier"] != "fusion":
                    dl = fcast["days_left"]
                    res["days"] = {"p10": dl[0], "p50": dl[1], "p90": dl[2], "guarantee": None, "source": "forecast"}
                    from .stages import store_info
                    res["store"] = store_info(res["stage"], fruit_type, cfg, days=res["days"])
            except Exception as e:  # noqa: BLE001
                log.warning("forecast failed: %s", e)
        d = res["days"] or {}
        row.update(pred_stage=res["stage"], pred_conf=round(res["confidence"], 4), pred_score=round(score, 3),
                   pred_days_p10=d.get("p10"), pred_days_p50=d.get("p50"), pred_days_p90=d.get("p90"),
                   agreement=res["sensors"])
        scan_id = store.insert(row)

        ir_view = None
        if aux.get("ir_overlay") is not None:
            ir_view = "/data/" + store.save_view(f"{key}_irview", aux["ir_overlay"])
        prof = None
        if fruit_type != "unknown" and res["stage"] != "uncertain":
            prof = fruit_info.profile(fruit_type, fused, score, feats)
        offline = sorted(set(offline))
        if offline:
            res["basis"] = res["basis"] + [f"{', '.join(offline)} offline: verdict from the remaining sensors"]
        res.update(scan_id=scan_id, fruit_type=fruit_type, fruit_id=row["fruit_id"],
                   image=f"/data/{rgb_p}" if rgb_p else None,
                   image_model=img_pred, stage_model=aux.get("stage"), spectral_prior=aux.get("spec_prior"),
                   fewshot=fs, score=round(score, 3), forecast=fcast, whatif=whatif, profile=prof,
                   ir={"image": ir_view, **{k: feats.get(k) for k in ("ir_bruise_count", "ir_bruise_pct",
                                                                       "ir_bruise_area_cm2")}} if ir_view else None,
                   offline=offline, why=None,
                   timestamp=row["timestamp"], tip=None, tip_source=None,
                   features={k: (None if isinstance(v, float) and np.isnan(v) else round(v, 4))
                             for k, v in feats.items()},
                   elapsed_s=round(time.time() - t0, 1))
        self.result = res
        self._set("result", 1.0, res["headline"])
        threading.Thread(target=self._after, args=(res, row, frame, aux.get("mask"), key), daemon=True).start()

    def _after(self, res: dict, row: dict, frame=None, mask=None, key: str = ""):
        # "Why" heatmap first (local, ~1 s), then the Gemini tip (network).
        if frame is not None and res["fruit_type"] != "unknown":
            try:
                from .models.explain import explain
                ex = explain(self.models, frame.bgr, mask, res["fruit_type"], res["fused_stage"])
                if ex:
                    p = store.save_view(f"{key}_why", ex.pop("image"))
                    res["why"] = {**ex, "image": f"/data/{p}"}
            except Exception as e:  # noqa: BLE001
                log.warning("why-map failed: %s", e)
        tip, src = gemini.tip(res["fruit_type"], res, self.cfg)
        if self.result is res:
            res["tip"], res["tip_source"] = tip, src
        st = res["store"]
        md = st.get("markdown_pct") or [None, None]
        tiger.log_scan(self.cfg, {
            "ts": datetime.fromisoformat(row["timestamp"]).astimezone(), "device": os.uname().nodename,
            "scan_id": res["scan_id"], "fruit_type": res["fruit_type"], "stage": res["stage"],
            "confidence": res["confidence"], "days_p50": (res["days"] or {}).get("p50"),
            "md_lo": md[0], "md_hi": md[1], "price": st.get("price"), "base_price": st.get("base_price")})

    # ------------------------------------------------------------------- teach
    def teach(self, label: str, n: int = 5, from_last: bool = False) -> dict:
        """Few-shot: add n views of the fruit in front of the camera under `label`
        ("kiwi" or "kiwi:ripe"). Turn the fruit between captures."""
        fsm = self.models.fewshot
        if fsm is None:
            raise RuntimeError("image model with embedding output not loaded")
        if not label.strip():
            raise ValueError("label required")
        from . import features as feat_mod
        added = 0
        if from_last and self.last_embedding is not None:
            fsm.add(label, self.last_embedding)
            added = 1
        else:
            cam = self.cfg["camera"]
            with self.lock:
                self.io.set("led_strip", True)
                try:
                    for i in range(n):
                        self._set("teaching", (i + 0.5) / n, f"Teaching '{label}': view {i + 1}/{n} — turn the fruit")
                        f = self.camera.capture()
                        m = feat_mod.fruit_mask(f.bgr, f.depth_mm, cam["tray_depth_mm"], cam["mask_min_height_mm"])
                        e = self.models.image.embed(f.bgr, m)
                        if e is not None:
                            fsm.add(label, e)
                            added += 1
                        time.sleep(0.6 if not self.mock else 0.05)
                finally:
                    self.io.set("led_strip", False)
                    self._set("idle", 0, "Place fruit, close door, tap")
        return {"label": label, "added": added, "taught": fsm.summary()}

    def reload_models(self):
        self.models = Models(config.models_dir())
        self.predictor.reload()

    def preview_jpeg(self) -> bytes | None:
        """Latest camera frame as JPEG, for aiming the camera (webcam backends)."""
        import cv2
        grab = getattr(self.camera, "latest", None)
        frame = grab() if grab else None
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    # ------------------------------------------------------------- calibration
    def calibrate_white(self) -> dict:
        """Place the white puck in the cradle, door shut, then call this."""
        if self.spectral is None:
            raise RuntimeError("spectral sensor not available")
        with self.lock:
            if hasattr(self.spectral, "target"):
                self.spectral.target = "white"
            try:
                self._illum("off")
                dark = self.spectral.read(5)
                self._illum("white")
                w = self.spectral.read(5)
                self._illum("ir")
                ir = self.spectral.read(5)
                self._illum("off")
            finally:
                if hasattr(self.spectral, "target"):
                    self.spectral.target = "fruit"
            ref = {"dark": dark, "white": spec_mod.combine_illuminations(w, ir),
                   "timestamp": datetime.now().isoformat(timespec="seconds")}
            spec_mod.save_white_ref(self.white_ref_path, ref)
            return ref
