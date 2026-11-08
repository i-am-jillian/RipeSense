"""Scan log in SQLite (data/ripesense.db) plus raw images, and CSV export for training."""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .features import FEATURES

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  fruit_type TEXT, fruit_id TEXT,
  stage TEXT, days_low REAL, days_mid REAL, days_high REAL, confidence REAL, uncertain INTEGER,
  mode TEXT, predictor TEXT,
  features TEXT, raw TEXT, result TEXT,
  label_stage TEXT, label_days REAL,
  synthetic INTEGER DEFAULT 0
);
"""


class Store:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)
        (self.dir / "scans").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.dir / "ripesense.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(scans)")}
        if "recipe" not in cols:   # added in v0.2
            self.db.execute("ALTER TABLE scans ADD COLUMN recipe TEXT")
            self.db.commit()
        self.lock = threading.Lock()

    def add(self, *, fruit_type, fruit_id, pred, mode, features, raw, result, synthetic=False,
            label_stage=None, label_days=None, ts=None) -> int:
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO scans (ts, fruit_type, fruit_id, stage, days_low, days_mid, days_high, confidence, "
                "uncertain, mode, predictor, features, raw, result, label_stage, label_days, synthetic) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ts or time.time(), fruit_type, fruit_id, pred.stage, _num(pred.days_low), _num(pred.days_mid),
                 _num(pred.days_high), pred.confidence, int(pred.uncertain), mode, pred.mode,
                 json.dumps(_clean(features)), json.dumps(_clean(raw)), json.dumps(_clean(result)),
                 label_stage, label_days, int(synthetic)))
            self.db.commit()
            return int(cur.lastrowid)

    def save_images(self, scan_id: int, frame, mask) -> None:
        d = self.dir / "scans"
        cv2.imwrite(str(d / f"{scan_id}_color.jpg"), frame.color, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if frame.depth is not None:
            np.save(d / f"{scan_id}_depth.npy", frame.depth.astype(np.float16))
        if mask is not None:
            cv2.imwrite(str(d / f"{scan_id}_mask.png"), mask)
        thumb = frame.color
        if mask is not None and mask.any():
            x, y, w, h = cv2.boundingRect(mask)
            pad = 20
            thumb = frame.color[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        th = cv2.resize(thumb, (160, int(160 * thumb.shape[0] / max(thumb.shape[1], 1))))
        cv2.imwrite(str(d / f"{scan_id}_thumb.jpg"), th)

    def set_recipe(self, scan_id: int, recipe: dict | None):
        with self.lock:
            self.db.execute("UPDATE scans SET recipe=? WHERE id=?", (json.dumps(recipe) if recipe else None, scan_id))
            self.db.commit()

    def get_recipe(self, scan_id: int) -> dict | None:
        r = self.db.execute("SELECT recipe FROM scans WHERE id=?", (scan_id,)).fetchone()
        return json.loads(r[0]) if r and r[0] else None

    def thumb_path(self, scan_id: int) -> Path:
        return self.dir / "scans" / f"{scan_id}_thumb.jpg"

    def label(self, scan_id: int, stage: str | None, days: float | None, fruit_id: str | None = None,
              fruit_type: str | None = None):
        with self.lock:
            self.db.execute("UPDATE scans SET label_stage=?, label_days=?, fruit_id=COALESCE(?, fruit_id), "
                            "fruit_type=COALESCE(?, fruit_type) WHERE id=?",
                            (stage, days, fruit_id, fruit_type, scan_id))
            self.db.commit()

    def recent(self, limit=50) -> list[dict]:
        rows = self.db.execute("SELECT id, ts, fruit_type, fruit_id, stage, days_low, days_mid, days_high, "
                               "confidence, uncertain, predictor, label_stage, label_days, synthetic "
                               "FROM scans ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get(self, scan_id: int) -> dict | None:
        r = self.db.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        for k in ("features", "raw", "result", "recipe"):
            d[k] = json.loads(d[k]) if d[k] else None
        return d

    def training_rows(self, include_synthetic=False) -> list[dict]:
        q = "SELECT fruit_type, fruit_id, features, label_stage, label_days FROM scans WHERE label_stage IS NOT NULL"
        if not include_synthetic:
            q += " AND synthetic = 0"
        rows = []
        for r in self.db.execute(q):
            f = {k: (v if v is not None else float("nan")) for k, v in json.loads(r["features"]).items()}
            rows.append({"features": f, "stage": r["label_stage"], "days_left": r["label_days"],
                         "fruit_id": r["fruit_id"], "fruit_type": r["fruit_type"]})
        return rows

    def export_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["scan_id", "ts", "fruit_type", "fruit_id", "label_stage", "label_days", "pred_stage",
                    "pred_days_mid", "confidence", "synthetic"] + FEATURES)
        for r in self.db.execute("SELECT * FROM scans ORDER BY id"):
            f = json.loads(r["features"] or "{}")
            w.writerow([r["id"], r["ts"], r["fruit_type"], r["fruit_id"], r["label_stage"], r["label_days"],
                        r["stage"], r["days_mid"], r["confidence"], r["synthetic"]] + [f.get(k) for k in FEATURES])
        return buf.getvalue()


def _num(x):
    return None if x is None or (isinstance(x, float) and x != x) else x


def _clean(o):
    """JSON-safe: NaN -> None, numpy -> python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and o != o:
        return None
    return o
