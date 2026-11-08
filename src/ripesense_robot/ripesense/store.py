"""Scan log: one SQLite row + one CSV row per scan, plus raw images on disk so
features can be re-extracted later without rescanning (doc p.14)."""
from __future__ import annotations

import csv
import json
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np

from . import config
from .sim.fruit import SPECTRAL_CHANNELS

CSV_COLUMNS = (
    ["scan_id", "fruit_id", "fruit_type", "timestamp", "session", "mode", "rgb_path", "depth_path"]
    + [f"spec_{c}" for c in SPECTRAL_CHANNELS]
    + ["spec_white_lit", "spec_ir_lit", "white_ref", "dark_ref",
       "gas_kind", "gas_baseline", "gas_120s", "gas_series", "temp", "humidity",
       "stage_label", "days_left_label",
       "pred_stage", "pred_conf", "pred_days_p10", "pred_days_p50", "pred_days_p90", "agreement",
       "saved", "source",
       # added for the forecast / IR / e-nose upgrades (appended so old CSVs still line up)
       "pred_score", "ir_path", "ir_depth_path", "gas_profile"]
)

_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(config.data_dir() / "ripesense.db", check_same_thread=False)
    con.row_factory = sqlite3.Row
    cols = ", ".join(f'"{c}" TEXT' for c in CSV_COLUMNS if c != "scan_id")
    con.execute(f'CREATE TABLE IF NOT EXISTS scans (scan_id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})')
    have = {r[1] for r in con.execute("PRAGMA table_info(scans)")}
    for c in CSV_COLUMNS:            # migrate databases made before a column existed
        if c not in have:
            con.execute(f'ALTER TABLE scans ADD COLUMN "{c}" TEXT')
    return con


def session_name(ts: datetime) -> str:
    return f"{ts:%Y-%m-%d}-{'AM' if ts.hour < 12 else 'PM'}"


def save_ir(scan_key: str, ir: np.ndarray | None, ir_depth: np.ndarray | None) -> tuple[str, str]:
    if ir is None:
        return "", ""
    img_dir = config.data_dir() / "images"
    img_dir.mkdir(exist_ok=True)
    ir_p = f"images/{scan_key}_ir.png"
    cv2.imwrite(str(config.data_dir() / ir_p), ir)
    d_p = ""
    if ir_depth is not None:
        d_p = f"images/{scan_key}_irdepth.png"
        cv2.imwrite(str(config.data_dir() / d_p), ir_depth.astype(np.uint16))
    return ir_p, d_p


def save_view(name: str, img: np.ndarray) -> str:
    """Derived images (why-heatmap, IR overlay, future preview) → relative path."""
    img_dir = config.data_dir() / "images"
    img_dir.mkdir(exist_ok=True)
    p = f"images/{name}.jpg"
    cv2.imwrite(str(config.data_dir() / p), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return p


def save_images(scan_key: str, bgr: np.ndarray | None, depth: np.ndarray | None) -> tuple[str, str]:
    img_dir = config.data_dir() / "images"
    img_dir.mkdir(exist_ok=True)
    rgb_p = depth_p = ""
    if bgr is not None:
        rgb_p = f"images/{scan_key}_rgb.jpg"
        cv2.imwrite(str(config.data_dir() / rgb_p), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if depth is not None:
        depth_p = f"images/{scan_key}_depth.png"  # 16-bit PNG, lossless mm
        cv2.imwrite(str(config.data_dir() / depth_p), depth.astype(np.uint16))
    return rgb_p, depth_p


def insert(row: dict) -> int:
    row = {k: _fmt(row.get(k)) for k in CSV_COLUMNS if k != "scan_id"}
    with _lock:
        con = _db()
        keys = list(row)
        cur = con.execute(
            f'INSERT INTO scans ({",".join(chr(34)+k+chr(34) for k in keys)}) VALUES ({",".join("?"*len(keys))})',
            [row[k] for k in keys])
        con.commit()
        scan_id = cur.lastrowid
        con.close()
    export_csv()
    return scan_id


def update(scan_id: int, **fields):
    with _lock:
        con = _db()
        for k, v in fields.items():
            if k not in CSV_COLUMNS or k == "scan_id":
                raise KeyError(k)
            con.execute(f'UPDATE scans SET "{k}"=? WHERE scan_id=?', (_fmt(v), scan_id))
        con.commit()
        con.close()
    export_csv()


def rows(limit: int | None = None, where: str = "", params=()) -> list[dict]:
    with _lock:
        con = _db()
        q = f"SELECT * FROM scans {where} ORDER BY scan_id DESC" + (f" LIMIT {int(limit)}" if limit else "")
        out = [dict(r) for r in con.execute(q, params)]
        con.close()
    return out


def get(scan_id: int) -> dict | None:
    r = rows(where="WHERE scan_id=?", params=(scan_id,))
    return r[0] if r else None


def export_csv(path: Path | None = None):
    """Rewrite scans.csv from SQLite so edits to labels show up in the CSV."""
    path = path or config.data_dir() / "scans.csv"
    data = list(reversed(rows()))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in data:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def mark_spoiled(fruit_id: str, spoiled_on: date) -> int:
    """Back-fill days_left_label for every scan of a fruit: days from the scan to
    the day it was judged spoiled (doc p.13, step 3). Returns rows updated."""
    n = 0
    for r in rows(where="WHERE fruit_id=?", params=(fruit_id,)):
        d = datetime.fromisoformat(r["timestamp"]).date()
        update(int(r["scan_id"]), days_left_label=max(0, (spoiled_on - d).days))
        n += 1
    return n


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v)
    if isinstance(v, float):
        return "" if v != v else repr(v)
    return str(v)
