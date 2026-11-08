"""Convert DeepHS Fruit hyperspectral scans into simulated AS7341 readings
(doc p.14: "average its bands down to the AS7341's 8 channels").

  python -m ripesense.models.prep_deephs datasets/raw/deephs/annotations-upd-2024-01-09.zip \\
      datasets/raw/deephs/Mango.zip datasets/raw/deephs/Kaki.zip --out datasets/deephs_as7341.csv

Each labelled record's cube (64×64 px, float32 reflectance, background = 0)
becomes one row: the median spectrum of the fruit pixels, integrated with
Gaussian AS7341 channel responses. Ripeness labels: unripe / perfect→ripe /
overripe. fruit_id groups front/back and repeat days of the same fruit so
cross-validation never sees one fruit on both sides.

Source: Varga, Makowski, Zell, "Measuring the Ripeness of Fruit with
Hyperspectral Imaging and Deep Learning" (IJCNN 2021); data from
cogsys.cs.uni-tuebingen.de. Check the repo's licence before redistributing.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path

import numpy as np

from .spectral_prior import to_as7341
from ..sim.fruit import SPECTRAL_CHANNELS

LABEL = {"unripe": "unripe", "perfect": "ripe", "ripe": "ripe", "overripe": "overripe"}


def parse_hdr(text: str) -> dict:
    d = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip().lower()] = v.strip()
    return d


def load_cube(z: zipfile.ZipFile, data_file: str):
    hdr = parse_hdr(z.read(data_file[:-4] + ".hdr").decode())
    dt = {"4": "f4", "5": "f8", "2": "i2", "12": "u2"}[hdr["data type"]]
    bo = "<" if hdr.get("byte order", "0") == "0" else ">"
    s, l, b = int(hdr["samples"]), int(hdr["lines"]), int(hdr["bands"])
    x = np.frombuffer(z.read(data_file), bo + dt).astype(np.float32)
    il = hdr.get("interleave", "bip").lower()
    if il == "bip":
        return x.reshape(l, s, b)
    if il == "bil":
        return x.reshape(l, b, s).transpose(0, 2, 1)
    return x.reshape(b, l, s).transpose(1, 2, 0)  # bsq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations")
    ap.add_argument("zips", nargs="+")
    ap.add_argument("--out", default="datasets/deephs_as7341.csv")
    a = ap.parse_args()
    ann = zipfile.ZipFile(a.annotations)
    zips = {}
    for zp in a.zips:
        z = zipfile.ZipFile(zp)
        for n in z.namelist():
            zips[n] = z
    rows, seen = [], set()
    for split, f in (("train", "annotations/train_all_v2.json"), ("val", "annotations/val_v2.json"),
                     ("test", "annotations/test_v2.json")):
        d = json.loads(ann.read(f))
        cams = {c["id"]: np.array(c["wavelengths"], float) for c in d["cameras"]}
        recs = {r["id"]: r for r in d["records"]}
        for an in d["annotations"]:
            r = recs.get(an["record_id"])
            lab = LABEL.get(an.get("ripeness_state") or "")
            if not r or not lab or r["camera_type"] not in ("VIS", "VIS_COR"):
                continue
            df = r["files"]["data_file"]
            if df not in zips or df in seen:
                continue
            seen.add(df)
            cube = load_cube(zips[df], df)
            wl = cams[r["camera_type"]]
            fruit_px = cube.reshape(-1, cube.shape[2])
            fruit_px = fruit_px[fruit_px.mean(1) > 0.02]
            if len(fruit_px) < 50:
                continue
            spec = np.median(fruit_px, axis=0)
            ch = to_as7341(wl, spec)
            stem = Path(df).stem  # e.g. mango_day_9_m3_38_back
            m = re.match(r"([a-z]+)_day_\w+?_(m\d+)_(\d+)_", stem)
            fid = f"{r['fruit'].lower()}_{m.group(2)}_{m.group(3)}" if m else stem
            rows.append({"fruit": r["fruit"].lower(), "camera": r["camera_type"], "fruit_id": fid, "split": split,
                         "label": lab, "storage_days": an.get("storage_days"), "firmness": an.get("firmness"),
                         **{c: round(ch.get(c, float("nan")), 5) for c in SPECTRAL_CHANNELS}})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"{len(rows)} rows → {out}")
    print(Counter((r["fruit"], r["camera"], r["label"]) for r in rows))
    print("distinct fruits:", len({r["fruit_id"] for r in rows}))


if __name__ == "__main__":
    main()
