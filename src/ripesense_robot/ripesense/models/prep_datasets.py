"""Build training folders for the image models from the public datasets in the
handoff (p.13), reading the downloaded .zip files directly (no unpacking: the
originals are several GB of high-res photos).

  # Ripeness stage: "Annotated Banana Ripeness Stage Dataset" (Mendeley 56c75hvm7r, CC BY 4.0)
  python -m ripesense.models.prep_datasets banana  datasets/raw/BANANA_DATASET75.zip --out datasets/banana_stages

  # Fruit type + fresh/rotten: "Fresh and Rotten Fruits" (Mendeley bdd69gyhv8, CC BY 4.0)
  #                       and  "FruitNet" (Mendeley b6fftwbr2v, CC BY 4.0)
  python -m ripesense.models.prep_datasets quality datasets/raw/FreshRotten_Original.zip datasets/raw/FruitNetDataset.zip \\
      --out datasets/fruit_quality

Output: <out>/{train,val,test}/<class>/*.jpg, long side ≤ --size px.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}
FRUITS = ["apple", "banana", "orange", "grape", "guava", "jujube", "pomegranate", "strawberry", "lime",
          "mango", "tomato", "avocado", "kiwi", "lemon", "peach", "pear", "papaya", "potato", "cucumber"]


def _decode(z: zipfile.ZipFile, name: str):
    buf = np.frombuffer(z.read(name), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _shrink(img, size):
    s = size / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return img


def _hash_split(key: str, val=0.1, test=0.1) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test else "val" if h < test + val else "train"


# ------------------------------------------------------------------ banana
def stage_name(name: str) -> str:
    s = re.sub(r"[^a-z]", "", name.lower())
    if "over" in s:
        return "overripe"
    if "unripe" in s or "green" in s or "raw" in s:
        return "unripe"
    if "rot" in s or "spoil" in s:
        return "spoiled"
    if "ripe" in s:
        return "ripe"
    return s


def prep_banana(zips, out: Path, size: int, pad: float, min_box: int, class_names=None):
    counts: Counter = Counter()
    for zp in zips:
        z = zipfile.ZipFile(zp)
        names = z.namelist()
        yml = [n for n in names if n.lower().endswith(".yaml")]
        classes = None
        for y in yml:
            d = yaml.safe_load(z.read(y))
            if isinstance(d, dict) and "names" in d:
                n = d["names"]
                classes = [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
                break
        classes = classes or class_names
        if classes is None:
            raise SystemExit(f"{zp}: no data.yaml; pass --classes (e.g. unripe,ripe,overripe)")
        print(f"{zp}: classes {classes} → {[stage_name(c) for c in classes]}")
        imgs = {PurePosixPath(n).stem: n for n in names if PurePosixPath(n).suffix.lower() in IMG_EXT}
        labels = [n for n in names if n.endswith(".txt") and "labels" in PurePosixPath(n).parts]
        for ln in labels:
            stem = PurePosixPath(ln).stem
            if stem not in imgs:
                continue
            split = next((SPLITS[p.lower()] for p in reversed(PurePosixPath(ln).parts) if p.lower() in SPLITS), "train")
            rows = [r.split() for r in z.read(ln).decode().splitlines() if len(r.split()) >= 5]
            if not rows:
                continue
            img = _decode(z, imgs[stem])
            if img is None:
                continue
            H, W = img.shape[:2]
            for k, parts in enumerate(rows):
                c = int(float(parts[0]))
                if len(parts) > 5:  # polygon label
                    xs, ys = [float(v) for v in parts[1::2]], [float(v) for v in parts[2::2]]
                    cx, cy, bw, bh = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
                else:
                    cx, cy, bw, bh = map(float, parts[1:5])
                bw, bh = bw * (1 + 2 * pad), bh * (1 + 2 * pad)
                x0, x1 = int(max(0, (cx - bw / 2) * W)), int(min(W, (cx + bw / 2) * W))
                y0, y1 = int(max(0, (cy - bh / 2) * H)), int(min(H, (cy + bh / 2) * H))
                if min(x1 - x0, y1 - y0) < min_box:
                    continue
                cls = stage_name(classes[c])
                d = out / split / cls
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{stem}_{k}.jpg"), _shrink(img[y0:y1, x0:x1], size), [cv2.IMWRITE_JPEG_QUALITY, 92])
                counts[(split, cls)] += 1
    return counts


# ----------------------------------------------------------------- quality
def quality_class(path: str) -> str | None:
    p = path.lower()
    if "mixed" in p or "augment" in p:
        return None
    rot = any(w in p for w in ("rotten", "bad", "spoil", "stale"))
    fresh = any(w in p for w in ("fresh", "good"))
    if rot == fresh:          # neither or both: ambiguous
        return None
    fruit = next((f for f in FRUITS if f in p.replace("pineapple", "")), None)
    if fruit is None:
        return None
    return f"{'rotten' if rot else 'fresh'}_{fruit}"


def prep_quality(zips, out: Path, size: int, max_per_class: int):
    counts: Counter = Counter()
    per_class: Counter = Counter()
    for zp in zips:
        z = zipfile.ZipFile(zp)
        members = [n for n in z.namelist() if PurePosixPath(n).suffix.lower() in IMG_EXT]
        # Deterministic shuffle so --max-per-class samples across the whole set.
        members.sort(key=lambda n: hashlib.md5(n.encode()).hexdigest())
        print(f"{zp}: {len(members)} images")
        for n in members:
            cls = quality_class(n)
            if cls is None or (max_per_class and per_class[cls] >= max_per_class):
                continue
            img = _decode(z, n)
            if img is None:
                continue
            split = _hash_split(Path(zp).name + "/" + n)
            d = out / split / cls
            d.mkdir(parents=True, exist_ok=True)
            key = hashlib.md5(n.encode()).hexdigest()[:12]
            cv2.imwrite(str(d / f"{Path(zp).stem[:8]}_{key}.jpg"), _shrink(img, size), [cv2.IMWRITE_JPEG_QUALITY, 90])
            counts[(split, cls)] += 1
            per_class[cls] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["banana", "quality"])
    ap.add_argument("zips", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--min-box", type=int, default=40)
    ap.add_argument("--max-per-class", type=int, default=0)
    ap.add_argument("--classes", help="class names by id, if the zip has no data.yaml. "
                    "BANANA_DATASET75.zip: unripe,ripe,overripe (checked against its file names)")
    a = ap.parse_args()
    out = Path(a.out)
    if a.kind == "banana":
        counts = prep_banana(a.zips, out, a.size, a.pad, a.min_box, a.classes.split(",") if a.classes else None)
    else:
        counts = prep_quality(a.zips, out, a.size, a.max_per_class)
    classes = sorted({c for _, c in counts})
    print(f"{'':8s}" + "".join(f"{c[:14]:>15s}" for c in classes))
    for s in ("train", "val", "test"):
        print(f"{s:8s}" + "".join(f"{counts[(s, c)]:15d}" for c in classes))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
