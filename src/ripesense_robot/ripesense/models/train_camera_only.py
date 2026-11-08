"""Train a real banana image classifier directly from the downloaded archive.

CPU training: OMP_NUM_THREADS=4 .venv/bin/python -m
ripesense.models.train_camera_only --epochs 8
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
import zipfile

import cv2
import numpy as np

SOURCE = "https://data.mendeley.com/datasets/56c75hvm7r/2"
CLASSES = ["overripe", "ripe", "unripe"]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(archive, root):
    """Retain source splits, keeping all boxes from a source image together."""
    root = Path(root)
    if any(root.glob("*/*/*.jpg")):
        raise ValueError(f"Use an empty output dataset directory: {root}; choose a new --data directory")
    manifest, seen, ids = [], {}, {}
    votes = {}
    with zipfile.ZipFile(archive) as z:
        images = {Path(n).stem: n for n in z.namelist() if n.lower().endswith(".jpg")}
        labels = sorted(n for n in z.namelist() if "/labels/" in n and n.endswith(".txt"))
        # This archive has no data.yaml. Check the filename-stage mapping
        # against every annotation instead of silently assuming class IDs.
        for name in labels:
            stage = Path(name).stem.split("_")[0]
            if stage not in CLASSES:
                raise ValueError(f"Unknown class in {name}")
            for line in z.read(name).decode().splitlines():
                if line.strip():
                    cid = int(line.split()[0])
                    votes.setdefault(cid, Counter())[stage] += 1
        for cid, counts in votes.items():
            stage, n = counts.most_common(1)[0]
            if n / sum(counts.values()) < .95:
                raise ValueError("Cannot establish annotation class mapping")
            ids[cid] = stage
        print("Annotation ID mapping:", ids, "filename votes:", votes, flush=True)
        # Reserve test/val duplicates first; never copy the same photograph
        # into training. This does not establish fruit-identity independence.
        labels.sort(key=lambda n: {"test": 0, "val": 1, "train": 2}[Path(n).parent.name])
        for name in labels:
            stem, split = Path(name).stem, Path(name).parent.name
            raw = z.read(images[stem])
            sha = hashlib.sha256(raw).hexdigest()
            if sha in seen:
                continue
            seen[sha] = split
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            h, w = image.shape[:2]
            for k, line in enumerate(z.read(name).decode().splitlines()):
                parts = line.split()
                if len(parts) != 5:
                    continue
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                bw, bh = bw * 1.24, bh * 1.24
                x0, x1 = max(0, int((cx-bw/2)*w)), min(w, int((cx+bw/2)*w))
                y0, y1 = max(0, int((cy-bh/2)*h)), min(h, int((cy+bh/2)*h))
                if min(x1-x0, y1-y0) < 40:
                    continue
                crop = image[y0:y1, x0:x1]
                scale = min(1, 384 / max(crop.shape[:2]))
                crop = cv2.resize(crop, (round(crop.shape[1]*scale), round(crop.shape[0]*scale)))
                dest = root / split / ids[cid] / f"{stem}_{k}.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest), crop)
                manifest.append({"path": str(dest), "source": images[stem], "source_sha256": sha,
                                 "split": split, "stage": ids[cid]})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Crops:", dict(Counter((m["split"], m["stage"]) for m in manifest)), flush=True)
    return manifest


def main():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="datasets/raw/BANANA_DATASET75.zip")
    ap.add_argument("--data", default="datasets/camera_banana")
    ap.add_argument("--out", default="models/camera_only")
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = prepare(args.archive, root)
    norm = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=.25, contrast=.2, saturation=.15, hue=.015),
        transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    ds = {s: datasets.ImageFolder(root/s, transform=train_tf if s == "train" else eval_tf)
          for s in ("train", "val", "test")}
    assert all(d.classes == CLASSES for d in ds.values())
    loaders = {s: DataLoader(d, batch_size=16, shuffle=s == "train", num_workers=0)
               for s, d in ds.items()}
    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(CLASSES))
    # Freeze backbone first, then fine-tune all layers at a lower rate.
    for p in net.features.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=.001)
    counts = np.bincount(ds["train"].targets, minlength=3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(ds["train"])/(3*counts), dtype=torch.float32))

    def evaluate(split):
        net.eval()
        actual, predicted = [], []
        with torch.inference_mode():
            for x, y in loaders[split]:
                actual.extend(y.tolist())
                predicted.extend(net(x).argmax(1).tolist())
        return {"n": len(actual), "accuracy": accuracy_score(actual, predicted),
                "macro_f1": f1_score(actual, predicted, labels=[0,1,2], average="macro", zero_division=0),
                "confusion": confusion_matrix(actual, predicted, labels=[0,1,2]).tolist()}

    best, history = -1, []
    for epoch in range(args.epochs):
        t0 = time.monotonic()
        if epoch == 2:
            for p in net.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(net.parameters(), lr=.0001, weight_decay=.0001)
        net.train()
        if epoch < 2:
            net.features.eval()  # retain pretrained batchnorm statistics
        losses = []
        for x, y in loaders["train"]:
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val = evaluate("val")
        record = {"epoch": epoch+1, "loss": float(np.mean(losses)), "val": val,
                  "seconds": round(time.monotonic()-t0, 1)}
        history.append(record)
        print(json.dumps(record), flush=True)
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            torch.save(net.state_dict(), out/"best.pt")
    net.load_state_dict(torch.load(out/"best.pt", weights_only=True))
    net.eval()
    metrics = {"classes": CLASSES, "val": evaluate("val"), "test": evaluate("test"),
               "history": history, "source": SOURCE, "license": "CC BY 4.0",
               "scope": "Banana crops only; no spoilage or remaining-shelf-life labels",
               "split_limit": "Source image split; fruit identities unavailable; nearby photos may be related",
               "crop_counts": {s: len(d) for s,d in ds.items()}, "seed": 42,
               "archive_sha256": file_sha256(args.archive),
               "unique_source_photos": len({m["source_sha256"] for m in manifest})}
    torch.onnx.export(net, torch.zeros(1,3,224,224), str(out/"stage.onnx"),
                      input_names=["image"], output_names=["logits"], opset_version=17, dynamo=False)
    (out/"classes.json").write_text(json.dumps(CLASSES))
    (out/"metrics.json").write_text(json.dumps(metrics, indent=2))
    print("FINAL", json.dumps(metrics["test"]), flush=True)


if __name__ == "__main__":
    main()
