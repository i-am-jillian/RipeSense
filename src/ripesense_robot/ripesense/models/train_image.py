"""Fine-tune MobileNetV3-Small on an ImageFolder dataset and export ONNX.

Two models in RipeSense use this (doc p.12, "Image model"):

  # fruit type + fresh/rotten  (Fresh and Rotten Fruits + FruitNet crops)
  python -m ripesense.models.train_image --data datasets/fruit_quality --name image_model
  # ripeness stage (Annotated Banana Ripeness Stages crops, later your own cabinet photos)
  python -m ripesense.models.train_image --data datasets/banana_stages --name stage_model

--data holds train/ val/ test/ ImageFolders (from the prep_* scripts). Class
folder names: fruit-quality classes look like "fresh_apple"/"rotten_apple";
stage classes are "unripe"/"ripe"/"overripe"[/"spoiled"].

Writes models/artifacts/<name>.onnx, <name>_classes.json, <name>_metrics.json.
CPU is fine for a few thousand images; Colab/GPU is faster.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


class BackgroundSwap:
    """Replace a light, unsaturated photo background (white paper/table) with a
    random dark or mid colour, so a model trained on studio photos also works
    in the black-acrylic chamber or in front of a webcam."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        import numpy as np
        from PIL import Image
        if random.random() > self.p:
            return img
        a = np.asarray(img.convert("RGB")).astype(np.int16)
        mx, mn = a.max(2), a.min(2)
        sat = (mx - mn) / np.maximum(mx, 1)
        bg = (sat < 0.18) & (mx > 140)
        if bg.mean() < 0.08:
            return img
        v = random.randint(8, 115)        # neutral grey/dark only: a green or yellow
        tint = random.randint(-6, 6)      # background would mimic the fruit's own colour
        col = np.array((v + tint, v, v - tint), np.int16)
        noise = np.random.randint(-6, 7, a.shape, dtype=np.int16)
        a = np.where(bg[..., None], np.clip(col + noise, 0, 255), a)
        return Image.fromarray(a.astype(np.uint8))


def main():
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from torchvision import datasets, models, transforms

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/artifacts")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="image_model", help="image_model | stage_model")
    ap.add_argument("--max-per-class", type=int, default=0, help="cap train images per class (speed)")
    ap.add_argument("--bg-swap", type=float, default=0.0,
                    help="probability of swapping a white photo background for a dark one (studio datasets)")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(a.data)

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Strong colour/lighting augmentation: the public photos and a USB camera
    # in a room (or the cabinet's LEDs) will not share white balance.
    tf_train = transforms.Compose([
        BackgroundSwap(a.bg_swap),
        transforms.RandomResizedCrop(224, scale=(0.55, 1.0), ratio=(0.6, 1.6)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.25, hue=0.025),
        transforms.RandomGrayscale(p=0.02),
        transforms.ToTensor(), norm,
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
    ])
    tf_eval = transforms.Compose([transforms.Resize(240), transforms.CenterCrop(224), transforms.ToTensor(), norm])

    tr = datasets.ImageFolder(root / "train", transform=tf_train)
    classes = tr.classes
    if a.max_per_class:
        keep, seen = [], {}
        for i in torch.randperm(len(tr.samples)).tolist():
            c = tr.samples[i][1]
            if seen.get(c, 0) < a.max_per_class:
                seen[c] = seen.get(c, 0) + 1
                keep.append(tr.samples[i])
        tr.samples = tr.imgs = keep
        tr.targets = [s[1] for s in keep]
    va = datasets.ImageFolder(root / "val", transform=tf_eval) if (root / "val").exists() else None
    te = datasets.ImageFolder(root / "test", transform=tf_eval) if (root / "test").exists() else None
    for d in (va, te):
        assert d is None or d.classes == classes, f"class folders differ: {d.classes} vs {classes}"
    counts = np.bincount(tr.targets, minlength=len(classes))
    print(f"classes {classes}  train {dict(zip(classes, counts.tolist()))}  "
          f"val {len(va) if va else 0}  test {len(te) if te else 0}  device {dev}")

    # Balance classes by sampling.
    w = (1.0 / counts)[tr.targets]
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(tr), replacement=True)
    tl = DataLoader(tr, batch_size=a.batch, sampler=sampler, num_workers=a.workers)

    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(classes))
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tl))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    def evaluate(ds):
        net.eval()
        cm = np.zeros((len(classes), len(classes)), int)
        with torch.no_grad():
            for x, y in DataLoader(ds, batch_size=64, num_workers=a.workers):
                p = net(x.to(dev)).argmax(1).cpu().numpy()
                for t_, p_ in zip(y.numpy(), p):
                    cm[t_, p_] += 1
        acc = np.trace(cm) / max(1, cm.sum())
        rec = cm.diagonal() / np.maximum(1, cm.sum(1))
        prec = cm.diagonal() / np.maximum(1, cm.sum(0))
        f1 = np.where(rec + prec > 0, 2 * rec * prec / np.maximum(1e-9, rec + prec), 0)
        return {"accuracy": float(acc), "macro_f1": float(f1.mean()), "confusion": cm.tolist()}

    best, best_state = -1.0, None
    for ep in range(a.epochs):
        net.train()
        t0 = time.time()
        tot = n = 0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(net(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        m = evaluate(va) if va else {"macro_f1": 0.0, "accuracy": 0.0}
        print(f"epoch {ep + 1:2d}/{a.epochs}  loss {tot / n:.3f}  val acc {m['accuracy']:.3f}  "
              f"val macro-F1 {m['macro_f1']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if m["macro_f1"] >= best:
            best, best_state = m["macro_f1"], {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    net.load_state_dict(best_state)
    net.to(dev)
    metrics = {"classes": classes, "val": evaluate(va) if va else None, "test": evaluate(te) if te else None,
               "train_counts": dict(zip(classes, counts.tolist())), "epochs": a.epochs,
               "source": str(root)}
    if metrics["test"]:
        t = metrics["test"]
        print(f"\nTEST  accuracy {t['accuracy']:.3f}  macro-F1 {t['macro_f1']:.3f}")
        print("confusion (rows = true, cols = predicted):", classes)
        for c, row in zip(classes, t["confusion"]):
            print(f"  {c:10s} {row}")

    net.cpu().eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, torch.randn(1, 3, 224, 224), str(out / f"{a.name}.onnx"),
                      input_names=["input"], output_names=["logits"], opset_version=17,
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    (out / f"{a.name}_classes.json").write_text(json.dumps(classes))
    (out / f"{a.name}_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"→ {out}/{a.name}.onnx")


if __name__ == "__main__":
    main()
