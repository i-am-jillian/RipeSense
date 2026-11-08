"""Fine-tune MobileNetV3-Small for fruit type + fresh/rotten, export ONNX for the Pi.

Run on a laptop or Google Colab (GPU helps, CPU works for a few epochs):

    pip install torch torchvision onnx
    python tools/train_image_model.py --data /path/to/images --out models/fruit_classifier.onnx

--data is a folder of class folders, e.g. the Mendeley "Fresh and Rotten Fruits" or FruitNet
datasets: freshbanana/, rottenbanana/, Banana_Good/, Banana_Bad/ ... Folder names are mapped to
labels like "banana__fresh" / "banana__rotten"; unknown fruits become "other__...".
Copy both output files (.onnx and .onnx.labels.json) to models/ on the Pi.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

FRUITS = ["banana", "avocado", "tomato", "apple", "mango", "other"]
ALIASES = {"bananas": "banana", "apples": "apple", "tomatoes": "tomato", "mangoes": "mango", "mangos": "mango",
           "avocados": "avocado"}


def normalise(folder: str) -> str:
    name = folder.lower()
    cond = "rotten" if re.search(r"rotten|bad|spoil|stale", name) else "fresh"
    fruit_part = re.sub(r"fresh|rotten|good|bad|spoiled|stale|quality|mixed|[^a-z]", "", name)
    fruit = ALIASES.get(fruit_part, fruit_part)
    return f"{fruit if fruit in FRUITS else 'other'}__{cond}"


def main():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, models, transforms

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/fruit_classifier.onnx")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-pretrained", action="store_true", help="skip downloading ImageNet weights")
    a = ap.parse_args()

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(),
                                   transforms.ColorJitter(0.25, 0.25, 0.1, 0.02), transforms.ToTensor(), norm])
    test_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])

    base = datasets.ImageFolder(a.data)
    labels = sorted({normalise(c) for c in base.classes})
    remap = {i: labels.index(normalise(c)) for i, c in enumerate(base.classes)}
    print("classes:", {c: normalise(c) for c in base.classes})

    class Mapped(torch.utils.data.Dataset):
        def __init__(self, tf):
            self.ds = datasets.ImageFolder(a.data, transform=tf)

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            x, y = self.ds[i]
            return x, remap[y]

    idx = list(range(len(base)))
    random.Random(0).shuffle(idx)
    cut = int(0.85 * len(idx))
    train_ds, val_ds = Subset(Mapped(train_tf), idx[:cut]), Subset(Mapped(test_tf), idx[cut:])
    tl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=2)
    vl = DataLoader(val_ds, batch_size=a.batch, num_workers=2)

    weights = None if a.no_pretrained else models.MobileNet_V3_Small_Weights.DEFAULT
    net = models.mobilenet_v3_small(weights=weights)
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(labels))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for ep in range(a.epochs):
        net.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss_fn(net(x), y).backward()
            opt.step()
        sched.step()
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                p = net(x.to(dev)).argmax(1).cpu()
                correct += int((p == y).sum()); total += len(y)
        print(f"epoch {ep + 1}/{a.epochs}  val accuracy {correct / max(total, 1):.3f}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.cpu().eval()
    torch.onnx.export(net, torch.randn(1, 3, 224, 224), str(out), input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    Path(str(out) + ".labels.json").write_text(json.dumps(labels))
    print("saved", out, "and", str(out) + ".labels.json")


if __name__ == "__main__":
    main()
