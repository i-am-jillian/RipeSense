"""Teach a new fruit (or a new stage) in ~30 seconds, with no retraining.

Each scan's fruit crop is embedded by the image model's backbone (576-d pooled
features, see image.OnnxClassifier.embed). A taught label is the mean of its
example embeddings (a prototype). At scan time the fruit is matched against
the taught prototypes *and* the built-in class prototypes (computed once from
the training photos, models/artifacts/image_model_prototypes.npz): a taught
label wins only if it is the nearest prototype and close enough.

Labels are "kiwi" (new fruit type) or "kiwi:ripe" (fruit + stage).

  python -m ripesense.models.fewshot --build-bank datasets/fruit_quality/train
  python -m ripesense.models.fewshot --selftest     # hold out a class, teach it, measure
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
MIN_SIM = 0.50          # cosine floor; tuned with --selftest (5 shots: recall 0.8-1.0, 0 false accepts)
MARGIN = 0.0            # taught prototype must beat the best built-in one by this


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


class FewShot:
    def __init__(self, store_path: Path, bank_path: Path | None = None):
        self.path = Path(store_path)
        self.lock = threading.Lock()
        self.examples: dict[str, list[list[float]]] = {}
        if self.path.exists():
            try:
                self.examples = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                log.warning("few-shot store %s unreadable; starting empty", self.path)
        self.bank_names: list[str] = []
        self.bank = np.zeros((0, 576), np.float32)
        if bank_path and Path(bank_path).exists():
            z = np.load(bank_path)
            self.bank_names, self.bank = list(z["names"]), _unit(z["protos"])

    # ---------------------------------------------------------------- teaching
    def add(self, label: str, emb: np.ndarray) -> int:
        label = label.strip().lower()
        with self.lock:
            self.examples.setdefault(label, []).append([round(float(x), 5) for x in _unit(emb)])
            self._save()
            return len(self.examples[label])

    def forget(self, label: str | None = None):
        with self.lock:
            if label is None:
                self.examples = {}
            else:
                self.examples.pop(label.strip().lower(), None)
            self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.examples))

    def summary(self) -> dict:
        return {k: len(v) for k, v in self.examples.items()}

    # --------------------------------------------------------------- matching
    def match(self, emb: np.ndarray | None) -> dict | None:
        """→ {label, fruit, stage, sim, best_builtin, builtin_sim, accepted} or None."""
        if emb is None or not self.examples:
            return None
        e = _unit(emb)
        with self.lock:
            labels = list(self.examples)
            protos = _unit(np.array([np.mean(self.examples[k], axis=0) for k in labels]))
        sims = protos @ e
        i = int(np.argmax(sims))
        bsim, bname = -1.0, ""
        if len(self.bank):
            bs = self.bank @ e
            j = int(np.argmax(bs))
            bsim, bname = float(bs[j]), self.bank_names[j]
        s = float(sims[i])
        label = labels[i]
        fruit, _, stage = label.partition(":")
        return {"label": label, "fruit": fruit, "stage": stage or None, "sim": s,
                "n_examples": len(self.examples[label]),
                "best_builtin": bname, "builtin_sim": bsim,
                "accepted": s >= MIN_SIM and s >= bsim + MARGIN}


# ------------------------------------------------------------------ tooling
def _embed_dir(model, d: Path, n: int, offset: int = 0):
    import cv2
    out = []
    for f in sorted(d.glob("*.jpg"))[offset:offset + n]:
        e = model.embed(cv2.imread(str(f)))
        if e is not None:
            out.append(e)
    return np.array(out, np.float32)


def build_bank(train_root: Path, out: Path, per_class: int = 40):
    from .. import config
    from .image import ImageModel
    m = ImageModel.load(config.models_dir())
    names, protos = [], []
    for d in sorted(p for p in Path(train_root).iterdir() if p.is_dir()):
        E = _embed_dir(m, d, per_class)
        if len(E):
            names.append(d.name)
            protos.append(_unit(E).mean(0))
            print(f"{d.name:22s} {len(E)} images")
    # Banana stage photos too, so a green banana is never mistaken for a taught fruit.
    bs = config.path("datasets/banana_stages/train")
    if bs.exists():
        for d in sorted(p for p in bs.iterdir() if p.is_dir()):
            E = _embed_dir(m, d, per_class)
            if len(E):
                names.append(f"banana_{d.name}")
                protos.append(_unit(E).mean(0))
    np.savez(out, names=np.array(names), protos=np.array(protos, np.float32))
    print(f"→ {out} ({len(names)} prototypes)")


def selftest(holdout: str = "strawberry", shots: int = 5, n_test: int = 40):
    """Pretend one fruit is unknown: drop it from the bank, teach it from `shots`
    photos, then measure recall on unseen photos of it and false accepts on others."""
    import tempfile

    from .. import config
    from .image import ImageModel
    m = ImageModel.load(config.models_dir())
    train, test = config.path("datasets/fruit_quality/train"), config.path("datasets/fruit_quality/test")
    bank = config.models_dir() / "image_model_prototypes.npz"
    z = np.load(bank)
    keep = [i for i, n in enumerate(z["names"]) if holdout not in n]
    with tempfile.TemporaryDirectory() as td:
        nb = Path(td) / "bank.npz"
        np.savez(nb, names=z["names"][keep], protos=z["protos"][keep])
        fs = FewShot(Path(td) / "fs.json", nb)
        for e in _embed_dir(m, train / f"fresh_{holdout}", shots):
            fs.add(holdout, e)
        hit = sum(fs.match(e)["accepted"] for e in _embed_dir(m, test / f"fresh_{holdout}", n_test))
        n_pos = min(n_test, len(list((test / f"fresh_{holdout}").glob("*.jpg"))))
        fa = n_neg = 0
        for d in sorted(test.iterdir()):
            if holdout in d.name:
                continue
            E = _embed_dir(m, d, 15)
            fa += sum(fs.match(e)["accepted"] for e in E)
            n_neg += len(E)
    r = {"holdout": holdout, "shots": shots, "recall": hit / max(1, n_pos), "false_accept": fa / max(1, n_neg),
         "n_pos": n_pos, "n_neg": n_neg}
    print(json.dumps(r))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-bank", metavar="TRAIN_DIR")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--holdout", default="strawberry")
    ap.add_argument("--shots", type=int, default=5)
    a = ap.parse_args()
    from .. import config
    if a.build_bank:
        build_bank(config.path(a.build_bank), config.models_dir() / "image_model_prototypes.npz")
    if a.selftest:
        selftest(a.holdout, a.shots)


if __name__ == "__main__":
    main()
