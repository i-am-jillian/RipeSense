"""ONNX image models trained by models/train_image.py (doc p.12, "Image model").

* image_model  — fruit type + fresh/rotten (Fresh and Rotten Fruits + FruitNet).
* stage_model  — ripeness stage from the photo (Annotated Banana Ripeness
                 Stages; retrain with your own cabinet photos per fruit).

Both are optional; each loads only if its .onnx + _classes.json exist.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_ROT = ("rotten", "bad", "spoiled", "stale")
_FRESH = ("fresh", "good")
STAGE_ORDER = {"unripe": 0, "ripe": 1, "overripe": 2, "spoiled": 3}


def parse_class(name: str) -> tuple[str, bool | None]:
    """'rotten_banana' / 'Fresh Banana' / 'freshbanana' / 'Banana_Bad' → ('banana', True)."""
    s = re.sub(r"[^a-z]", "", name.lower())
    rot = None
    for p in _ROT:
        if p in s:
            rot, s = True, s.replace(p, "")
    for p in _FRESH:
        if p in s:
            rot, s = (False if rot is None else rot), s.replace(p, "")
    if s.endswith("oes"):        # tomatoes → tomato
        s = s[:-2]
    elif s.endswith("s"):        # bananas → banana
        s = s[:-1]
    return s or "unknown", rot


def fruit_crop(bgr: np.ndarray, mask: np.ndarray | None, pad: float = 0.12) -> np.ndarray:
    """Crop to the fruit (the models were trained on padded per-fruit crops).
    Falls back to the whole frame if the mask is missing or implausible."""
    if mask is None or not mask.any():
        return bgr
    frac = (mask > 0).mean()
    if frac < 0.01 or frac > 0.92:
        return bgr
    ys, xs = np.nonzero(mask)
    H, W = mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    py, px = int((y1 - y0) * pad), int((x1 - x0) * pad)
    return bgr[max(0, y0 - py):min(H, y1 + py), max(0, x0 - px):min(W, x1 + px)]


class OnnxClassifier:
    """MobileNetV3 classifier from train_image.py. Also exposes the 576-d pooled
    embedding (input of the classifier head) for few-shot teaching, and batched
    inference for the occlusion "why" map."""

    def __init__(self, onnx_path: Path, classes_path: Path):
        import onnxruntime as ort
        model_bytes = Path(onnx_path).read_bytes()
        self.emb_name = None
        try:  # add the pooled features as a second graph output (no retraining needed)
            import onnx
            m = onnx.load_from_string(model_bytes)
            gemm = next(n for n in m.graph.node if n.op_type == "Gemm")
            self.emb_name = gemm.input[0]
            m.graph.output.append(onnx.helper.make_tensor_value_info(self.emb_name, onnx.TensorProto.FLOAT, None))
            model_bytes = m.SerializeToString()
        except Exception as e:  # noqa: BLE001
            log.info("embedding output unavailable (%s); few-shot teaching disabled", e)
            self.emb_name = None
        self.sess = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.out = self.sess.get_outputs()[0].name
        self.classes: list[str] = json.loads(Path(classes_path).read_text())
        self.path = onnx_path

    @classmethod
    def load(cls, models_dir: Path, name: str):
        p, c = models_dir / f"{name}.onnx", models_dir / f"{name}_classes.json"
        if p.exists() and c.exists():
            try:
                m = cls(p, c)
                log.info("loaded %s (%d classes)", name, len(m.classes))
                return m
            except Exception as e:  # noqa: BLE001
                log.warning("%s failed to load: %s", name, e)
        return None

    @staticmethod
    def prepare(bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Crop to the fruit, then the training eval transform (resize short side
        240, centre-crop 224). Returns the 224x224 BGR uint8 image the net sees."""
        img = fruit_crop(bgr, mask)
        h, w = img.shape[:2]
        s = 240 / min(h, w)
        img = cv2.resize(img, (max(224, round(w * s)), max(224, round(h * s))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        y0, x0 = (h - 224) // 2, (w - 224) // 2
        return img[y0:y0 + 224, x0:x0 + 224]

    @staticmethod
    def to_tensor(views: list[np.ndarray]) -> np.ndarray:
        x = np.stack([cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for v in views]).astype(np.float32) / 255.0
        return ((x - MEAN) / STD).transpose(0, 3, 1, 2).astype(np.float32)

    def run(self, views: list[np.ndarray], embed: bool = False):
        names = [self.out] + ([self.emb_name] if embed and self.emb_name else [])
        outs = self.sess.run(names, {self.inp: self.to_tensor(views)})
        logits = outs[0]
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        return (p, outs[1] if len(outs) > 1 else None) if embed else p

    def probs(self, bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        return self.run([self.prepare(bgr, mask)])[0]

    def embed(self, bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray | None:
        """L2-normalised 576-d embedding of the fruit crop (None if unavailable)."""
        if not self.emb_name:
            return None
        _, e = self.run([self.prepare(bgr, mask)], embed=True)
        e = e[0].astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-9)


class ImageModel(OnnxClassifier):
    """Fruit type + probability of rot."""

    def __init__(self, onnx_path, classes_path):
        super().__init__(onnx_path, classes_path)
        self.parsed = [parse_class(c) for c in self.classes]

    @classmethod
    def load(cls, models_dir: Path, name: str = "image_model"):
        return super().load(models_dir, name)

    def predict(self, bgr: np.ndarray, mask: np.ndarray | None = None, with_embedding: bool = False):
        """→ dict, or (dict, embedding) with with_embedding (one forward pass for both)."""
        p, emb = self.run([self.prepare(bgr, mask)], embed=True)
        p = p[0]
        if emb is not None:
            emb = emb[0].astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-9
        fruit_p: dict[str, float] = {}
        rot_p = rot_known = 0.0
        for (fruit, rot), pi in zip(self.parsed, p):
            fruit_p[fruit] = fruit_p.get(fruit, 0.0) + float(pi)
            if rot is not None:
                rot_known += pi
                rot_p += pi if rot else 0.0
        fruit = max(fruit_p, key=fruit_p.get)
        # Rot probability *given* the predicted fruit is less confounded by fruit type.
        fp = [(rot, pi) for (f, rot), pi in zip(self.parsed, p) if f == fruit and rot is not None]
        tot = sum(pi for _, pi in fp)
        rot_given = sum(pi for rot, pi in fp if rot) / tot if tot > 0 else (rot_p / rot_known if rot_known else float("nan"))
        top = sorted(fruit_p.items(), key=lambda kv: -kv[1])[:3]
        out = {"fruit_type": fruit, "fruit_conf": fruit_p[fruit], "rot_prob": float(rot_given),
               "fruit_top": [(f, round(v, 3)) for f, v in top]}
        return (out, emb) if with_embedding else out


class StageModel(OnnxClassifier):
    """Ripeness stage from the photo. Returns probabilities keyed by stage name."""

    @classmethod
    def load(cls, models_dir: Path, name: str = "stage_model"):
        return super().load(models_dir, name)

    def predict(self, bgr: np.ndarray, mask: np.ndarray | None = None) -> dict:
        p = self.probs(bgr, mask)
        probs = {c: float(pi) for c, pi in zip(self.classes, p)}
        score = sum(STAGE_ORDER.get(c, 1) * pi for c, pi in probs.items())  # 0 unripe … 2 overripe
        return {"probs": probs, "score": float(score), "fruits": ["banana"]}


class Models:
    """Everything optional that featurize/infer can use. Loaded once."""

    def __init__(self, models_dir: Path):
        self.image = ImageModel.load(models_dir)
        self.stage = StageModel.load(models_dir)
        from .spectral_prior import SpectralPrior
        self.spec_prior = SpectralPrior.load(models_dir)
        self.fewshot = None
        if self.image is not None and self.image.emb_name:
            from .. import config
            from .fewshot import FewShot
            self.fewshot = FewShot(config.data_dir() / "fewshot.json", models_dir / "image_model_prototypes.npz")

    def summary(self) -> dict:
        return {"image_model": self.image is not None, "stage_model": self.stage is not None,
                "spectral_prior": self.spec_prior is not None,
                "taught": self.fewshot.summary() if self.fewshot else {}}
