"""'Why' heatmap: occlusion sensitivity on the crop the model actually saw.

RISE: show the network ~100 copies of the fruit with random halves greyed out
and see which regions keep the verdict's class probability high. Those regions
are where the evidence is (brown spots, mould, green skin). Model-agnostic, no
gradients needed, ~0.5 s on a laptop, a few seconds on a Pi 5.
"""
from __future__ import annotations

import cv2
import numpy as np

from .image import OnnxClassifier

GREY = (124, 116, 104)  # ImageNet mean in BGR: 'nothing here' to the network
N_MASKS, CELLS = 96, 7


def _masks(n: int, seed: int = 0) -> np.ndarray:
    """RISE masks: random 7x7 on/off grids, upsampled smoothly with a random shift."""
    rng = np.random.default_rng(seed)
    cell = int(np.ceil(224 / CELLS))
    up = (CELLS + 1) * cell
    out = np.empty((n, 224, 224), np.float32)
    for i in range(n):
        g = (rng.random((CELLS + 1, CELLS + 1)) < 0.5).astype(np.float32)
        m = cv2.resize(g, (up, up), interpolation=cv2.INTER_LINEAR)
        dx, dy = rng.integers(0, cell, 2)
        out[i] = m[dy:dy + 224, dx:dx + 224]
    return out


_MASKS = None


def occlusion(clf: OnnxClassifier, bgr: np.ndarray, mask: np.ndarray | None, target: int | None = None,
              batch: int = 32):
    """RISE saliency (Petsiuk et al. 2018): show the net many randomly half-hidden
    copies; pixels that are visible whenever the class probability is high are the
    evidence. Works even when evidence is spread out (spots all over a banana),
    where single-patch occlusion shows nothing.
    Returns (heat 224x224 0-1, view 224x224 BGR, target index, base prob)."""
    global _MASKS
    if _MASKS is None:
        _MASKS = _masks(N_MASKS)
    view = clf.prepare(bgr, mask)
    p0 = clf.run([view])[0]
    k = int(np.argmax(p0)) if target is None else int(target)
    grey = np.array(GREY, np.float32)
    scores = []
    for i in range(0, N_MASKS, batch):
        ms = _MASKS[i:i + batch, ..., None]
        views = (view[None].astype(np.float32) * ms + grey * (1 - ms)).astype(np.uint8)
        scores.append(clf.run(list(views))[:, k])
    s = np.concatenate(scores)
    sal = np.tensordot(s - s.mean(), _MASKS, axes=1) / (_MASKS.sum(0) + 1e-6)
    sal = np.clip(sal, 0, None)
    m = sal.max()
    heat = sal / m if m > 1e-9 else sal
    return heat.astype(np.float32), view, k, float(p0[k]), float(s.std())


def overlay(view: np.ndarray, heat: np.ndarray, scale: int = 2) -> np.ndarray:
    """Colour only where the evidence is (alpha grows with heat)."""
    h = np.clip((heat - 0.35) / 0.65, 0, 1)
    alpha = 0.65 * h ** 1.2
    col = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = (view * (1 - alpha[..., None]) + col * alpha[..., None]).astype(np.uint8)
    return cv2.resize(out, (224 * scale, 224 * scale), interpolation=cv2.INTER_CUBIC)


def explain(models, bgr, mask, fruit_type: str, result_stage: str) -> dict | None:
    """Pick the model behind the verdict and explain its top class."""
    if bgr is None or models is None:
        return None
    st, im = models.stage, models.image
    if st is not None and fruit_type in ("banana",) and result_stage not in ("spoiled",):
        clf, name = st, "stage model"
    elif im is not None:
        clf, name = im, "fruit/rot model"
    else:
        return None
    heat, view, k, p, spread = occlusion(clf, bgr, mask)
    # Where on the fruit (rough): centroid of the hot area, as words for the "why" text.
    hot = heat >= 0.6
    where = ""
    if hot.any():
        ys, xs = np.nonzero(hot)
        cy, cx = ys.mean() / 224, xs.mean() / 224
        where = ("top" if cy < 0.33 else "bottom" if cy > 0.67 else "middle") + "-" + \
                ("left" if cx < 0.33 else "right" if cx > 0.67 else "centre")
    return {"model": name, "class": clf.classes[k], "prob": p, "sensitivity": spread,
            "where": where, "image": overlay(view, heat), "hot_frac": float(hot.mean())}
