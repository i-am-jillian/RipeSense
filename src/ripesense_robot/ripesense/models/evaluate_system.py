"""System-level evaluation: held-out photos → the same featurize + Predictor
path a real scan uses → score the *final verdict*, not each model alone.

  python -m ripesense.models.evaluate_system [--fruit-type auto|given]

Why: per-model test accuracy hid a real failure — the fruit-type model called
green bananas "guava", so the banana stage model never ran. Only an end-to-end
check catches that kind of hand-off error.

Scoring (camera-only, pretrained tier):
* banana stage photos: unripe→unripe; ripe→ripe or ripe_to_overripe;
  overripe→overripe or ripe_to_overripe. "spoiled" on a non-rotten banana is wrong.
* fresh/rotten photos: rotten→spoiled; fresh→anything but spoiled.
* "uncertain" is reported separately (it is a safe answer, not a correct one).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from .. import config
from .featurize import row_features
from .image import Models, parse_class
from .infer import Predictor

OK_STAGE = {"unripe": {"unripe"}, "ripe": {"ripe", "ripe_to_overripe"},
            "overripe": {"overripe", "ripe_to_overripe"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="datasets/banana_stages/test")
    ap.add_argument("--quality", default="datasets/fruit_quality/test")
    ap.add_argument("--fruit-type", choices=["auto", "given"], default="auto",
                    help="auto = image model decides the fruit; given = the user picked it in the UI")
    ap.add_argument("--max-per-class", type=int, default=60)
    a = ap.parse_args()
    cfg = config.load()
    models = Models(config.models_dir())
    pred = Predictor(cfg)
    pred.art = None  # evaluate the camera-only pretrained tier
    out = defaultdict(Counter)
    fruit_hits = Counter()
    errors = []

    def run(path, given):
        bgr = cv2.imread(str(path))
        row = {"fruit_type": given if a.fruit_type == "given" else ""}
        feats, aux = row_features(row, cfg, models, frame=(bgr, None, 615.0, 615.0))
        ft = row["fruit_type"] or (aux["image"]["fruit_type"] if aux.get("image") and aux["image"]["fruit_conf"] >= 0.4 else "unknown")
        return pred.predict(feats, ft, aux), ft

    for d in sorted(Path(a.stages).iterdir()):
        for f in sorted(d.glob("*.jpg"))[: a.max_per_class]:
            r, ft = run(f, "banana")
            fruit_hits["banana_stage", ft == "banana"] += 1
            key = "uncertain" if r["stage"] == "uncertain" else ("ok" if r["stage"] in OK_STAGE[d.name] else "wrong")
            out[f"banana {d.name}"][key] += 1
            if key == "wrong":
                errors.append((str(f), d.name, ft, r["stage"]))
    for d in sorted(Path(a.quality).iterdir()):
        fruit, rot = parse_class(d.name)
        for f in sorted(d.glob("*.jpg"))[: a.max_per_class]:
            r, ft = run(f, fruit)
            fruit_hits["quality", ft == fruit] += 1
            if r["stage"] == "uncertain":
                key = "uncertain"
            else:
                key = "ok" if (r["stage"] == "spoiled") == bool(rot) else "wrong"
            out["rotten" if rot else "fresh"][key] += 1
            out[f"  {d.name}"][key] += 1
            if key == "wrong":
                errors.append((str(f), d.name, ft, r["stage"]))

    print(f"fruit type: {a.fruit_type}")
    summary = {}
    for k in sorted(out, key=lambda s: (s.startswith("  "), s)):
        c = out[k]
        n = sum(c.values())
        summary[k] = {"n": n, "correct": c["ok"], "wrong": c["wrong"], "uncertain": c["uncertain"]}
        print(f"{k:24s} n={n:4d}  correct {c['ok'] / n:6.1%}  wrong {c['wrong'] / n:6.1%}  uncertain {c['uncertain'] / n:6.1%}")
    for g in ("banana_stage", "quality"):
        n = fruit_hits[g, True] + fruit_hits[g, False]
        if n and a.fruit_type == "auto":
            print(f"fruit type recognised ({g}): {fruit_hits[g, True] / n:.1%}")
    tot = Counter()
    for k, c in out.items():
        if not k.startswith("  "):   # per-class rows are already inside fresh/rotten
            tot.update(c)
    n = sum(tot.values())
    print(f"OVERALL n={n}  correct {tot['ok'] / n:.1%}  wrong {tot['wrong'] / n:.1%}  uncertain {tot['uncertain'] / n:.1%}")
    p = config.models_dir() / f"system_eval_{a.fruit_type}.json"
    p.write_text(json.dumps({"summary": summary, "overall": dict(tot), "errors": errors[:200]}, indent=1))
    print(f"→ {p} ({len(errors)} errors listed)")


if __name__ == "__main__":
    main()
