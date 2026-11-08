"""Evaluate the exported camera model using the same decoder as the web API."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ..camera_only import CameraPredictor, ROOT, decode_photo


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    model_dir = ROOT / "models/camera_only"
    model = CameraPredictor(model_dir)
    rows = []
    for photo in sorted((ROOT / "datasets/camera_banana/test").glob("*/*.jpg")):
        result = model.predict(decode_photo(photo.read_bytes()))
        rows.append({"photo": str(photo.relative_to(ROOT)), "actual": photo.parent.name,
                     "predicted": result["predicted_stage"], "displayed": result["stage"],
                     "score": result["score"], "inference_ms": result["inference_ms"]})
    actual, predicted = [r["actual"] for r in rows], [r["predicted"] for r in rows]
    assert rows, "No held-out test photos"
    metrics = {"n": len(rows), "accuracy": accuracy_score(actual, predicted),
               "macro_f1": f1_score(actual, predicted, labels=model.classes, average="macro"),
               "classes": model.classes,
               "confusion": confusion_matrix(actual, predicted, labels=model.classes).tolist(),
               "uncertain": sum(r["displayed"] == "uncertain" for r in rows),
               "median_inference_ms": statistics.median(r["inference_ms"] for r in rows),
               "model_sha256": digest_file(model_dir / "stage.onnx")}
    with (model_dir / "test_predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    artifact = model.metrics
    artifact["onnx_test"] = metrics
    archive = ROOT / "datasets/raw/BANANA_DATASET75.zip"
    if archive.exists():
        artifact["archive_sha256"] = digest_file(archive)
    else:
        artifact["archive_note"] = "Archive unavailable after training; per-photo SHA256 values remain in datasets/camera_banana/manifest.json"
    (model_dir / "metrics.json").write_text(json.dumps(artifact, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
