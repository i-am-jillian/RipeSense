# Development validation — 2026-11-07

## Executed

- Validated notebook JSON with `nbformat`; parsed every Python cell with `ast` (the `%pip` cell was excluded from plain-Python parsing).
- Installed Ultralytics 8.3.235 in an isolated environment and executed the notebook's setup/detection/crop code locally on Apple M3 CPU, Python 3.13.3, PyTorch 2.15.0. Local `/tmp` paths replaced Colab paths and the Drive mount was omitted.
- Downloaded official YOLO11n weights and COCO val2017 image **189078**. At confidence 0.25 and image size 640, the baseline returned **7 detections: 2 banana, 3 apple, 2 orange**. Exported seven PNG crops, the CSV and numbered-box image. Inspected the rendered image. Detection counts are a smoke-test observation, not an accuracy measurement.
- Downloaded the official COCO annotation ZIP through `https://s3.amazonaws.com/images.cocodataset.org/`. The `https://images.cocodataset.org/` hostname failed TLS hostname validation here; notebook requests use the working S3 endpoint without disabling TLS checks.
- Executed the actual subset builder with caps reduced to **one image per class per split**: 3 train images / 10 boxes and 3 validation images / 24 boxes. All selected target annotations were retained; image IDs and file hashes were disjoint between splits. Checked bounding-box normalization including clipping.
- Ran baseline validation and **one training epoch** with the generated 80-class YAML and fruit class filter, `imgsz=160`, `batch=2`, CPU. Training produced `best.pt`; automatic checkpoint validation completed. Verified baseline checkpoint bytes were unchanged. This deliberately tiny test is not the default 480-image, five-epoch experiment and provides no meaningful quality claim.
- Queried Mendeley v2 using the public API's vendor Accept header and explicit pagination. The simpler unpaginated URL in Sujal's notebook returned 403 here. Retrieved the ripeness ZIP central directory via HTTP range requests: **3,699 members, 3,231 JPEG files, no non-image metadata files**.
- Parsed all JPEG paths (including `day 3/red delicious new zealand/APPLE 4/apple4-9.jpg`, whose suffix separator differs). Found **20 candidate variety/apple-number identities** across 18 days. Day 9 has 171 JPEGs; the other days have 180 each. These are candidate identifiers, not verified per-stage physical-apple counts.
- Downloaded the nine `day 1/kinnaur/APPLE 1` JPEGs by range reads. All are RGB **17,280 × 1,080** strips. Inspected a strip as nine equal-width panels: one dark-looking panel and eight differently illuminated panels. No wavelength, black-reference identity, stage or persistent specimen mapping was established from this appearance.
- Executed the notebook's actual inventory parser and default spectral gate. It reported the missing mappings and did not fit a classifier or produce spectral accuracy.

The small source inventory is recorded in `mendeley_ripeness_audit.json`; no images, ZIPs, model binaries or local test metrics are stored in Git.

## Not executed or established

- Google Colab runtime, interactive Drive authorization, GPU training, full default COCO fine-tuning, or the final full-run detector comparison.
- Full Mendeley archive download/hash verification fallback (range inventory and nine member downloads were used instead). The API-supplied SHA-256 is recorded as publisher metadata, not claimed as locally verified for the full archive.
- Conditional spectral training with real author-verified metadata/masks: these are absent. No spectral classifier or spectral performance claim is delivered.
- Sujal's ONNX inference, phone/gantry acquisition or live AS7341 integration; trained RGB artifacts and hardware are not available.

## Interpretation

Mendeley v2 documents under-ripe, ripe and over-ripe stages, but the inspected ripeness ZIP organizes files by day and candidate apple ID. Each file is already a concatenation; its numbered suffix is not a wavelength. Panel pairing is visually suggested, while wavelength ordering, stage assignment and physical identity continuity are unverified. Counts of identifiable physical apples **per class are unknown**, and `spoiled` is not a supplied class. The notebook therefore stops spectral training before reporting accuracy and describes the evidence required to enable its conditional training path.

Source: Gaikwad, Shilpa; Kothari, Sonali (2025), [Multispectral images of apples for ripeness, sweetness and variety grading, v2](https://doi.org/10.17632/y5h6v8w6ms.2), CC BY 4.0. Also see [Ultralytics detection documentation](https://docs.ultralytics.com/tasks/detect/) and [COCO](https://cocodataset.org/#download).
