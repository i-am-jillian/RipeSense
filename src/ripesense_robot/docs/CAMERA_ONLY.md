# Camera-only RipeSense

For the expanded multi-fruit and description-based app, use
[CAMERA_GENERAL.md](CAMERA_GENERAL.md). This document records the earlier
banana-specific model and its evaluation.

This version uses a trained MobileNetV3-Small to classify a **banana** as
unripe, ripe, or overripe from an RGB photograph. It runs on the CPU, offline,
with no gas sensor, spectral sensor, Raspberry Pi GPIO, or cloud account.

Start from this directory:

```bash
.venv/bin/python -m ripesense.camera_only
```

Open **http://localhost:5001**. Choose **Start camera**, frame one banana so
it fills most of the photo, and press **Take photo & predict**. You can also
upload a photograph or choose **Use attached camera** to capture directly
from the computer running the app. After a scan, **Start camera** resumes the
live preview. **Save result** downloads the prediction as JSON.

Use even lighting and a plain background. This is a banana classifier;
other fruit, empty frames, faces, and unrelated objects are outside its
training scope and can still receive high scores. It does not detect fruit
automatically. Scores below 65% display an uncertain result; the scores have
not been calibrated as probabilities of correctness.

The model predicts appearance stages. It does not measure internal spoilage,
food safety, or days until spoilage. Any shelf-life text is clearly labeled
as general stage guidance. Overripe is a separate label from spoiled.

Verified on November 5, 2026: the exported ONNX model classified **103 of 104**
held-out dataset crops correctly (99.04% accuracy, macro-F1 0.9900). One result
was below the 65% score threshold and displayed as uncertain. Median inference
time on this laptop was 6.0 ms, excluding photo capture and decoding. All 20
automated tests passed. Browser upload, camera capture using a test video feed,
result download, mobile layout, and an actual attached-webcam capture were
also checked. No live banana-in-front-of-camera validation has been performed.
Full metrics and per-photo predictions are in `models/camera_only/metrics.json`
and `models/camera_only/test_predictions.csv`.

## Camera options

```bash
# Another USB/UVC camera (including compatible RealSense RGB streams)
.venv/bin/python -m ripesense.camera_only --camera 2
# Access from a second device on the same network
.venv/bin/python -m ripesense.camera_only --host 0.0.0.0
```

Browser camera access requires localhost or HTTPS. Over a plain HTTP network
connection, use **Use attached camera** or upload instead. The attached camera
button uses the camera connected to the server, not the visiting phone.
The default app binds only to localhost. The Flask development server is for
local demos; this app has no authentication and is not configured for public hosting.

## Install and retrain

Runtime dependencies: `pip install -r requirements-camera.txt`.
The supplied artifacts live in `models/camera_only/`: `stage.onnx`,
`classes.json`, `metrics.json`, and the training checkpoint `best.pt`.
Only the ONNX file and JSON files are needed to run.

For training, install `requirements-camera-train.txt`. On a CPU machine,
install CPU PyTorch wheels rather than the much larger CUDA packages:

```bash
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install onnx
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python -u \
  -m ripesense.models.train_camera_only --epochs 8
```

Training reads `datasets/raw/BANANA_DATASET75.zip` directly and prepares
annotated crops in `datasets/camera_banana/`. It retains the source archive's
train/validation/test split, keeps crops from a photograph together, and
removes exact duplicate source photos across splits. Validation chooses the
checkpoint; test data is evaluated only after that selection. Fruit identities
are unavailable, so related photos of the same banana may occur across splits.
Dataset test accuracy does not establish performance in your room or cabinet.
When retraining, choose a fresh directory with `--data datasets/camera_banana_v2`
to avoid mixing crops from different preparation runs.
Download `BANANA_DATASET75.zip` from the dataset source below if it is no
longer in `datasets/raw/`; the archive is not needed to run the trained app.

Dataset: Natiq-Ul-Haque Rohit, Abu Raihan, Afia Sarkar Mim, Fahima Fahima
Tabassum, and Aziza Haque, **Annotated Banana Ripeness Stage Dataset**, v2,
DOI [10.17632/56c75hvm7r.2](https://data.mendeley.com/datasets/56c75hvm7r/2),
CC BY 4.0. Crops are derived from its annotated images. The archive has no
class YAML; the preparation code establishes annotation IDs by majority vote
from labeled filenames and rejects an ambiguous mapping. Annotation IDs,
rather than filenames, determine individual crop labels.

## API

* `GET /api/health`: model status and dataset test metrics.
* `POST /api/predict`: multipart form, photo field named `image`.
* `POST /api/capture`: capture one frame from the configured attached camera.

Re-evaluate the exported model with:
`.venv/bin/python -m ripesense.models.evaluate_camera_only`.

```bash
curl -F image=@banana.jpg http://localhost:5001/api/predict
```
