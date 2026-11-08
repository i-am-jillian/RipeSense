# Camera assessment for fruits and vegetables

```bash
.venv/bin/python -m ripesense.camera_general
```

Open **http://localhost:5001**. Start the browser camera, upload a photo, or
use the camera attached to the computer running the service. Place one item
close to the camera in even lighting. You can leave the description empty
for automatic recognition, or enter a name such as `avocado`, `tomato`,
`green grapes`, or another produce description.

The app uses different kinds of evidence, and prints the method on every
result:

| Input | Assessment |
|---|---|
| Banana | Trained unripe / ripe / overripe classifier, plus the quality classifier when available |
| Apple, banana, grape, guava, jujube, lime, orange, pomegranate, strawberry | Trained fruit identification and fresh / rotten visual-quality classifier |
| Hass avocado | Trained five-stage classifier: unripe, breaking, first ripe stage, second ripe stage, overripe |
| Tomato | Trained fresh / spoiled visible-quality classifier |
| Other produce, including kiwi, mango, papaya and persimmon | Experimental CLIP comparison against ripeness descriptions |
| A new produce description | Experimental comparison of the photo and supplied description |
| Unclear identity, conflicting models, or mismatched description | Uncertain; asks for a closer photo or corrected description |
| Detected non-produce or empty scene | Requests a fruit or vegetable |

The recognition catalog contains 37 produce names. A free-text description
can go beyond that catalog. **This does not establish validated ripeness
assessment for every item.** The broad model compares an image to text; it
does not supply validated ripeness estimates for every produce type. Some different
fruits look alike, and both recognition and rejection can fail. An item may
be identified while its condition remains uncertain.

Fresh-looking does not mean ripe, safe, or internally sound. The camera
cannot measure firmness, sugar, invisible internal deterioration, or actual
days until spoilage. The JSON therefore leaves `days_left` empty, and the
page does not present made-up remaining-life numbers. The original sensor
fusion system remains available separately.

## Local models

The app runs offline after installation. Photos are processed locally, without
Gemini or another remote vision API. Model weights and tokenizer are pinned
to a specific repository revision and verified with SHA-256:

```bash
.venv/bin/pip install -r requirements-camera.txt
.venv/bin/python -m ripesense.models.download_openvision
```

The downloaded general model uses about 149 MiB of files in
`models/openvision/`. Provenance is in `models/openvision/source.json`.
It is the MIT-licensed
[Xenova ONNX export of CLIP ViT-B/32](https://huggingface.co/Xenova/clip-vit-base-patch32),
based on [CLIP](https://github.com/openai/CLIP). The app uses the image/text
embeddings for recognition and description matching. Relative caption-match
scores are not calibrated probabilities of ripeness or correctness.

The multi-fruit specialist is in `models/camera_quality/`. It uses a frozen
ImageNet-pretrained MobileNetV3-Small with a trained multinomial classification
head. Training chooses regularization using validation data and evaluates
test data afterward. Its input photographs come from
[Fresh and Rotten Fruits](https://data.mendeley.com/datasets/bdd69gyhv8/1)
(Nusrat Sultana, Musfika Jahan, Mohammad Shorif Uddin) and
[FruitNet](https://data.mendeley.com/datasets/b6fftwbr2v/1)
(Vishal Meshram, Kailas Patil), both CC BY 4.0. Exact duplicate files are
removed across splits. Related photographs of the same fruit may still be
in different splits because fruit identities are unavailable.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -u \
  -m ripesense.models.train_camera_quality
```

The service also recognizes a completed fine-tuned quality model in
`models/artifacts/`. The running service loads a specialist when its final
model, class list, and metrics files become available. `/api/health` lists
the trained item types actually loaded, separately from the general catalog.

The earlier banana model and its evaluation remain documented in
[CAMERA_ONLY.md](CAMERA_ONLY.md).

Avocado and tomato have additional trained heads in `models/camera_extras/`,
using the same fixed image features with separate classification heads. The
avocado model is specific to Hass avocados. Its source is
[Hass Avocado Ripening Photographic Dataset](https://data.mendeley.com/datasets/3xd9n945v8/1)
by Pedro Xavier, Pedro Rodrigues and Cristina L. M. Silva (CC BY 4.0).
Preparation reconciled 14,710 image files with 14,722 spreadsheet rows,
excluding 12 rows without image files. It selects 1,500 photographs with
balanced stage counts; all images of a fruit stay in the same split, using
storage group plus sample ID as the identity. Tomato crops use the source
split, keeping augmented copies of a source photograph together. Tomato labels
describe visible quality, not the unripe-to-ripe progression.

To reproduce these extra heads, obtain the source spreadsheet
`datasets/raw/Avocado_Ripening.xlsx` and the tomato archive
`datasets/raw/Tomato_Dataset.v5.zip` from their linked sources, then run:

```bash
.venv/bin/python -m ripesense.models.prep_camera_extras
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python \
  -m ripesense.models.train_camera_extras
```

Their per-class metrics, fruit/group split counts and test predictions live
alongside the heads. The avocado figures apply to the sampled Hass fruit
dataset, not to other cultivars or a different camera setup.

## API

* `GET /api/health`: readiness, loaded trained item types, general catalog,
  and specialist dataset metrics.
* `POST /api/predict`: multipart form with `image` and optional `item` text.
* `POST /api/capture`: use the server's camera; optional JSON `{"item":"avocado"}`.

```bash
curl -F image=@photo.jpg -F item=avocado http://localhost:5001/api/predict
```

Run with `--camera 2` to select another UVC camera or `--host 0.0.0.0` to
access it from another device on the local network. Browser camera access
requires localhost or HTTPS; attached-camera capture and upload can be used
over ordinary HTTP. This is a local demonstration service, with no public
hosting or authentication configured. Pi performance and live fruit-camera
accuracy have not been validated.

## Additional example photographs

The local `datasets/openvision_examples/` directory contains a small set of
downloaded demonstration photographs, not a measured test benchmark. Its
`source.json` records download URLs and hashes. Sources are
[strawberry and avocado images](https://data.mendeley.com/datasets/zysvgmxcyz/1)
(Pooja Kamat and coauthors) and
[Tomato_detection](https://data.mendeley.com/datasets/vy5y8p2kdb/1)
(Arafat Islam, Talukder Oishe, Mahbubul Hasan Fuad), CC BY 4.0.
