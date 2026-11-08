# RipeSense: smart fruit ripeness cabinet

**Camera version:** run `.venv/bin/python -m ripesense.camera_general` and
open **http://localhost:5001** to photograph or upload produce, automatically
identify it, or enter an item description. It uses trained fruit classifiers
where available and explicitly labeled experimental image/text matching for
other produce. No gas, spectral or GPIO hardware is required. See
[CAMERA_GENERAL.md](CAMERA_GENERAL.md) for setup, evaluation, and model limits.

**Hackathon features** (forecast, hidden-bruise IR, teach-a-fruit, why-heatmap,
markdown optimizer, conformal ranges, e-nose, fruit profile, and more): see
[HACKATHON_FEATURES.md](HACKATHON_FEATURES.md) for what each one does, how to
demo it, measured numbers and honest limits. The laptop demo page is
**http://localhost:5000/camera**.

A Pi 5 cabinet that scores a fruit's ripeness in about 2 minutes by fusing three
sensors: an Intel RealSense (colour, size, volume, spots), an AS7341 spectral
sensor (chlorophyll and pigment chemistry) and a gas sensor (spoilage volatiles).
It shows a verdict, a days-left range, a confidence and how many sensors agree.
In Store mode it also shows a markdown price.

Everything runs on the Pi. The cloud is optional: a Gemini tip sentence and a
Tiger Data store log.

```
ripesense/
  config.yaml              pins, I2C addresses, gas sensor type, markdown tiers
  ripesense/
    hw/                    camera.py · spectral.py · gas.py · gpio.py · check.py (bench test)
    scan.py                the touch → camera → spectral ‖ gas → model → result sequence
    features.py            per-sensor features (doc p.11-12)
    store.py               SQLite + data/scans.csv + raw images
    models/                featurize · train_fusion · infer · image (ONNX) · train_image
                           prep_datasets (public photos) · prep_deephs + spectral_prior (DeepHS)
    cloud/                 gemini.py (tip only) · tiger.py (store log)
    sim/                   simulated fruit + synthetic dataset (laptop dev only)
    app.py, templates/     kiosk (/), camera view (/camera), data collection (/collect),
                           store dashboard (/store), simulator (/dev)
  cad/
    make_panels.py         laser-cut panels → cad/laser/sheet1|2.svg/.dxf, parts/, preview.png
    hybrid_assembly.scad   printed parts + assembly preview; export_stl.sh
  deploy/                  setup_pi.sh · build_librealsense.sh · systemd + kiosk · SCREEN.md
  tests/
```

## 1 · Try it on a laptop (no hardware)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
# Optional: a synthetic dataset + model so all three sensor dots light up
RIPESENSE_DATA_DIR=data_sim .venv/bin/python -m ripesense.sim.make_dataset
RIPESENSE_DATA_DIR=data_sim RIPESENSE_MODELS_DIR=data_sim/models .venv/bin/python -m ripesense.models.train_fusion
# Run with simulated sensors (FAST = 8 s gas read instead of 120 s)
RIPESENSE_MOCK=1 RIPESENSE_FAST=1 RIPESENSE_DATA_DIR=data_sim RIPESENSE_MODELS_DIR=data_sim/models \
  .venv/bin/python -m ripesense.app
```
Open http://localhost:5000/dev. It shows the 480×320 screen next to controls
that pick the simulated fruit and stage. Press **Calibrate** on /collect once
before scanning.

> Synthetic data only proves the code runs. Its accuracy numbers mean nothing.
> Real numbers come from real scans.

Tests: `.venv/bin/python -m pytest`. If ROS or other global pytest plugins
interfere, run `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest`.

## 1b · Pre-trained models from public data (handoff p.13–14)

These models work before you have collected any cabinet scans. Each one is
optional: the system uses whatever is in `models/artifacts/`.

| Model | Data | Result (held-out test) | Used as |
|---|---|---|---|
| `stage_model.onnx`: unripe / ripe / overripe from the photo | Annotated Banana Ripeness Stages (Mendeley 56c75hvm7r, CC BY 4.0), 820 banana crops | 98.4% accuracy, 3 errors out of 125 | Feature `img_stage_score`; votes on the stage (bananas only) |
| `image_model.onnx`: fruit type + fresh/rotten, 9 fruits | Fresh and Rotten Fruits (bdd69gyhv8) + FruitNet (b6fftwbr2v) + the banana-stage crops as fresh_banana, CC BY 4.0, 17.7k images | 98.7% accuracy, 18 classes | Feature `img_rot_prob`; fruit type; the "spoiled" share |
| `spectral_prior.joblib`: ripeness from AS7341-band reflectance | DeepHS Fruit mango + kaki, Specim/Corning cubes averaged to AS7341 channels (80 fruits) | **51% vs a 51% majority baseline: no signal.** Kept but not used (load guard) | Not used; spectral learns from your scans |

**End-to-end check** (`python -m ripesense.models.evaluate_system`): 971
held-out photos run through the same featurize → predictor path as a real
scan, scoring the final verdict.

| Photos | Correct | Wrong | Uncertain |
|---|---|---|---|
| Unripe / ripe / overripe bananas | 98% / 100% / 95% | 0% | 2–5% |
| Fresh fruit (9 kinds), not called spoiled | 96% | 0.2% | 4% |
| Rotten fruit, called spoiled | 99% | 0.7% | 0.2% |
| **Overall** | **97.5%** | **0.4%** | **2.1%** |

This check found a real bug that per-model accuracy hid. Before the banana
crops were added to the fruit-type model, green bananas were read as guava,
so the stage model never ran (overripe bananas: 22% correct, unripe: 36%).

All results come from photos taken in the dataset's own setup (white
backgrounds, phone cameras). Expect lower accuracy in the cabinet until you
add your own photos. The stage model is trained with `--bg-swap`, which
swaps white backgrounds for dark ones to narrow that gap.

**How predictions are made** (`models/infer.py`), best tier first:
1. **fusion**: your own LightGBM, trained on cabinet scans. The image-model
   outputs above are among its features.
2. **pretrained**: before you have scans. The stage model sets the stage,
   the image model sets P(rotten) → spoiled, and other fruits get
   fresh/edible vs spoiled. The screen's `/camera` view shows the reasoning.
3. **heuristic**: no models at all. A colour rule, flagged "uncalibrated".

Rebuild everything (about 8 GB of downloads; each zip is read in place, then
deleted):
```bash
python -m ripesense.models.prep_datasets banana  BANANA_DATASET75.zip --classes unripe,ripe,overripe --out datasets/banana_stages
python -m ripesense.models.prep_datasets quality "Original Image.zip" FruitNetDataset.zip --max-per-class 1200 --out datasets/fruit_quality
python -m ripesense.models.train_image --data datasets/banana_stages --name stage_model --bg-swap 0.5 --epochs 15
python -m ripesense.models.train_image --data datasets/fruit_quality --name image_model --epochs 8 --max-per-class 600
python -m ripesense.models.prep_deephs annotations-upd-2024-01-09.zip Mango.zip Kaki.zip --out datasets/deephs_as7341.csv
python -m ripesense.models.spectral_prior --csv datasets/deephs_as7341.csv
```
(Training needs `torch torchvision onnx`. Run it on a laptop or Colab, not the Pi.)

### Camera-only mode: the whole system with just a USB webcam

```bash
RIPESENSE_CAMERA_ONLY=1 RIPESENSE_WEBCAM=0 .venv/bin/python -m ripesense.app
```
Open http://localhost:5000/camera for a live view, the kiosk screen, a
**Scan** button and the model's reasoning. Scans are logged and labelled
exactly like full-cabinet scans (/collect), so photos you take now become
training data. This is also the handoff's fallback if librealsense won't
build on the Pi (`camera.backend: webcam`). Use `RIPESENSE_WEBCAM=1` for a
second (USB) camera.

## 2 · Pi 5 setup

```bash
git clone <this repo> ~/ripesense && cd ~/ripesense
bash deploy/setup_pi.sh             # apt, I2C/SPI, venv, service, kiosk autostart
bash deploy/build_librealsense.sh   # 45–90 min: start it early (RSUSB backend, no kernel patches)
# Screen: follow deploy/SCREEN.md (don't run the vendor LCD-show script on a Pi 5)
.venv/bin/python -m ripesense.hw.check      # PASS/FAIL per device
sudo systemctl start ripesense              # then browse to http://<pi-ip>:5000/collect
```

In `config.yaml`, set `gas.type` to `mq` (MQ module + ADS1115) or `bme680`
(BME680/688). If you have no reed switch, set `gpio.require_door: false`.

If librealsense won't build, set `camera.backend: webcam`. That reads the
RealSense colour stream as a plain UVC camera, with no depth. Size and volume
features then become NaN, and the model handles that because it is trained with
sensor dropout.

### Wiring (BCM GPIO numbers; physical pin in brackets)

| Device | Connect to |
|---|---|
| AS7341 (STEMMA QT) | 3V3 (1), GND, SDA GPIO2 (3), SCL GPIO3 (5). Address 0x39 |
| BME680/688 *or* ADS1115 | Same I2C bus, 3V3. BME 0x77 (0x76 if SDO→GND); ADS1115 ADDR→GND = 0x48 |
| MQ module (if MQ) | VCC 5V (2), GND; AO → 10 kΩ → ADS A0, with 20 kΩ from A0 to GND (5 V → 3.3 V; `divider_ratio: 1.5`) |
| TTP223 touch | VCC 3V3, GND, OUT → GPIO5 (29) |
| Reed switch | GPIO6 (31) ↔ GND (internal pull-up) |
| MOSFET: LED strip | SIG ← GPIO12 (32), V+ 5 V |
| MOSFET: 2× white 5 mm LED | SIG ← GPIO13 (33); ~100 Ω per LED at 5 V |
| MOSFET: 2× 850 nm IR LED | SIG ← GPIO16 (36); ~180 Ω per LED at 5 V |
| MOSFET: 40 mm fan | SIG ← GPIO26 (37), 5 V |
| 3.5" screen | 40-pin ribbon (uses SPI0, GPIO 17/24/25, which is why the pins above avoid them) |
| RealSense | Blue USB 3 port |
| Power | Official 27 W USB-C supply (RealSense, screen, fan and MQ heater all draw from the 5 V rail) |

## 3 · Collect data (Thu → Sat)

Open **/collect** on a laptop on the same network.

1. Number each fruit with a marker. Buy bananas at 4 stages, plus avocados if
   you want a second fruit.
2. Calibrate each session: put the white puck in the cradle, close the door,
   then press **Calibrate**.
3. Enter the fruit ID, type and stage (from the peel chart), then **Scan**. Do 3
   scans per fruit per session, turning the fruit between scans, twice a day.
4. When a fruit is judged spoiled, enter its ID under **When a fruit spoils**.
   This fills in days-left for every earlier scan of it.
5. Press **Train**. You get macro-F1 and days MAE, both evaluated with fruits
   held out by fruit, plus `feature_importance.png` for the pitch.

Labels can be edited in the table at any time; `data/scans.csv` follows. Raw
images are kept, so features can be re-extracted without rescanning.

**Image model (optional, on Colab/laptop):**
`python -m ripesense.models.train_image --data <ImageFolder>` (e.g. Mendeley
"Fresh and Rotten Fruits" / FruitNet). Copy `image_model.onnx` and
`image_classes.json` into `models/artifacts/`. It adds fruit type and
`img_rot_prob`. The model works without it.

## 4 · Build the cabinet

```bash
python cad/make_panels.py --t 2.9      # measure your acrylic first; regenerates everything
bash cad/export_stl.sh                 # needs OpenSCAD; STLs → cad/stl/
```
* `cad/laser/sheet1|2.svg/.dxf` are two 600×300 mm sheets with about 11 m of
  cut line. Red is cut. Black is engrave; convert the text to paths first.
* **Cut a cardboard test corner first** and tune `--clear` for your laser's kerf.
* If the AS7341 isn't centred under the aperture, move the cradle hole with
  `--bx/--by`.
* Printed parts (`part=` in the .scad): cradle_insert and camera_clamp in
  black, cal_puck in white, screen_clip ×2, magnet_block ×4, door_handle.
  Print the cradle upright with supports on the build plate only, for the
  flange overhang.
* Seal the floor/ceiling cable holes with hot glue or putty so gas can build up.
* The camera clamp defaults to a D435 (90×25×25). For a D415/D455, change `cam`
  and `cam_screw_dx` in the .scad and `cam_screw_dx` in make_panels.py.

* **No laser cutter?** `bash cad/export_stl.sh --printed` also exports an
  all-printed cabinet: three stacking sections (base / chamber / head) that
  fit an MK4S bed, plus door and lids. About 25–30 h of printing.

> The .scad and panel geometry were written without OpenSCAD available, so they
> haven't been rendered. Open `hybrid_assembly.scad` in OpenSCAD (F5), check
> each part, and dry-fit before gluing.

## 5 · Cloud (optional)

* **Gemini** (`GEMINI_API_KEY` in `.env`): writes one tip sentence from the
  model's structured output. It never decides the verdict. Offline, a built-in
  tip is used instead.
* **Tiger Data** (`TIGER_DATABASE_URL` in `.env`, `cloud.tiger_enabled: true`):
  every scan is logged to a hypertable. `/store` shows stock entering each
  markdown tier per day. Without it, `/store` reads the local SQLite log.

## Known limits (be upfront in the pitch)

* With two days of data, days-left is rough. It is shown as a range, and stage
  classification is the headline metric.
* One banana in about 3.5 L for 120 s gives a weak MQ signal. BME68x is better.
  Gas mostly separates overripe from spoiled, and the feature-importance chart
  shows honestly how much each sensor contributes.
* Markdown tiers are a starting point, not researched pricing. They can be
  edited in `config.yaml`.
