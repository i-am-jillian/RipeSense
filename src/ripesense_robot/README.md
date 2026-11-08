# RipeSense Robot: xArm 6 fruit sorter (end-to-end)

An xArm 6 on a linear rail looks for fruit (first **in front**, at bush/tree height, then on the **table/floor**),
classifies each one as ripe / spoiled / unripe / not sure, and **sorts it**: ripe fruit into the ripe bin, spoiled
fruit into the spoiled bin, unripe fruit left to ripen. Every scan is stored in **Tiger Data (TimescaleDB)** so the
system can answer *"which fruit should we inspect now, and what changed since the last scan?"* A laptop dashboard
shows the live camera, the missions, the sorting and the fruit positions.

```
 RealSense D435i (wrist)          Raspberry Pi 5 (on the arm)                         Laptop
 colour + depth  ──────────►  1 detect   depth -> table plane -> objects >12 mm     dashboard.py  (localhost:5050)
                               2 classify crop -> MobileNetV3 (ONNX, 17 classes)  ◄─ ssh: start/stop missions,
 xArm 6 + gripper ◄──────────  3 decide   ripe / spoiled / unripe / not sure        status, live log
 (192.168.1.232)               4 pick     gentle creep-close grip, slip check       camera via :8091 tunnel
                               5 drop     swing above the bin, straight down
                               6 log      every scan + action -> Tiger Data  ────►  Tiger Cloud (TimescaleDB)
```

## Folder map

| Path | What is in it |
|---|---|
| `robot_xarm/runtime/` | Everything that runs **on the Pi on the arm** (mission scripts, picking, search, recovery, Tiger logging, calibration, data collection). See [its README](robot_xarm/README.md). |
| `robot_xarm/dashboard/` | Laptop dashboard (Flask + one HTML page; `/classic` = detailed view with charts). |
| `robot_xarm/calibration/` | Hand-eye calibration board (ChArUco 6x8) and the captured calibration images. |
| `training/robot_fruit/` | `train_robot_fruit.py` = training script of the **deployed model (v4)**; `history/` = v1-v3. |
| `training/logs/` | Full training logs + metrics JSON of v1, v2, v3, v4 (and the cabinet camera models). |
| `models/robot_fruit_v4/` | **Deployed model**: `robot_fruit_model.onnx` (6 MB), class list, metrics. |
| `datasets/` | **Our own data**: `xarm_realsense/` (890 wrist-camera crops) and `ainex_robot/` (1,190 AiNex photos). See [datasets/README.md](datasets/README.md). |
| `robot_ainex/` | Earlier version on the Hiwonder AiNex humanoid (pick + dashboard). |
| `ripesense/`, `models/camera_*`, `training/camera/`, `deploy/`, `cad/`, `tests/` | The RipeSense smart-cabinet / kiosk app this grew out of (camera + spectral + forecast). See [docs/CABINET_README.md](docs/CABINET_README.md). |

## The pipeline, step by step

1. **Find objects (depth, no model needed).** 15 depth frames are median-filtered, back-projected to the arm base
   frame with the hand-eye calibration, a table plane is fitted robustly, and everything 12-130 mm above it becomes
   a candidate with an exact 3-D position, height and footprint (`pick_all.py: detect()`).
   For forward views (`row_search.py`) the same is done against depth edges so fruit separate from the background.
2. **Classify.** Each candidate's colour crop (224 x 224) goes through the MobileNetV3 ONNX model with OpenCV-DNN.
3. **Decide.** `< 50 %` confidence -> *not sure* (left alone) · `*_rotten` -> spoiled bin · `*_unripe` -> left to
   ripen · `not_fruit` -> ignored · otherwise -> ripe bin.
4. **Pick gently.** Close-up look to re-centre on the fruit -> descend -> the gripper *creeps* closed in 2 mm steps
   until the fingers stop (contact), then squeezes only 1-2.5 mm past contact (per fruit; spoiled fruit x0.6).
   Wide fruit (> 78 mm) are gripped at their 75 mm section. A slow 2.5 cm test lift re-squeezes if the fruit slips.
5. **Drop.** Lift, pull in (far fruit), rise to carry height, swing to a point **above the bin** (computed from the
   hand-taught bin pose so the wrist never flips), come straight down, open, shake the wrist for wide fruit.
6. **Log.** Every scan (where, when, what was measured, what the model said) and every action goes to Tiger Data.
7. **Recover.** If any step fails, `recover.py` clears the arm error, drops a fruit that is still held into the
   right bin, returns to the start pose and the mission continues. Three failures in a row stop the mission.

### Missions (dashboard buttons, or on the Pi)

| Mission | Command on the Pi | What it does |
|---|---|---|
| Full demo | `~/job.sh demo ~/demo.sh full` | front search -> table search -> ripe -> ripe bin -> spoiled -> spoiled bin (4 table views each) |
| Sort into bins | `~/job.sh sort ~/demo.sh pick` | the two picking phases only |
| Autopilot | `~/job.sh auto env FRONT=1 ~/autopilot.sh` | loops on its own: search every 5 cycles, sort, wait 30 s, repeat; stop with `touch /tmp/autopilot.stop` |
| Search | `~/job.sh search ~/ripesense_env/bin/python ~/row_search.py --exit` | find and sort into categories, no picking |

`job.sh` guarantees one arm job at a time and hands the camera back to the live viewer afterwards.

## Tiger Data scan history (`robot_xarm/runtime/scan_history.py`)

* `fruit_scans` **hypertable**: `ts, session, source (front/floor/overview/inspect/pick), fruit_id,
  x/y/z, rail_mm, cam_pose, height/length/width, hue/sat/val, dark_frac, spectral (AS7341, when fitted), model,
  cls, conf, p_rotten, probs (top 3), action (to_bin / to_spoiled_bin / pick_failed), crop_path`.
* `fruits`: identity by **position + fruit type** (same kind within 40 mm = same fruit).
* `python scan_history.py next` -> ranked "inspect now" list (40 % spoiled probability, 20 % uncertainty,
  20 % getting worse, 10 % label changed, 10 % time since last scan); `changes` -> what changed per fruit;
  `history N`; `flush` (rows buffered while offline).
* Connection string: environment variable `TIGER_DATABASE_URL` (or `~/.tiger_env`). **Never commit it.**

## Running it

**Pi (Raspberry Pi 5, Bookworm, on the arm):** librealsense 2.56 built with the RSUSB backend, a venv
`~/ripesense_env` with `xarm-python-sdk opencv-python numpy pyyaml psycopg[binary]`, and the files of
`robot_xarm/runtime/` + `models/robot_fruit_v4/*` copied to `~/` and `~/models/`. Arm on the Pi's ethernet
(192.168.1.10/24 -> arm 192.168.1.232).

**Laptop:**
```bash
pip install flask "psycopg[binary]"
export TIGER_DATABASE_URL=postgres://...            # from the Tiger Cloud console
ssh -N -L 8091:localhost:8091 sujal@<pi-ip> &       # live camera tunnel
PI_SSH=sujal@<pi-ip> python robot_xarm/dashboard/dashboard.py   # -> http://localhost:5050
```

**Teach the bins (once):** `python manual.py on` (arm becomes hand-movable) -> move the gripper ~10 cm above the bin,
pointing down -> `python manual.py save ripe_bin` (or `spoiled_bin`) -> `python manual.py off`.

**Calibration:** `handeye_calib.py --auto` (15 poses over the ChArUco board, printed at 93 % = 23.3 mm squares) +
an F/T touch-down for the table height. Result lives in `runtime/config.yaml` (`camera.eef_to_color`).

## The model and how it was trained

* **MobileNetV3-Small** (ImageNet weights, fine-tuned), 6 MB ONNX, **17 classes**: apple fresh/rotten,
  avocado ripe/rotten, banana unripe/ripe/overripe/rotten, lemon fresh, mango fresh/rotten,
  orange fresh/rotten, strawberry fresh/rotten, other fruit, not fruit.
* **Data**: public photos (x1 sampling weight) + our AiNex robot photos (x3) + our xArm RealSense crops (x5).
* **Training**: 12 epochs, AdamW, one-cycle LR 1e-3, batch 64, label smoothing 0.05, class-balanced sampler;
  augmentation = random crop, flips, ±25° rotation, colour jitter, background swap. ~35 min on a laptop CPU.
* **Honest split**: robot and RealSense photos are split *by time* inside each collection round
  (75 / 10 / 15 %), so the test views were never seen. Best epoch = 0.25 public + 0.15 robot + **0.6 RealSense** F1.
* **Retrain**: `python training/robot_fruit/train_robot_fruit.py` (needs `datasets/public/`, see datasets README).

| Version | What changed (each fixed something seen on the real robot) | Test (all) | RealSense test |
|---|---|---|---|
| v1 | public + AiNex only | 97.9 % (n 1155) | n/a (avocados misread on the arm) |
| v2 | + 890 xArm RealSense crops | 97.6 % (n 1273) | 100 % (n 118) |
| v3 | + not-fruit crops (tools, tape), spoiled avocado with sticker hidden | 95.7 % (n 1283) | 100 % (n 128) |
| **v4 (deployed)** | the berries were spoiled, relabelled `strawberry_rotten` | **97.6 %** (n 1283, macro-F1 0.986) | **100 %** (n 128) |

Full logs with per-class recall and confusion matrices: `training/logs/robot_fruit_v*_train.log`.

## Results on the real arm (8 Nov 2026)

* Full demo: **9 fruit sorted in one run** (7 ripe -> ripe bin, 2 spoiled -> spoiled bin), no collisions, no
  recoveries; an avocado at 31-36 % confidence was correctly left as "not sure".
* Gentle grip: fingers close on the widest part (lemon: 60 mm contact on a 58 mm lemon); a 100 mm mango is
  gripped at its 75 mm section and released cleanly.
