# RipeSense

A wall-mounted cabinet that tells you how ripe a fruit is, and how many days it has left, by fusing
three sensors on a Raspberry Pi 5:

| Sensor | What it measures | Features |
| --- | --- | --- |
| Intel RealSense (color + depth) | Peel colour, spots, size and volume | `c_*`, `i_*` |
| AS7341 11-channel spectral | Chlorophyll and pigment reflectance under white and 850 nm light | `s_*` |
| Gas sensor (MQ via ADS1115, or BME680/688) | Ripening and spoilage volatiles in the sealed chamber | `g_*` |

One tap on the pad runs: **camera → spectral → gas (120 s, running the whole time) → model → screen**.
The screen shows a verdict, a days-left range, a confidence and which sensors agree. **Store mode**
suggests a markdown price instead.

## Try it now, without hardware

```bash
pip install -e ".[dev]"
python -m ripesense run --sim          # http://localhost:8000  (kiosk)   /collect  (data + simulator)
python -m pytest -q                    # 10 end-to-end tests in simulation
```

On `/collect`, use **Simulator → Put fruit in**, then **Tap the ring**.

## Wiring (BCM pin numbers)

The 3.5" SPI screen uses GPIO 7–11, 17, 24 and 25, so connect it through a 40-pin ribbon and avoid those pins.

| Part | Pin | Notes |
| --- | --- | --- |
| AS7341, ADS1115 or BME680 | SDA 2, SCL 3 (I2C) | addresses 0x39, 0x48, 0x76/0x77 |
| TTP223 touch pad | GPIO 5 | output goes HIGH on touch |
| Door reed switch | GPIO 6 → GND | set `door = -1` if not fitted |
| LED strip MOSFET | GPIO 12 | camera light |
| White spectral LEDs MOSFET | GPIO 13 | |
| 850 nm IR LEDs MOSFET | GPIO 19 | |
| Mixing fan MOSFET | GPIO 26 | |
| MQ sensor analog out | ADS1115 A0 | divide 5 V down if the ADS1115 runs at 3.3 V |

## Set up the Pi

```bash
git clone <this repo> ripesense && cd ripesense
bash deploy/install.sh               # packages, I2C/SPI, venv, service, kiosk autostart
bash deploy/build_librealsense.sh    # ~1 hour, RSUSB backend, Python bindings
.venv/bin/python -m ripesense doctor # finds each part and says what is missing
```

Edit `config.toml` for your parts (`[gas] kind`, camera model, pins). Put `GEMINI_API_KEY=...` in
`/etc/ripesense.env` for Gemini tips; without it, built-in tips are used.

## Calibrate (each session)

```bash
.venv/bin/python -m ripesense calibrate
```

1. White puck in the seat, door shut: white reference for the spectral sensor.
2. Chamber empty, door shut: reference image and floor depth for the camera.

Or use the buttons on `/collect`. Power the MQ sensor 20+ minutes before scanning; the screen shows the warm-up.

## Collect data and train

1. Number each fruit with a marker (B01, B02 …). Buy several ripeness stages.
2. On `/collect`, enter the fruit ID and press **Scan now**, 3 scans per fruit, turning it in between.
3. Label each scan with its true stage. When a fruit is judged spoiled, fill in days-left for its earlier scans.
4. **Train model**, or `python -m ripesense train`. Accuracy is measured with whole fruits held out.
   `top_features` shows which sensors matter, which is useful for the pitch.

Until a model is trained, a rule-based estimate is used and the screen shows "untrained".

The camera can also use an image model for fruit type and rot. Train it on a laptop or Colab with
`tools/train_image_model.py` (Fresh-and-Rotten or FruitNet datasets), then copy
`fruit_classifier.onnx` and its `.labels.json` into `models/`.

## Recipes and the QR code

After each result, a recipe that suits the fruit's exact ripeness arrives a few seconds later
(very ripe bananas → banana bread, green tomatoes → fried green tomatoes). Gemini writes it as
structured JSON when `GEMINI_API_KEY` is set; otherwise a built-in recipe is used, so it also works
offline. A QR code appears on the screen; tap it to enlarge. A phone opens the recipe at
`http://<pi>:8000/r/<scan>`, with checkable ingredients and numbered steps.

The phone must be on the same Wi-Fi as the Pi. Hackathon Wi-Fi often blocks device-to-device
traffic. If phones can't open the page, either set `[qr] target = "search"`, so the QR opens a web
search for the recipe title, or run the Pi as its own hotspot and set `public_url`. The model name
is in `[gemini]`; if Google rejects it, the `fallback_models` are tried in order.

## How the gas baseline works

The gas monitor samples all the time. When the door opens, the chamber fills with room air, and the
median of those readings becomes the baseline. The scan measures how far the closed chamber rises
above it. This avoids the previous fruit's gas leaking into the next reading.

## Layout

```
ripesense/
  hardware/   camera.py (RealSense/OpenCV/mock), spectral.py (AS7341), gas.py (MQ/BME680 + monitor),
              io.py (gpiozero), sim.py (shared simulated fruit)
  features.py segmentation + colour/size, reflectance indices, gas rise
  model.py    heuristic + fusion model (gradient-boosted trees, quantile days-left, calibration, agreement)
  scan.py     the sequence and the state the UI polls
  verdict.py  home advice and store markdowns
  storage.py  SQLite log, images, CSV export
  web.py      Flask API, QR codes, recipe page
  recipes.py  Gemini recipe + tip, built-in fallback   sinks.py   Tiger Data / PostgreSQL log (optional)
  static/     kiosk.* (screen), collect.html (data), recipe.html (phone page from the QR)
deploy/       install, librealsense build, systemd service, kiosk
tools/        image-model trainer
tests/        end-to-end tests in simulation
```

## Hackathon note

MLH events expect projects to be built during the event. If you bring this as starter code, tell
the organizers and say so on your Devpost page.
