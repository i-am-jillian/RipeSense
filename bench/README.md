# RipeSense AS7341 local bench test

Run from the VS Code terminal; no Colab or camera is needed. The supplied root-level `deephs_ripeness.joblib` contains avocado, kiwi, persimmon, papaya and mango pipelines, using scikit-learn **1.6.1**. Apple is supported for **sensor-only measurements**, not ripeness prediction.

## 1. Wiring and illumination

The connected board was identified as **Arduino UNO R4 WiFi**, USB serial `/dev/cu.usbmodemF412FA6378E02`. Your table said R3; select **R4 WiFi** for this connected board. Use the main labeled SDA/SCL header (`Wire`), not the separate Qwiic connector.

| UNO R4 WiFi / Minima main header | Adafruit AS7341 breakout |
|---|---|
| 5V | VIN |
| GND | GND |
| SDA (A4) | SDA |
| SCL (A5) | SCL |

This supply wiring is for the **Adafruit breakout**, which has regulation/level shifting. See [Adafruit pinouts](https://learn.adafruit.com/adafruit-as7341-10-channel-light-color-sensor-breakout/pinouts) and the [R4 WiFi board documentation](https://docs.arduino.cc/hardware/uno-r4-wifi/).

Clamp the sensor and use a stable, fixed white lamp with ambient light excluded. Mark the fruit spot and distance; put the diffuse white reference at that same position. Let the lamp warm up. The sketch defaults to **external illumination, onboard LED off**, fixed **16× gain**, `ATIME=99`, `ASTEP=999`: 278 ms per integration (`(ATIME+1)*(ASTEP+1)*2.78 µs`). F1–F4 and F5–F8 require sequential integrations, so keep the fruit still for the whole scan.

Optionally set `USE_BREAKOUT_LED=true` in the sketch for fixed 4 mA breakout illumination if your setup uses that LED. Use the same choice/current for every measurement, including dark (cover the aperture while leaving lighting settings unchanged). There is no automatic exposure or gain. If you change gain, timing, lamp intensity, LED current, angle or distance, restart **all** calibration.

## 2. Install and upload the Arduino sketch

**Arduino IDE:** open `arduino/RipeSense_AS7341/RipeSense_AS7341.ino`. In Boards Manager install **Arduino UNO R4 Boards** (Renesas UNO core). In Library Manager install **Adafruit AS7341**, accepting its **Adafruit BusIO** dependency. Select **Tools → Board → Arduino UNO R4 Boards → Arduino UNO R4 WiFi**, and **Tools → Port → /dev/cu.usbmodemF412FA6378E02**. Compile, then Upload. For a different R4 Minima board, choose UNO R4 Minima instead.

Alternatively, these commands run from the VS Code terminal on this Apple Silicon Mac:

```sh
cd /Users/jill/Desktop/Projects/RipeSense
brew install arduino-cli
arduino-cli core update-index
arduino-cli core install arduino:renesas_uno
arduino-cli lib install "Adafruit AS7341"
arduino-cli board list
arduino-cli compile --fqbn arduino:renesas_uno:unor4wifi arduino/RipeSense_AS7341
arduino-cli upload --port /dev/cu.usbmodemF412FA6378E02 --fqbn arduino:renesas_uno:unor4wifi arduino/RipeSense_AS7341
```

The Minima FQBN is `arduino:renesas_uno:minima`; use it only for that board. Re-list ports if USB reconnects change the name. Uploading replaces the board's current sketch.

You can inspect the stream with Serial Monitor at **115200 baud**. Then **close Serial Monitor and Serial Plotter**, including any VS Code serial extension, before running Python; only one program should own the port.

The sketch repeats this header every ten rows:

```text
kind,seq,millis,F1,F2,F3,F4,F5,F6,F7,F8,gain,atime,astep,led_ma,saturated
DATA,1,1000,100,120,130,140,150,160,170,180,16,99,999,0,0
ERR,2,1700,READ_ALL_CHANNELS_FAILED
```

The numbers above illustrate the protocol, not measured sensor data. Named Adafruit F1–F8 channel enums avoid the duplicate clear/NIR slots in the library's raw array. `saturated=1` means the latched hardware saturation status or a visible count near the integration-dependent limit was detected. Read/configuration errors produce `ERR` rows, not fabricated measurements. If I²C communication stalls inside the library, Python times out; check wiring and reset the board.

## 3. Python environment in VS Code

Use Python **3.11–3.13** (the checked environment here is 3.13). The laptop's default `python3` may be 3.14, so use the explicit Homebrew Python 3.13 executable below:

```sh
cd /Users/jill/Desktop/Projects/RipeSense
/opt/homebrew/bin/python3.13 -m venv .venv-bench
source .venv-bench/bin/activate
python -m pip install -r bench/requirements.txt
python bench/as7341_bench.py --check-model --model deephs_ripeness.joblib
python bench/as7341_bench.py --list-ports
```

In VS Code, **Python: Select Interpreter → Enter interpreter path** and select `.venv-bench/bin/python`. The script refuses a scikit-learn version mismatch instead of silently running a pipeline under a different version. Load your own trusted training bundle. If replacing the bundle with one trained in another environment, match that training environment before running.

For a modeled fruit:

```sh
python bench/as7341_bench.py --port /dev/cu.usbmodemF412FA6378E02 --model deephs_ripeness.joblib --fruit mango --samples 20 --scans 3
```

For your demo apple (including a rotten apple, with no implied classifier support):

```sh
python bench/as7341_bench.py --port /dev/cu.usbmodemF412FA6378E02 --model deephs_ripeness.joblib --fruit apple --samples 20 --scans 3
```

Use any unsupported fruit name to record a sensor-only session. On Windows, activate `.venv-bench\Scripts\Activate.ps1` and pass a listed port such as `COM4` instead. Model and log paths are configurable with `--model` and `--log-dir`.

## 4. Follow the prompts

1. **Dark:** cover the sensor aperture completely. Leave the lamp and sensor settings unchanged. Press Enter; Python discards two in-flight readings and averages 20 fresh rows.
2. **White:** uncover the sensor and position a clean diffuse white reference under scan lighting, at the fruit surface position/distance. Press Enter. Avoid shiny paper/highlights; a characterized diffuse reflectance tile is better than ordinary white material.
3. **Fruit:** replace the white reference with the fruit at the same geometry. Mark one spot, keep it still, and press Enter. Repeat this same-spot scan three times when prompted, using the same dark/white references.

For each visible channel, Python computes:

```text
relative_reflectance = (mean_fruit - mean_dark) / (mean_white - mean_dark)
```

Saturation in any accepted row rejects the measurement. Each white-dark signal must exceed both **100 counts** and **10 × combined observed dark/white standard deviation**. These are bench quality guards, configurable via `--min-reference-counts` and `--reference-snr`, not established sensor accuracy thresholds. Do not lower them simply to force a prediction. An invalid scan is logged and prompts a repeat. If the lighting/gain must change, Ctrl-C and start again from dark. Board resets or streamed setting changes abort the calibration session automatically.

Normalized fruit values are retained without clipping; values outside the broad bench guard `[-0.02, 1.5]` trigger a repeat. A value above 1 can result from white-reference reflectance, surface geometry or highlights; inspect it rather than treating the reference as an absolute reflectance standard. The script does **not** normalize the eight values to sum to 1, rescale per scan, or refit a scaler.

The input DataFrame follows `bundle['feature_order']` exactly, mapping F1–F8 to the saved `F1_415nm_mean_reflectance` … `F8_680nm_mean_reflectance` names. It calls `bundle['models'][fruit]` directly: the trained StandardScaler and classifier are already in that pipeline. Output includes the experimental stage and scores in `model.classes_` order. Scores are uncalibrated model probabilities, not validated confidence or food-safety judgments. Apple prints “No apple ripeness model” and still saves measurements.

## 5. Repeatability and logs

The final table has F1–F8 rows and `scan_1`, `scan_2`, `scan_3`, mean, sample standard deviation, range and coefficient of variation (%). These describe variation of the **three averaged normalized scans**, not merely noise within one scan. CV is undefined near zero and saved blank. Start with the same clamped spot; a later re-positioning experiment is a separate check.

Every session creates `bench_logs/<UTC timestamp>/`:

- `serial.jsonl`: exact received lines, phase and host timestamp, including error/invalid rows;
- `measurements.jsonl`: raw mean/SD/min/max, normalized fruit scans (including failed QC attempts when computable), calibration settings, model hash, predictions, unsupported-fruit status and aborts;
- `repeatability.csv`: per-channel variation across completed fruit scans.

Dark/white scans log raw statistics; normalized fruit readings become computable after valid references exist. Logs and `.venv-bench` are excluded from Git. Ctrl-C retains partial logs.

## Limits and testing

The notebook trained on spatial mean spectra inside a fruit mask from referenced hyperspectral cameras, integrated with Gaussian approximations to AS7341 bands. A real AS7341 measures its physical channel response over a smaller field of view. Dark/white ratios approximate those features but do not remove differences in illumination spectra, optics, response curves, white-standard reflectance or spatial sampling. Accuracy on live hardware still needs paired measurements and independently labeled, held-out physical specimens; repeatability alone is not accuracy. Keep `over-ripe` separate from `spoiled`; the bundle has no spoiled class.

See [bench validation](../docs/as7341_bench_validation.md) for executed checks. Hardware dark/white/fruit placement requires you at the bench; no live calibration or accuracy is claimed from software tests.

References: [Adafruit library examples/API](https://github.com/adafruit/Adafruit_AS7341), [AS7341 datasheet](https://look.ams-osram.com/m/24266a3e584de4db/original/AS7341-DS000504.pdf) (integration timing and saturation registers), and the repo's `RipeSense_Detection_Spectral.ipynb` / `src/ripesense_deephs.py` for the trained feature definition.
