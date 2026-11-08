# AS7341 local bench validation — 2026-11-08

## Executed

- Identified the attached USB device as **UNO WiFi R4 CMSIS-DAP**, VID:PID `2341:1002`, serial `F412FA6378E0`, port `/dev/cu.usbmodemF412FA6378E02`. This is the R4 WiFi, not the R3 in the supplied wiring-table heading.
- Inspected `deephs_ripeness.joblib` and the training notebook/helper. The bundle has five models (avocado, kiwi, persimmon, papaya, mango), no apple model, and eight F1–F8 named mean-reflectance inputs. Its pipelines contain the fitted StandardScaler and LogisticRegression.
- Found that the supplied model was serialized with scikit-learn **1.6.1**. Created `.venv-bench` with Python 3.13 and `bench/requirements.txt`. `--check-model` loads the bundle without estimator-version warnings and checks every pipeline's input names/count against the bundle schema.
- Invoked each of the five existing fitted pipelines with an explicitly synthetic eight-feature vector to check executable compatibility and class-score ordering. This was not a sensor measurement or accuracy test. The unsupported-apple branch also passed.
- Compiled `arduino/RipeSense_AS7341/RipeSense_AS7341.ino` for `arduino:renesas_uno:unor4wifi`, Renesas UNO core **1.6.0**, with installed Adafruit AS7341 **1.4.1** and Adafruit BusIO. Result: **67,016 bytes flash**, **9,280 bytes RAM**; successful compilation. Toolchain/build files were isolated in `/tmp`.
- Ran seven automated bench tests:
  1. CSV channel ordering, headers and read/malformed-row errors;
  2. digital/flagged saturation, short-integration limits, weak/noisy references and setting changes;
  3. saved feature-order mapping, class-score mapping and unsupported apple behavior;
  4. invalid-scan retry with failed raw/normalized statistics retained;
  5. stale-reading discard and board-reset detection;
  6. complete simulated apple dark/white/three-fruit session, raw serial logs and normalized logs;
  7. per-channel repeatability SD/CV calculation.

Run tests from the repo root:

```sh
.venv-bench/bin/python -m unittest discover -s tests -p 'test_as7341_bench.py' -v
```

The simulated apple session used normalized scan values 0.4, 0.5, 0.6, confirming a sample SD of 0.1 and CV of 20% in each channel. Those numbers are test fixtures, **not measurements from the attached sensor**.

## Not executed

- Firmware upload: the existing board program was not replaced.
- Live I²C acquisition, hardware saturation/read-error injection, dark-reference coverage, white-reference quality, fruit placement, or actual three-scan repeatability.
- Live ripeness accuracy or AS7341-to-hyperspectral transfer calibration.

Use [bench/README.md](../bench/README.md) to upload, close Serial Monitor, and run the guided physical measurements. Python retains raw lines, averaged readings, relative-reflectance values, model identity/settings and predictions in ignored `bench_logs/` directories. The same path logs apple measurements without claiming an apple ripeness result. Existing training notebooks and models were not modified.

## Follow-up: serial stall resolved (2026-11-08)

All eight attempted bench sessions had zero received serial lines; the active Arduino IDE showed a blank sketch with ESP32 examples selected. Uploaded the previously compiled UNO R4 WiFi bench firmware successfully. A short USB serial check then received real `DATA` rows containing F1–F8 counts and `saturated=0`. This verifies live streaming, not dark/white calibration or fruit accuracy. The diagnostic serial port was closed afterward. Added collection progress (1/20, 5/20, etc.) and a zero-reading upload/port hint to Python; all seven bench tests still pass. The earlier “firmware upload not executed” entry describes the initial validation, before this follow-up.
