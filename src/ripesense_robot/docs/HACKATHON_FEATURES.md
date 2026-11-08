# RipeSense: the features that go past "is it ripe?"

> **Pitch line:** "Other apps tell you a banana is ripe. RipeSense tells you when
> it will be at its best, shows bruises your eyes can't see yet, learns a new fruit
> in 30 seconds, and tells the store exactly what to mark down so nothing gets
> thrown away."

Everything below runs on the laptop now: open the camera page, or the mock cabinet on
`/dev` + `/camera`. It switches to the real sensors on the Pi without code changes.

| # | Feature | Where it lives | How to demo it (≈20 s each) |
|---|---|---|---|
| 1 | **Ripening forecast** (Bayesian per-fruit model, temperature-aware: rate ×2.3 per +10 °C) | `ripesense/forecast.py` | Scan a fruit with a **fruit id**. The card says "Best tomorrow evening". Tap *Fridge*, *Warm spot* or *Bag + banana*: the date moves. Scan the same fruit again later and the shaded band narrows. |
| 2 | **Hidden-bruise "X-ray"** (850 nm IR, projector off) | `features.ir_bruises`, `hw/camera.py` `_ir()` | Press a thumb hard into a banana, then scan. The IR card shows the bruise in red before the skin browns. RealSense only; on the laptop, use `/dev` → hidden bruises 1/2. |
| 3 | **Teach a new fruit in 30 s** (few-shot, no retraining) | `models/fewshot.py` | Hand it a fruit it doesn't know (kiwi, pear). Type `kiwi` and press *Teach 5 views* while turning it. Then scan: it says kiwi. `kiwi:ripe` teaches a stage too. |
| 4 | **"Why" heatmap** (RISE saliency over the model's own crop) | `models/explain.py` | Every result shows where the model looked: brown spots, mould, the green tip. |
| 5 | **Guaranteed days-left range** (conformal prediction) | `train_fusion.conformal`, `infer.py` | Once the fusion model is trained on cabinet scans, the range on the kiosk gets a ✓, meaning it contains the true value 90% of the time on new fruit. |
| 6 | **Markdown optimizer** (expected-value pricing + donate) | `pricing.py`, `stages.store_info` | Store mode / store card. Ripe with 3 days left → 10% off; 1.5 days → 40%; under a day → "Donate today". The `/store` page totals $ recovered, kg of waste avoided and kg CO₂e. |
| 7 | **Ethylene advisor** | `fruit_info.ethylene_advice` | Fruit card: "Keep away from avocado, kiwi, pear: its ethylene will ripen them faster." |
| 8 | **"The cabinet learned during the event"** | `/api/label_queue`, `train_history.json`, `/store` | The `/collect` page lists the scans the model was least sure about, so you label those first. Each *Train* adds a point to the accuracy chart on `/store`. |
| 9 | **Smell fingerprint** (BME688 heater profile, 200 → 400 °C) | `hw/gas.py`, `features.enose_features` | The 5 hotplate temperatures respond to different gases (fruity esters vs fermentation ethanol). The fusion model uses the 5-channel shape, not one number. Needs a BME688/680 rather than an MQ sensor. |
| 10 | **Pull the cable** | `/api/unplug`, `scan.py` | Untick *gas* (or really unplug it) and scan. The verdict still comes from the other sensors, and its dot goes dashed. |
| 11 | **"Your banana in N days"** | `future.py`, `/api/future/<id>.jpg` | Drag the slider: the photo turns yellow, then spotted, then brown, following the forecast. Labelled *simulated*. |
| 12 | **Recipe for tonight** | `cloud/gemini.recipe` | Fruit card → *Use it tonight*. Uses Gemini with `GEMINI_API_KEY`, otherwise built-in recipes. |
| 13 | **Fruit profile card** | `fruit_info.py` | Weight from the 3D volume, sweetness (°Brix), starch→sugar, kcal / sugar / fibre / vitamin C / potassium for *this* fruit, best uses, storage. |

## Demo order (3 minutes)
1. Scan a **green banana** and point at the forecast: "best Saturday evening". Tap *Fridge*.
2. Slide **"in 4 days"** to show the simulated future banana.
3. Scan a **spotty banana**: Store mode → optimal price, then the `/store` totals.
4. Scan a **bruised** banana: IR card. (This needs the RealSense.)
5. Ask a judge for any fruit, then **teach it** live and scan it.
6. **Unplug** the gas sensor, rescan: it still works, with the dot dashed.
7. Close on the `/store` learning chart and the "why" heatmap.

## Measured numbers (be precise in the pitch)
- **Teach-a-fruit:** we hid one fruit from the reference set and taught it with 5 photos. Unseen photos of it were recognised 80–100% of the time (strawberry 100%, lime 93%, guava 80%, pomegranate 78%). 0 of 238 other-fruit photos were wrongly accepted. Caveat: the network had seen these fruits during training, so a truly new fruit will score lower. Re-run with `python -m ripesense.models.fewshot --selftest --holdout lime`.
- **Conformal ranges:** on simulated cabinet data, the raw p10–p90 ranges contained the true value 65% of the time. After calibration (±0.21 d) they contained it 91% of the time, against a 90% target. Real numbers come from your own scans: `train_fusion` prints them.
- **Forecast:** simulated fruit with a known ripening rate. One scan gives a spoil date of 4.5–11.7 days; six scans narrow it to 2.3–4.9 days around the true 3.0 days.
- **Laptop camera, real photos:** unripe, ripe and overripe bananas are read correctly at about 90% confidence. The forecast, profile and heatmap all come out of the same scan.

## Honest limits (say them before a judge does)
- **Forecast:** the default ripening rates and Q10 = 2.3 come from postharvest literature, not our measurements. The band narrows with *your* scans of the same fruit id.
- **Pricing:** the demand model (buy rate, elasticity 3.0, donation value) is an assumption in `config.yaml`. A store would fit it from its own sales.
- **IR bruises:** tested on simulated IR frames so far. Check on a real pressed banana with the RealSense before claiming it on stage.
- **E-nose:** needs `adafruit-circuitpython-bme680` with `set_gas_heater` (recent versions). It falls back to a single reading if missing.
- **Nutrition and Brix:** USDA tables scaled by estimated weight and stage. They are estimates, not measurements, and are labelled as such on screen.
- **Future preview:** a colour and spot simulation, not a generative model.
