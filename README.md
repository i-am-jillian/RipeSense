# RipeSense

**xArm 6 fruit-sorting robot (end-to-end):** code, deployed model, our datasets, training scripts and logs are in
[`src/ripesense_robot/`](src/ripesense_robot/README.md): search (front, then table), classify, gentle pick, ripe and
spoiled bins, Tiger Data scan history, dashboard and autopilot.

**Local AS7341 bench test:** see [setup, Arduino upload and VS Code run commands](bench/README.md). Includes guided dark/white calibration, three same-spot scans, model predictions, and apple sensor-only logging—no Colab or camera.

Open [`RipeSense_Detection_Spectral.ipynb`](RipeSense_Detection_Spectral.ipynb) in Google Colab with **File → Upload notebook**, or its GitHub tab after pushing the repo. The notebook runs without local training data or a sensor.

## Colab run order

1. **YOLO (optional for spectral work):** run the original install/setup and public-image detection cells. Numbered boxes, RGB crops and CSV are saved to Drive. Optional COCO fine-tuning retains both baseline and fine-tuned checkpoints. These working YOLO code cells are unchanged.
2. **DeepHS starter:** run the new DeepHS section's install/setup, helper, inventory/plan, batch, training/evaluation and **Try one recording** cells. It can run independently of YOLO and Mendeley. Defaults download nine public mango recordings (~22 MB), train on six independent specimens and test on three. Smoke scores demonstrate execution, not model quality.
3. **All five fruits:** change `DEEPHS_MODE='full'`, rerun setup and inventory/plan, then rerun the batch cell until `remaining == 0`. This covers avocado, kiwi, persimmon, papaya and mango, with 596 labeled VIS recordings. Defaults process 20 recordings per batch. HTTP range requests fetch individual members, never whole archives; completed feature rows are skipped on restart. Full extraction transfers ~23 GB over time, but holds only one recording at once. Failed records are logged and retried; training requires the complete plan.
4. **Evaluate/try:** run training/evaluation, then choose `TRY_FRUIT` and `TRY_INDEX` in **Try one recording**. It shows a held-out preview, prediction and dataset label. Results include per-fruit specimen-level scores and stage-support warnings. Papaya's over-ripe stage has only one test specimen in the full split, so its stage recall is withheld.
5. **Mendeley (optional):** the original ripeness archive audit remains below DeepHS. Its missing wavelength/stage/specimen metadata blocks only the Mendeley classifier, not DeepHS training.

DeepHS features/previews persist under `MyDrive/RipeSense_Detection_Spectral/DeepHS/features_<signature>/`. Models, feature order, provenance and locked plans are under `results_<signature>/<mode>/`. Use the same settings to resume; smoke and full results stay separate. For a larger five-fruit smoke check, set `DEEPHS_SMOKE_FRUITS` to all five names (~1.66 GB for 45 recordings).

The eight features approximate AS7341 visible bands from referenced hyperspectral cubes. **Real AS7341 counts still require paired calibration or sensor-specific retraining and independent testing.** `over-ripe` and `spoiled` remain distinct; neither spectral source supplies a validated spoiled classifier. Reserve all three physical apples for the live demo.

Sujal's [`src/RipeSense_Train.ipynb`](src/RipeSense_Train.ipynb) is unchanged. Its separately trained ONNX model can later consume YOLO RGB crops with its own preprocessing and labels.

See [DeepHS tests and limitations](docs/deephs_validation.md), [original detection/Mendeley validation](docs/detection_spectral_validation.md), and [Mendeley audit](docs/mendeley_ripeness_audit.json). No large data or model binaries are committed. The DeepHS helper is embedded in the notebook and also available as [`src/ripesense_deephs.py`](src/ripesense_deephs.py).
