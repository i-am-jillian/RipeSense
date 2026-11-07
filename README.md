# RipeSense

Open [`RipeSense_Detection_Spectral.ipynb`](RipeSense_Detection_Spectral.ipynb) in Google Colab using **File → Upload notebook** (or Colab's GitHub tab after this repo is pushed).

1. Run install and setup; authorize Google Drive. CPU works for the default run; select a T4 GPU for fine-tuning.
2. Run the YOLO11n public-image demo. Numbered boxes, RGB crops and `detections.csv` are saved to Drive for Sujal's classifier.
3. Optionally set `RUN_FINETUNE=True` and run the COCO subset and training cells. Defaults cap selection at 120 images per class for training and 40 per class for validation (at most 480 images total), plus a ~252 MB annotation ZIP. Both model checkpoints and their comparison are retained.
4. Run the Mendeley inspection cells. They fetch only the ripeness archive directory and sample images using range requests; the fallback downloads the 1.56 GB ripeness ZIP. Review the inventory, panel contact sheet and missing-mapping report.
5. Read the demo handoff. All results/models go to `MyDrive/RipeSense_Detection_Spectral/<run ID>/`; large caches stay in temporary `/content` storage.

**Spectral training is blocked by the public metadata, not by missing hardware.** The inspected archive contains 3,231 concatenated JPEGs and 20 candidate variety/apple-number identities. It does not supply verified panel wavelengths, stage assignments, or confirmation of identity across days/views. Physical-apple counts per stage are therefore unknown. The notebook reports this before any accuracy and includes a conditional grouped-training path for future author-verified metadata and reviewed fruit masks. No wavelengths or ripeness labels are guessed from filenames/days.

`over-ripe` and `spoiled` remain distinct. This dataset does not provide a spoiled class. Keep all three physical apples, including the rotten one, for the live demo. Phone/gantry RGB images can use the detector immediately; AS7341 counts require instrument-specific calibration and independent validation before any spectral-model use.

Sujal's original [`src/RipeSense_Train.ipynb`](src/RipeSense_Train.ipynb) is unchanged. Its trained classifier is not included; apply its exact preprocessing and labels to the exported crops once available.

See [actual validation and limitations](docs/detection_spectral_validation.md) and the [recorded Mendeley audit](docs/mendeley_ripeness_audit.json). No training data, weights or invented accuracy are committed.
