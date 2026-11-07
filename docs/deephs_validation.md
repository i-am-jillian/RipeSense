# DeepHS implementation and validation — 2026-11-07

The notebook now contains an independently runnable DeepHS section. Its helper is also saved as `src/ripesense_deephs.py` and embedded verbatim in the notebook, so uploading the notebook alone to Colab works. All pre-existing YOLO code cells are unchanged (compared with the repository HEAD).

## Public sources actually inspected

- Official repository at commit `5f47812a3fb5bd8f0ed5367cedf9b464f83076c1`: naming convention, measurement series, annotation export/label merging, wavelength vectors, background masking and already-referenced cube loading.
- Corrected `annotations-upd-2024-01-09.zip` from the server linked by that repository. The three v2 JSON files are joined to their own annotations before concatenation; numeric record IDs restart per file.
- Central directories of all five public archives via HTTP ranges, plus ENVI headers. Every selected labeled VIS recording and header exists: 596 recordings total. Raw ENVI data are BIP float32 with 224 channels. Inspected headers omit wavelengths; the matching annotation camera definition supplies them (397.66–1003.81 nm).
- The complete selected data would transfer approximately 22.94 GB compressed across batches. Largest selected uncompressed member: 162.1 MB. No full fruit archive was downloaded.

## Independent physical specimen inventory

IDs include fruit, measurement series and fruit number; side/day are not independent specimens. Coarse `perfect` is displayed as `ripe`; original coarse/fine labels remain recorded. All labeled views of each specimen agree on stage.

| Fruit | Unripe | Ripe | Over-ripe | Total |
|---|---:|---:|---:|---:|
| Avocado | 18 | 48 | 27 | 93 |
| Kiwi | 29 | 40 | 17 | 86 |
| Persimmon (Kaki) | 5 | 25 | 10 | 40 |
| Papaya | 5 | 21 | 4 | 30 |
| Mango | 13 | 16 | 11 | 40 |
| Total | 70 | 150 | 69 | 289 |

The fixed full-plan split holds out 5/12/7 avocado, 8/10/5 kiwi, 2/7/3 persimmon, 2/6/1 papaya, and 4/4/3 mango specimens (unripe/ripe/over-ripe). Papaya over-ripe has only one held-out specimen, so its stage recall is withheld as insufficient. All scores on such small groups need caution. This is a custom grouped split, not a reproduction of the official benchmark partitions.

## Executed on public recordings

- Downloaded nine labeled mango recordings (~22.1 MB compressed), extracted eight features, trained on six specimens and tested on three. Executed the notebook's actual training/report and “Try one recording” cells locally, with local paths replacing Drive paths. Reloaded the saved model/feature order successfully.
- This smoke run predicted 2 of 3 held-out mango specimens correctly. It is a pipeline check, **not a credible accuracy estimate**. Each stage has only one test specimen, and the report marks every stage too small for stage recall. The default try-one example was an incorrect over-ripe → unripe prediction; the UI exposes that mismatch instead of selecting a successful example.
- Downloaded and extracted an additional labeled recording for each of avocado, kiwi, papaya and persimmon. Inspected pseudo-RGB previews and fruit-mask overlays for all five fruits.
- Initial source-style masking at mean reflectance 0.2 selected only a small patch of dark avocado skin. Lowered the fixed threshold to 0.05, retained the largest connected component, and eroded one pixel. The final inspected masks follow the visible fruit regions. This visual preprocessing check is not quantitative segmentation validation; all-fruit mask QA remains necessary on a larger sample.
- Reran extraction on the completed mango plan: zero pending recordings, zero re-downloads. Atomic per-recording JSON commits and cached PNG previews support restarting batches; failed recordings remain pending. Training rejects an incomplete plan rather than reporting on an accidental convenience subset.
- Verified all full-plan specimen assignments are disjoint between train and test; feature caches can be reused across modes, but split assignments are always replaced by the current locked plan.

## Offline checks

Run `python -m unittest discover -s tests -v` with the notebook dependencies installed. Tests cover:

- repeated days/views staying together and deterministic plans under input reordering;
- ENVI decoding, truncated-data rejection, feature order/weights, constant-spectrum extraction and empty-mask rejection;
- five-fruit training/report serialization using clearly synthetic features, specimen-level aggregation, and rejection of incomplete batches;
- exact agreement between the embedded helper and Python module.

Notebook schema validation and Python-cell syntax parsing also passed. Existing YOLO code-cell contents match HEAD exactly. The `%pip` cell is Colab syntax, excluded from plain Python parsing.

## Not executed or established

- Interactive Google Colab/Drive authorization, a complete 596-recording extraction, or full five-fruit real-data training/evaluation. The notebook implements those paths; no full-data results are claimed here.
- Quantitative ground-truth mask validation, a calibrated confidence estimate, or robustness to acquisition/cultivar changes.
- Live AS7341 transfer. Gaussian filters use datasheet centers/FWHM, not measured response curves; F1's lower tail is truncated by the camera. Raw AS7341 counts must not be passed directly to these referenced-camera models. Paired sensor calibration/retraining and held-out physical-specimen testing are required.

No raw cubes, archives, preview images or trained binaries are added to Git. The notebook saves feature caches, previews, locked splits, source hashes, models and reports to Drive. There is no spoiled class, and over-ripe is never relabeled spoiled.

Sources: [official DeepHS repo](https://github.com/cogsys-tuebingen/deephs_fruit), [public data/annotations](https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/), [AS7341 datasheet](https://look.ams-osram.com/m/24266a3e584de4db/original/AS7341-DS000504.pdf), Figure 7.
