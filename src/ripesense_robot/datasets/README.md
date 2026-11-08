# Datasets

## Ours (committed here)

### `xarm_realsense/`: xArm wrist RealSense D435i crops (890 images, 8 Nov 2026)
Collected by the arm itself with `robot_xarm/runtime/collect_class.py` (`round.sh <class>`): one fruit type on the
table, the arm sweeps a dense grid of heights, tilts and orbit views, every depth-detected object is cropped.
File names start with the session time, which the training script uses for the time-based split.

| class | crops | class | crops |
|---|---|---|---|
| apple_fresh | 65 | mango_rotten | 50 |
| avocado_ripe | 85 | not_fruit (screwdriver, tape, case, socket, block) | 31 |
| avocado_rotten (60: 26 sticker up, 34 rolled) | 60 | orange_fresh | 160 |
| banana_unripe | 58 | orange_rotten | 33 |
| lemon_fresh | 152 | strawberry_rotten (the dark berries; first labelled fresh, corrected in v4) | 149 |
| mango_fresh | 47 | | |

### `ainex_robot/`: Hiwonder AiNex humanoid head-camera photos (1,190 images)
apple 162 · avocado 175 · banana 151 · mango 57 · not_fruit 53 · orange 180 · orange_rotten 36 · strawberry 376
(mapped in training to apple_fresh, avocado_ripe, banana_unripe, mango_fresh, not_fruit, orange_fresh,
orange_rotten, strawberry_fresh).

## Public (not committed: download into `datasets/public/`)

* `public/fruit_quality/`: [Fresh and Rotten Fruits](https://data.mendeley.com/datasets/bdd69gyhv8/1)
  (Sultana, Jahan, Uddin) + [FruitNet](https://data.mendeley.com/datasets/b6fftwbr2v/1) (Meshram, Patil),
  CC BY 4.0, arranged as `train|val|test/<class>/`; capped at 650 images per class in training.
* `public/banana_stages/`: [Annotated Banana Ripeness Stage Dataset v2](https://data.mendeley.com/datasets/56c75hvm7r/2)
  (Rohit et al.), CC BY 4.0, crops by ripeness stage.
