# xArm robot runtime: file map

## `runtime/` (copied to the Pi's home folder)

| File | Role |
|---|---|
| `demo.sh` | Mission: `full` = search + ripe->ripe bin + spoiled->spoiled bin; `pick` = picking phases only. Auto-recovery on any failed step. |
| `autopilot.sh` | Autonomous loop (search every 5 cycles, sort, wait 30 s). `FRONT=1` enables the forward views. Stops on `/tmp/autopilot.stop`. |
| `job.sh` | One arm job at a time (flock), frees the camera, writes `/tmp/job.log` + `/tmp/job.status`, restarts the live view. |
| `row_search.py` | Search: front views (bush height, ±30°) then table views; classify, merge across views, log to Tiger, live page on :8091. |
| `pick_all.py` | Detect + classify from one view and pick: close-up re-centre, reach check, gentle grip, slip check, bin drop from above. `--to_bin --target ripe|spoiled`. |
| `above.py` | Joint angles straight above a taught pose without a wrist flip (Newton steps on forward kinematics). |
| `recover.py` | Clear errors, drop a held fruit into the right bin, return to the start pose. |
| `scan_history.py` | Tiger Data (TimescaleDB) schema, logging, "inspect next" ranking, "what changed". |
| `manual.py` | Hand-guiding mode + saving drop poses (`ripe_bin`, `spoiled_bin`) to `~/drop_poses.json`. |
| `arm_status.py` / `arm_stop.py` | Status JSON for the dashboard / STOP button. |
| `live_view.py` / `classify_view.py` | Live camera page / one-shot classify from the current pose. |
| `collect_class.py` + `round.sh` | Training-data collection rounds (dense grid of views over one fruit class). |
| `handeye_calib.py` / `handeye_solve.py` | Eye-in-hand calibration with the ChArUco board (OpenCV 4 calibrateHandEye). |
| `touch_test.py` / `hover_test.py` / `pick_test.py` | Calibration checks (F/T touch-down, hover over board, single pick). |
| `iden_load.py` | Payload identification (1.98 kg end effector; the default 0.82 kg caused false collision stops). |
| `rail_tool.py` / `rail_test.py` / `rail_test2.py` | Linear rail status / watchdog tests (halts on any uncommanded motion). |
| `config.yaml` | Camera intrinsics, hand-eye result `eef_to_color`, table height, gripper length. |

## `dashboard/` (laptop)
`dashboard.py` (Flask, port 5050) talks to the Pi over ssh and to Tiger directly; `dashboard.html` is the main page
(camera, missions, sorting, positions), `dashboard_classic.html` the detailed view (charts, history, Tiger stats).
