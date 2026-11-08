#!/usr/bin/env bash
# Missions.  demo.sh full : row search (front first, then table), then every ripe fruit -> bin (3 table views)
#            demo.sh pick : every ripe fruit -> bin from the 3 table views (no search)
# HOLD = comma list of fruit kinds never put in the bin (empty = trust the model).
PY=/home/sujal/ripesense_env/bin/python

# a failed step -> automatic recovery (clear error, finish dropping a held fruit, back to start) and carry on;
# 3 failures in a row -> stop for real
FAILS=0
recover() {
  echo "=== problem: $1 -> recovering automatically"
  grep -q "STOPPED BY USER" /tmp/job.log && { echo "=== stopped by user"; exit 1; }
  FAILS=$((FAILS+1))
  [ $FAILS -ge 3 ] && { echo "=== MISSION FAILED: 3 problems in a row ($1)"; exit 1; }
  $PY /home/sujal/recover.py $2 || { echo "=== MISSION FAILED: recovery did not work ($1)"; exit 1; }
}
MODE=${1:-full}
HOLD=${HOLD-}
if [ "$MODE" = full ]; then
  echo "=== STEP 1/2: search (front first, then table)"
  $PY /home/sujal/row_search.py --exit && FAILS=0 || recover "search failed"
  echo "=== STEP 2/2: ripe fruit -> bin"
else
  echo "=== STEP 1/1: ripe fruit -> bin"
fi
k=0
for look in "263.7 36.7 459" "300 -160 459" "300 200 459" "430 -60 459"; do
  k=$((k+1)); echo "=== picking from view $k of 4"
  $PY /home/sujal/pick_all.py --go --to_bin --hold "$HOLD" --look $look && FAILS=0 || recover "pick failed in view $k"
done
if grep -q spoiled_bin /home/sujal/drop_poses.json 2>/dev/null; then
  echo "=== SPOILED: spoiled fruit -> spoiled bin"
  k=0
  for look in "263.7 36.7 459" "300 -160 459" "300 200 459" "430 -60 459"; do
    true
    k=$((k+1)); echo "=== SPOILED picking from view $k of 4"
    $PY /home/sujal/pick_all.py --go --to_bin --target spoiled --look $look && FAILS=0 || recover "spoiled pick failed in view $k" spoiled_bin
  done
fi
echo "=== MISSION COMPLETE"
