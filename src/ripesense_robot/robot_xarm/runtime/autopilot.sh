#!/usr/bin/env bash
# AUTOPILOT: runs until /tmp/autopilot.stop exists (dashboard "Autopilot OFF") or STOP kills it.
#   every FULL_EVERY cycles: row search (front views only if FRONT=1, then table) -> sorting page + Tiger
#   every cycle: look at the 4 table views, put every ripe fruit in the bin, then wait WAIT_S seconds
# Any failed step ends the autopilot (no blind retries with a fruit possibly in the gripper).
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
FRONT=${FRONT:-0}; WAIT_S=${WAIT_S:-30}; FULL_EVERY=${FULL_EVERY:-5}
rm -f /tmp/autopilot.stop
VIEWS=("263.7 36.7 459" "300 -160 459" "300 200 459" "430 -60 459")
cycle=0
stop_req() { [ -f /tmp/autopilot.stop ]; }
while ! stop_req; do
  cycle=$((cycle+1))
  if [ $(( (cycle-1) % FULL_EVERY )) -eq 0 ]; then
    if [ "$FRONT" = 1 ]; then
      echo "=== AUTO cycle $cycle: search (front first, then table)"
      $PY /home/sujal/row_search.py --exit && FAILS=0 || recover "search failed"
    else
      echo "=== AUTO cycle $cycle: search (table)"
      $PY /home/sujal/row_search.py --exit --floor_only && FAILS=0 || recover "search failed"
    fi
  fi
  k=0
  for look in "${VIEWS[@]}"; do
    stop_req && break
    k=$((k+1)); echo "=== AUTO cycle $cycle: picking from view $k of 4"
    $PY /home/sujal/pick_all.py --go --to_bin --look $look && FAILS=0 || recover "pick failed in view $k"
  done
  if grep -q spoiled_bin /home/sujal/drop_poses.json 2>/dev/null; then
    echo "=== SPOILED: spoiled fruit -> spoiled bin"
    k=0
    for look in "263.7 36.7 459" "300 -160 459" "300 200 459" "430 -60 459"; do
      stop_req && break
      k=$((k+1)); echo "=== SPOILED picking from view $k of 4"
      $PY /home/sujal/pick_all.py --go --to_bin --target spoiled --look $look && FAILS=0 || recover "spoiled pick failed in view $k" spoiled_bin
    done
  fi
  stop_req && break
  echo "=== AUTO cycle $cycle: waiting ${WAIT_S}s for new fruit"
  for ((s=0; s<WAIT_S; s++)); do stop_req && break; sleep 1; done
done
rm -f /tmp/autopilot.stop
echo "=== AUTOPILOT OFF after $cycle cycles"
