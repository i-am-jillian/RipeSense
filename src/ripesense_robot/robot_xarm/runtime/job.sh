#!/usr/bin/env bash
# usage: ~/job.sh <name> <command...>   One arm job at a time: frees the camera (stops the live viewer and any
# other arm script), runs the job with its output in /tmp/job.log, then restarts the live viewer.
exec 9>/tmp/job.lock
flock -n 9 || { echo "busy: $(cat /tmp/job.status 2>/dev/null)"; exit 1; }
name=$1; shift
pkill -f "^/home/sujal/ripesense_env/bin/python /home/sujal/(live_view|row_search|pick_all)\.py"; sleep 1.5
echo "running $name $(date +%s)" > /tmp/job.status
"$@" > /tmp/job.log 2>&1; rc=$?
if [ $rc -eq 0 ]; then st=done; else st=failed; fi
grep -q "STOPPED BY USER" /tmp/job.log && st=stopped
echo "$st $name $(date +%s) rc=$rc" > /tmp/job.status
(setsid nohup /home/sujal/ripesense_env/bin/python /home/sujal/live_view.py > /tmp/live_view.log 2>&1 < /dev/null 9>&- &)
