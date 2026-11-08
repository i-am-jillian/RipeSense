#!/usr/bin/env bash
# usage: ~/round.sh <label>   (one collection round; pauses the live viewer while the camera is busy)
pkill -f "^/home/sujal/ripesense_env/bin/python /home/sujal/live_view.py" 2>/dev/null; sleep 1
/home/sujal/ripesense_env/bin/python /home/sujal/collect_class.py "$1" "$2" 2>&1 | grep -v -E "^SDK_VERSION|ROBOT_IP|change protocol|^\[SDK\]|set_state|ControllerError had|motion_enable|ControllerWarning"
(setsid nohup /home/sujal/ripesense_env/bin/python /home/sujal/live_view.py > /tmp/live_view.log 2>&1 < /dev/null &)
