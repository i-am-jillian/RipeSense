#!/usr/bin/env bash
# Usage on the robot:  ~/ros_ws/src/ainex_fruit_pick/run.sh detect|pick|stop [roslaunch args...]
#   detect  perception only (robot does not move); view at laptop http://localhost:5001/robot
#   pick    walk to ripe fruit, pick it, carry to basket tag  (robot WALKS)
#   collect <class>  save training crops of every blob as <class> (ripe|unripe|overripe|rotten|not_banana)
#                    into ~/ripesense_collect/<class>/ ; robot stays still. Show ONLY that kind of thing.
#   action <name>    run one Hiwonder action group (e.g. move_up = pick, put_down, walk_ready) with the robot standing
#   grab [squeeze] / release [squeeze]   two-hand fruit grab test with the robot standing (default squeeze 50)
#   stop    stop any of them
source /opt/ros/noetic/setup.bash
source /home/ubuntu/ros_ws/devel/setup.bash
PIDFILE=/tmp/ainex_fruit_pick.pid
stop() {
  if [ -f $PIDFILE ]; then kill -INT "$(cat $PIDFILE)" 2>/dev/null; sleep 4; kill "$(cat $PIDFILE)" 2>/dev/null; rm -f $PIDFILE; fi
  rosnode kill /fruit_pick /fruit_detection /apriltag_ros_continuous_node >/dev/null 2>&1 || true
}
case "$1" in
  detect|pick)
    mode=$1; shift; stop
    setsid nohup roslaunch ainex_fruit_pick fruit_$mode.launch "$@" > /tmp/fruit_$mode.log 2>&1 < /dev/null &
    echo $! > $PIDFILE; echo "started fruit_$mode (log /tmp/fruit_$mode.log)";;
  collect)
    [ -n "$2" ] || { echo "usage: $0 collect ripe|unripe|overripe|rotten|not_banana"; exit 1; }
    label=$2; shift 2; stop
    setsid nohup roslaunch ainex_fruit_pick fruit_detect.launch collect_label:=$label "$@" > /tmp/fruit_collect.log 2>&1 < /dev/null &
    echo $! > $PIDFILE; echo "collecting '$label' crops into ~/ripesense_collect/$label (run '$0 stop' when done)";;
  action)
    [ -n "$2" ] || { echo "usage: $0 action move_up|put_down|walk_ready|stand"; exit 1; }
    python3 -c "
import rospy
rospy.init_node('run_action', anonymous=True)
from ainex_kinematics.motion_manager import MotionManager
m = MotionManager(); rospy.sleep(0.5); m.run_action('$2'); print('done: $2')
";;
  grab|release)
    python3 /home/ubuntu/ros_ws/src/ainex_fruit_pick/scripts/two_hand_grab.py $1 ${2:-50};;
  stop) stop; echo stopped;;
  *) echo "usage: $0 detect|pick|collect <class>|action <name>|stop [args]"; exit 1;;
esac
