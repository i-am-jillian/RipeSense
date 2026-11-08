#!/usr/bin/env python3
# encoding: utf-8
"""Two-hand grab / release for the AiNex, sized for fruit.

Same keyframes as Hiwonder's 'move_up' / 'put_down' action groups (both arms go
down beside the object, then the shoulder-roll servos swing the arms inward to
squeeze it between the two hands), except the inward squeeze is a parameter.
'move_up' squeezes to 910/90 for a ~4 cm block; a ~7-8 cm apple stops the arms
much earlier, so commanding 910/90 would leave both servos stalled and pushing.

squeeze = how far the arms swing inward from the open 800/200 pose, in servo
units (0..110). 110 = Hiwonder's block grip. Start low (~40) and raise until the
fruit is held when the robot stands up.

  python3 two_hand_grab.py grab 50     # crouch, grab, stand, lift
  python3 two_hand_grab.py release     # put it down again (Hiwonder put_down frames)

Calibration (robot holds each pose so you can look at the hands next to the fruit):
  python3 two_hand_grab.py reach        # crouch, arms down and fully open, HOLD
  python3 two_hand_grab.py squeeze 40   # (while crouched) close both arms to this squeeze, HOLD
  python3 two_hand_grab.py stand        # back to walk_ready
"""
import sys

import rospy

# servo ids: 13/14 shoulder pitch, 15/16 shoulder roll (squeeze), 17/18 elbow pitch,
# 19/20 elbow yaw, 21/22 grippers, 1-12 legs. Frames are [ms, positions of servos 1..22].
_CROUCH_LEGS = [500, 500, 817, 206, 780, 220, 270, 730, 500, 500, 500, 500]
_STAND_LEGS = [501, 500, 667, 330, 571, 434, 327, 675, 498, 500, 500, 501]


# Pick pose found by testing on the robot with a ~7.5 cm apple (2026-11-07):
#   lean 40 at the hips (servos 7/8), arms forward to 500 (13/14) so the hands clear the knees,
#   shoulder-roll squeeze 110 (15/16) and elbows curled in 650/350 (17/18) so both hands wrap the fruit.
LEAN, ARMS, ELBOW = 40, 500, 150


def _frames_grab(squeeze, lean=LEAN, arms=ARMS, elbow=ELBOW):
    l_in, r_in = 800 + squeeze, 200 - squeeze          # move_up uses 910 / 90 (squeeze 110)
    crouch = list(_CROUCH_LEGS)
    crouch[6], crouch[7] = 270 - lean, 730 + lean       # hip pitch: lean forward
    return [
        (500, [500, 501, 670, 329, 567, 436, 283, 717, 498, 501, 499, 501, 873, 125, 830, 168, 500, 498, 70, 928, 498, 500]),
        (1500, crouch + [arms, 1000 - arms, 800, 200, 500, 500, 490, 510, 500, 500]),        # crouch, lean, arms forward & open
        (300, crouch + [arms, 1000 - arms, 800, 200, 500, 500, 490, 510, 500, 500]),
        (1200, crouch + [arms, 1000 - arms, l_in, r_in, 500 + elbow, 500 - elbow, 490, 510, 480, 520]),  # wrap with both hands
        (400, crouch + [arms, 1000 - arms, l_in, r_in, 500 + elbow, 500 - elbow, 490, 510, 480, 520]),
        (1200, _STAND_LEGS + [arms, 1000 - arms, l_in, r_in, 500 + elbow, 500 - elbow, 490, 510, 480, 520]),  # stand up holding it
        (1200, _STAND_LEGS + [300, 700, l_in, r_in, 500 + elbow, 500 - elbow, 490, 510, 480, 520]),  # raise a little to carry
    ]


def _frames_release(squeeze):
    l_in, r_in = 800 + squeeze, 200 - squeeze
    return [
        (500, _STAND_LEGS + [150, 850, l_in + 10, r_in - 10, 500, 500, 150, 850, 480, 520]),
        (2000, _STAND_LEGS + [520, 480, l_in, r_in, 500, 500, 490, 510, 480, 520]),     # arms down in front
        (1000, [500, 500, 780, 220, 720, 280, 300, 700, 500, 500, 500, 500, 550, 450, l_in, r_in, 500, 500, 490, 510, 480, 520]),
        (500, [500, 500, 780, 220, 720, 280, 300, 700, 500, 500, 500, 500, 550, 450, 800, 200, 500, 500, 490, 510, 500, 500]),  # open
        (1000, [500, 500, 640, 360, 500, 500, 330, 670, 500, 500, 500, 500, 835, 165, 830, 170, 500, 500, 70, 930, 500, 500]),  # stand
    ]


def play(motion_manager, frames):
    for ms, pos in frames:
        motion_manager.set_servos_position(ms, [[i + 1, int(p)] for i, p in enumerate(pos)])
        rospy.sleep(ms / 1000.0 + 0.05)


def grab(motion_manager, squeeze=110):
    play(motion_manager, _frames_grab(max(0, min(120, int(squeeze)))))


def release(motion_manager, squeeze=50):
    play(motion_manager, _frames_release(max(0, min(110, int(squeeze)))))


def reach(motion_manager):
    play(motion_manager, _frames_grab(0)[:3])          # crouched, arms down, open (squeeze 0)


def squeeze_to(motion_manager, squeeze, ms=600):
    squeeze = max(0, min(110, int(squeeze)))
    motion_manager.set_servos_position(ms, [[15, 800 + squeeze], [16, 200 - squeeze]])
    rospy.sleep(ms / 1000.0 + 0.05)


if __name__ == '__main__':
    rospy.init_node('two_hand_grab', anonymous=True)
    from ainex_kinematics.motion_manager import MotionManager
    mm = MotionManager()
    rospy.sleep(0.5)
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'grab'
    sq = int(sys.argv[2]) if len(sys.argv) > 2 else 110
    if cmd == 'reach':
        reach(mm)
    elif cmd == 'squeeze':
        squeeze_to(mm, sq)
    elif cmd == 'stand':
        mm.run_action('walk_ready')
    else:
        (grab if cmd == 'grab' else release)(mm, sq)
    print('%s done (squeeze %d)' % (cmd, sq))
