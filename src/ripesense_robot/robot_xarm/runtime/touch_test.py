#!/usr/bin/env python3
"""Measure the true table height with the force-torque sensor and fix the camera height (z) in the
hand-eye calibration.  Look (tool vertical) -> board centre corner from the camera -> closed fingertips
40 mm above it -> zero F/T -> descend at 5 mm/s until contact force > threshold -> back up.

  ~/ripesense_env/bin/python touch_test.py --x_vec <6 numbers> [--square_mm 23.3]
Prints the corrected eef_to_color z.
"""
import argparse, math, os, sys, time
import cv2, numpy as np, pyrealsense2 as rs, yaml
from xarm.wrapper import XArmAPI

ap = argparse.ArgumentParser()
ap.add_argument('--x_vec', type=float, nargs=6, required=True, help='camera->flange xyz m + rpy rad to test')
ap.add_argument('--square_mm', type=float, default=23.3)
ap.add_argument('--gripper_mm', type=float, default=230)
ap.add_argument('--look', type=float, nargs=3, default=[263.7, 36.7, 313.5])
ap.add_argument('--force_n', type=float, default=2.5)
ap.add_argument('--max_down_mm', type=float, default=100)
args = ap.parse_args()


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.ravel(t); return M


X = T(rpy_to_R(*args.x_vec[3:]), args.x_vec[:3])
arm = XArmAPI('192.168.1.232'); time.sleep(0.3)
arm.clean_warn(); arm.clean_error(); arm.motion_enable(True); arm.set_mode(0); arm.set_state(0); time.sleep(0.5)
arm.set_collision_sensitivity(3)
code, cur = arm.get_position(is_radian=True); yaw = cur[5]
DOWN = (math.pi, 0.0)


def go(x, y, z, speed, wait=True):
    ret = arm.set_position(x=x, y=y, z=z, roll=DOWN[0], pitch=DOWN[1], yaw=yaw, speed=speed, mvacc=200, is_radian=True, wait=wait)
    print('  move to flange (%.0f, %.0f, %.0f) @ %.0f mm/s -> code %s, state %s, error %s' % (x, y, z, speed, ret, arm.state, arm.error_code), flush=True)
    if ret != 0:
        raise SystemExit('move rejected (code %s); stopping' % ret)
    return ret


if cur[2] < args.look[2] - 5: go(cur[0], cur[1], args.look[2], 60)
go(*args.look, 60); time.sleep(0.8)
arm.set_gripper_enable(True); arm.set_gripper_position(0, wait=True)          # closed fingers touch first

pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
prof = pipe.start(cfg); i = prof.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]]); dist = np.array(i.coeffs[:5])
sq = args.square_mm / 1000
board = cv2.aruco.CharucoBoard((6, 8), sq, sq * 18 / 25, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
det = cv2.aruco.CharucoDetector(board)
code, pose = arm.get_position(is_radian=True)
Tbf = T(rpy_to_R(*pose[3:]), np.array(pose[:3]) / 1000)
pts = []
for _ in range(12):
    img = np.asanyarray(pipe.wait_for_frames(5000).get_color_frame().get_data())
    cc, ci, mc, mi = det.detectBoard(img)
    if ci is None or len(ci) < 12: continue
    objp, imgp = board.matchImagePoints(cc, ci); ok, rv, tv = cv2.solvePnP(objp, imgp, K, dist)
    pts.append((Tbf @ X @ T(cv2.Rodrigues(rv)[0], tv) @ np.array([3 * sq, 4 * sq, 0, 1]))[:3] * 1000)
pipe.stop()
if len(pts) < 5: sys.exit('board not visible from the look pose')
P = np.median(np.array(pts), axis=0)
print('predicted board centre corner: %s mm' % np.round(P, 1), flush=True)

# fingertip straight down: flange = tip + [0,0,gripper]
start_tip_z = P[2] + 60
go(P[0], P[1], start_tip_z + args.gripper_mm, 50)
time.sleep(1.0)
arm.ft_sensor_enable(1); time.sleep(0.3); arm.ft_sensor_set_zero(); time.sleep(0.5)
arm.set_mode(0); arm.set_state(0); time.sleep(0.5)          # enabling the F/T sensor can leave the arm not ready
code, f0 = arm.get_ft_sensor_data(); fz0 = f0[2]
print('F/T zeroed (Fz %.2f N); descending at 5 mm/s...' % fz0, flush=True)
go(P[0], P[1], start_tip_z - args.max_down_mm + args.gripper_mm, 5, wait=False)
contact = None; t0 = time.time()
while time.time() - t0 < (args.max_down_mm / 5 + 5):
    code, f = arm.get_ft_sensor_data()
    if code == 0 and abs(f[2] - fz0) > args.force_n:
        arm.set_state(4); time.sleep(0.2)                          # stop now
        code, p = arm.get_position(); contact = p[2] - args.gripper_mm
        print('CONTACT: |dFz| %.2f N at fingertip z %.1f mm' % (abs(f[2] - fz0), contact), flush=True)
        break
    if arm.error_code: print('arm error', arm.error_code); break
    if int((time.time() - t0) * 4) != int((time.time() - t0 - 0.011) * 4):
        code2, pp = arm.get_position()
        print('   t=%4.1fs tip z %7.1f mm  dFz %+5.2f N  state %s' % (time.time() - t0, pp[2] - args.gripper_mm, f[2] - fz0, arm.state), flush=True)
    time.sleep(0.01)
arm.clean_error(); arm.set_mode(0); arm.set_state(0); time.sleep(0.5)
code, p = arm.get_position(); go(p[0], p[1], p[2] + 60, 20)        # back off upward
go(*args.look, 60)
arm.disconnect()
if contact is None:
    sys.exit('no contact detected within %.0f mm' % args.max_down_mm)
e = contact - P[2]
print('true table height %.1f mm vs predicted %.1f mm -> error %+.1f mm' % (contact, P[2], e))
print('corrected camera z offset: %.4f m (was %.4f)' % (args.x_vec[2] - e / 1000, args.x_vec[2]))
