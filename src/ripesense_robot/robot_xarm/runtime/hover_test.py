#!/usr/bin/env python3
"""Hand-eye check: put the closed fingertips HOVER_MM above the ChArUco board's centre corner.

Uses the calibrated camera->flange transform (config eef_to_color) and the gripper length
(flange face -> fingertips). The arm moves across at the current height, then descends slowly,
and holds there so a person can judge the error by eye.  --back returns to the saved start pose.
  ~/ripesense_env/bin/python hover_test.py [--hover_mm 30] [--back]
"""
import argparse, json, math, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np, pyrealsense2 as rs, yaml
from xarm.wrapper import XArmAPI

ap = argparse.ArgumentParser()
ap.add_argument('--hover_mm', type=float, default=30)
ap.add_argument('--gripper_mm', type=float, default=230)
ap.add_argument('--speed', type=float, default=40)
ap.add_argument('--back', action='store_true')
ap.add_argument('--port', type=int, default=8091)
args = ap.parse_args()
CFG = yaml.safe_load(open(os.path.expanduser('~/config.yaml')))
X_VEC = CFG['camera']['eef_to_color']                     # camera -> flange, xyz m + rpy rad
STATE = os.path.expanduser('~/handeye/hover_start.json')


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.ravel(t); return M


arm = XArmAPI('192.168.1.232'); time.sleep(0.3)
arm.clean_warn(); arm.clean_error(); arm.motion_enable(True); arm.set_mode(0); arm.set_state(0); time.sleep(0.5)
arm.set_collision_sensitivity(3)

if args.back:
    s = json.load(open(STATE))
    print('returning to start pose', [round(v, 1) for v in s[:3]])
    arm.set_position(x=s[0], y=s[1], z=s[2] + 0, roll=s[3], pitch=s[4], yaw=s[5], speed=args.speed, is_radian=True, wait=True)
    print('done, error', arm.error_code); arm.disconnect(); sys.exit()

# ---- live stream (same port as the calibration viewer) ----
LATEST = {'jpg': None}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if not self.path.startswith('/stream'):
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
            self.wfile.write(b'<html><body style="background:#111;color:#eee;text-align:center;font-family:sans-serif">'
                             b'<h3>xArm hover test</h3><img src="/stream" style="max-width:96vw"></body></html>'); return
        self.send_response(200); self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f'); self.end_headers()
        try:
            while True:
                j = LATEST['jpg']
                if j: self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(j) + j + b'\r\n')
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError):
            pass
threading.Thread(target=ThreadingHTTPServer(('0.0.0.0', args.port), H).serve_forever, daemon=True).start()

pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
prof = pipe.start(cfg)
i = prof.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]]); dist = np.array(i.coeffs[:5])
board = cv2.aruco.CharucoBoard((6, 8), 0.025, 0.018, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
det = cv2.aruco.CharucoDetector(board)
CENTER = np.array([3 * 0.025, 4 * 0.025, 0.0])          # inner corner at the board centre (board frame, m)

def frame():
    img = np.asanyarray(pipe.wait_for_frames(5000).get_color_frame().get_data()).copy()
    ok, enc = cv2.imencode('.jpg', img); LATEST['jpg'] = enc.tobytes() if ok else LATEST['jpg']
    return img

try:
    code, start = arm.get_position(is_radian=True)
    json.dump([float(v) for v in start], open(STATE, 'w'))
    Tbf = T(rpy_to_R(*start[3:]), np.array(start[:3]) / 1000)
    X = T(rpy_to_R(*X_VEC[3:]), X_VEC[:3])
    pts = []
    for _ in range(10):
        img = frame(); cc, ci, mc, mi = det.detectBoard(img)
        if ci is None or len(ci) < 12: continue
        objp, imgp = board.matchImagePoints(cc, ci); ok, rv, tv = cv2.solvePnP(objp, imgp, K, dist)
        Tcb = T(cv2.Rodrigues(rv)[0], tv)
        pts.append((Tbf @ X @ Tcb @ np.append(CENTER, 1))[:3] * 1000)
        cv2.drawFrameAxes(img, K, dist, rv, tv, 0.05)
        c_img = cv2.projectPoints(CENTER.reshape(1, 3), rv, tv, K, dist)[0].ravel()
        cv2.drawMarker(img, (int(c_img[0]), int(c_img[1])), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        ok, enc = cv2.imencode('.jpg', img); LATEST['jpg'] = enc.tobytes()
        cv2.imwrite('/tmp/hover_target.jpg', img)
    if len(pts) < 5:
        print('board not visible enough from the current pose'); sys.exit(1)
    P = np.median(np.array(pts), axis=0); jitter = np.linalg.norm(np.array(pts) - P, axis=1).max()
    print('board centre corner in base frame: %s mm (jitter %.1f mm over %d frames)' % (np.round(P, 1), jitter, len(pts)))

    # fingertips straight above P: keep the current tool orientation (pointing down)
    R = rpy_to_R(*start[3:]); tip_dir = R @ np.array([0, 0, 1.0])
    print('tool axis in base frame:', np.round(tip_dir, 3), '(should be ~[0,0,-1] = pointing down)')
    if tip_dir[2] > -0.9:
        print('tool is not pointing down enough; refusing'); sys.exit(1)
    goal_tip = P + np.array([0, 0, args.hover_mm])
    goal_flange = goal_tip - tip_dir * args.gripper_mm
    above = goal_flange.copy(); above[2] = max(goal_flange[2], start[2])
    move = np.linalg.norm(goal_flange - np.array(start[:3]))
    print('flange: start %s -> above %s -> goal %s mm (%.0f mm total)' % (np.round(start[:3], 1), np.round(above, 1), np.round(goal_flange, 1), move))
    assert move < 350, 'target too far from the start pose; refusing'
    assert goal_flange[2] - args.gripper_mm > -400, 'goal looks below a sane table height; refusing'
    for name, q in (('across', above), ('down', goal_flange)):
        print('moving', name, flush=True)
        arm.set_position(x=q[0], y=q[1], z=q[2], roll=start[3], pitch=start[4], yaw=start[5],
                         speed=args.speed if name == 'across' else 30, mvacc=300, is_radian=True, wait=False)
        t0 = time.time(); time.sleep(0.3)
        while (arm.state == 1 or time.time() - t0 < 0.6) and time.time() - t0 < 30:
            frame()
            if arm.error_code: break
        if arm.error_code:
            print('ARM ERROR', arm.error_code, '- stopped'); sys.exit(1)
    code, now = arm.get_position()
    tip_now = np.array(now[:3]) + tip_dir * args.gripper_mm
    print('HOLDING: fingertips should be %.0f mm above the board centre corner (red cross). tip at %s mm, target point %s mm'
          % (args.hover_mm, np.round(tip_now, 1), np.round(P, 1)), flush=True)
    t_end = time.time() + 60
    while time.time() < t_end: frame()            # keep streaming for a minute
finally:
    pipe.stop(); arm.disconnect()
