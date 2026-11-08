#!/usr/bin/env python3
"""Pick test for the hand-eye calibration: find ONE object standing on the table with the
RealSense depth image, grasp it from straight above, lift, hold, put it back.

  ~/ripesense_env/bin/python pick_test.py           # look + detect only (shows result, no grasp)
  ~/ripesense_env/bin/python pick_test.py --go      # look + detect + grasp/lift/hold/place
Live view on port 8091 (tunnel to laptop localhost:8091).
"""
import argparse, json, math, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np, pyrealsense2 as rs, yaml
from xarm.wrapper import XArmAPI

ap = argparse.ArgumentParser()
ap.add_argument('--go', action='store_true', help='actually grasp')
ap.add_argument('--gripper_mm', type=float, default=230, help='flange face -> fingertips (closed)')
ap.add_argument('--look', type=float, nargs=3, default=[263.7, 36.7, 313.5], help='flange xyz of the look pose (mm)')
ap.add_argument('--speed', type=float, default=60)
ap.add_argument('--lift_mm', type=float, default=60)
ap.add_argument('--port', type=int, default=8091)
ap.add_argument('--target', type=float, nargs=2, help='pick the object nearest this base-frame x y (mm)')
args = ap.parse_args()
CFG = yaml.safe_load(open(os.path.expanduser('~/config.yaml')))
XV = CFG['camera']['eef_to_color']


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.ravel(t); return M


X = T(rpy_to_R(*XV[3:]), XV[:3])

# ---- live stream ----
LATEST = {'jpg': None}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if not self.path.startswith('/stream'):
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
            self.wfile.write(b'<html><body style="background:#111;color:#eee;text-align:center;font-family:sans-serif">'
                             b'<h3>xArm pick test</h3><img src="/stream" style="max-width:96vw"></body></html>'); return
        self.send_response(200); self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f'); self.end_headers()
        try:
            while True:
                j = LATEST['jpg']
                if j: self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(j) + j + b'\r\n')
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError):
            pass
threading.Thread(target=ThreadingHTTPServer(('0.0.0.0', args.port), H).serve_forever, daemon=True).start()
def show(img):
    ok, enc = cv2.imencode('.jpg', img); LATEST['jpg'] = enc.tobytes() if ok else LATEST['jpg']

arm = XArmAPI('192.168.1.232'); time.sleep(0.3)
arm.clean_warn(); arm.clean_error(); arm.motion_enable(True); arm.set_mode(0); arm.set_state(0); time.sleep(0.5)
arm.set_collision_sensitivity(3)
DOWN = (math.pi, 0.0)            # roll, pitch: tool straight down

pipe = rs.pipeline(); cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30); cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
prof = pipe.start(cfg); align = rs.align(rs.stream.color)
intr = prof.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
dscale = prof.get_device().first_depth_sensor().get_depth_scale()
spat, temp_f, hole = rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()



def fit_table_plane(Pmm, valid):
    """Robust plane z = a*x + b*y + c through the table (iteratively drops points that stick up)."""
    X = Pmm[valid]
    keep = np.abs(X[:, 2] - np.median(X[:, 2])) < 40
    for _ in range(6):
        A = np.c_[X[keep, 0], X[keep, 1], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(A, X[keep, 2], rcond=None)
        res = X[:, 2] - (coef[0] * X[:, 0] + coef[1] * X[:, 1] + coef[2])
        keep = np.abs(res) < max(4.0, 2.5 * np.std(res[keep]))
    return coef


def move(x, y, z, yaw, speed, what):
    print('  move %-10s -> flange (%.0f, %.0f, %.0f)' % (what, x, y, z), flush=True)
    arm.set_position(x=x, y=y, z=z, roll=DOWN[0], pitch=DOWN[1], yaw=yaw, speed=speed, mvacc=300, is_radian=True, wait=False)
    t0 = time.time(); time.sleep(0.3)
    while (arm.state == 1 or time.time() - t0 < 0.6) and time.time() - t0 < 40:
        f = align.process(pipe.wait_for_frames(5000)); show(np.asanyarray(f.get_color_frame().get_data()))
        if arm.error_code: break
    if arm.error_code:
        raise SystemExit('ARM ERROR %s during %s' % (arm.error_code, what))


try:
    code, cur = arm.get_position(is_radian=True)
    yaw = cur[5]
    # 1) look pose, tool straight down (go up first if we are low)
    if cur[2] < args.look[2] - 5:
        move(cur[0], cur[1], args.look[2], yaw, args.speed, 'up')
    move(args.look[0], args.look[1], args.look[2], yaw, args.speed, 'look')
    time.sleep(0.8)
    code, pose = arm.get_position(is_radian=True)
    Tbc = T(rpy_to_R(*pose[3:]), np.array(pose[:3]) / 1000) @ X

    # 2) depth -> 3D points in the base frame (average of several filtered frames)
    deps = []
    for _ in range(15):
        f = align.process(pipe.wait_for_frames(5000))
        d = hole.process(temp_f.process(spat.process(f.get_depth_frame())))
        deps.append(np.asanyarray(d.get_data()).astype(np.float32) * dscale * CFG['camera'].get('depth_scale_correction', 1.0))
        color = np.asanyarray(f.get_color_frame().get_data()).copy()
    depth = np.median(np.array(deps), axis=0)
    H_, W_ = depth.shape
    us, vs = np.meshgrid(np.arange(W_), np.arange(H_))
    Z = depth; valid = (Z > 0.15) & (Z < 1.2)
    Xc = (us - intr.ppx) / intr.fx * Z; Yc = (vs - intr.ppy) / intr.fy * Z
    P = np.stack([Xc, Yc, Z, np.ones_like(Z)], -1).reshape(-1, 4) @ Tbc.T          # base frame, metres
    Pmm = P[:, :3].reshape(H_, W_, 3) * 1000
    coef = fit_table_plane(Pmm, valid)
    zb = Pmm[..., 2] - (coef[0] * Pmm[..., 0] + coef[1] * Pmm[..., 1] + coef[2])   # height above the table plane
    table_z = float(np.median((Pmm[..., 2] - zb)[valid]))
    print('table plane tilt: %.1f mm/m in x, %.1f mm/m in y' % (1000 * coef[0], 1000 * coef[1]), flush=True)
    above = valid & (zb > 12) & (zb < 150)
    m = cv2.morphologyEx(above.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    if n < 2:
        raise SystemExit('nothing standing on the table found (table z %.0f mm)' % table_z)
    # ignore blobs cut by the image border (table edges, things half out of view) and tiny specks
    cand = [j for j in range(1, n) if stats[j, cv2.CC_STAT_AREA] > 300 and stats[j, 0] > 2 and stats[j, 1] > 2
            and stats[j, 0] + stats[j, 2] < W_ - 2 and stats[j, 1] + stats[j, 3] < H_ - 2]
    if not cand:
        raise SystemExit('nothing fully in view standing on the table (table z %.0f mm)' % table_z)
    if args.target:
        def dist_to_target(j):
            q = Pmm[lab == j]
            return math.hypot(np.median(q[:, 0]) - args.target[0], np.median(q[:, 1]) - args.target[1])
        k = min(cand, key=dist_to_target)
        print('target (%.0f, %.0f): nearest object is %.0f mm away' % (args.target[0], args.target[1], dist_to_target(k)), flush=True)
        if dist_to_target(k) > 80:
            raise SystemExit('no object within 8 cm of the target; not grasping')
    else:
        k = max(cand, key=lambda j: stats[j, cv2.CC_STAT_AREA])
    obj = lab == k
    pts = P.reshape(H_, W_, 4)[obj][:, :3] * 1000
    height = float(np.percentile(zb[obj], 95))
    table_z = float(coef[0] * np.median(pts[:, 0]) + coef[1] * np.median(pts[:, 1]) + coef[2])   # table right under the object
    top = table_z + height
    cx, cy = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
    span = np.percentile(pts[:, :2], 95, axis=0) - np.percentile(pts[:, :2], 5, axis=0)
    x, y, w, h = stats[k, :4]
    vis = color.copy()
    vis[obj] = (0.5 * vis[obj] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    msg = 'object: centre (%.0f, %.0f) mm, top z %.0f, height %.0f mm, footprint %.0f x %.0f mm | table z %.0f' % (
        cx, cy, top, height, span[0], span[1], table_z)
    cv2.putText(vis, msg[:90], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    show(vis); cv2.imwrite('/tmp/pick_detect.jpg', vis)
    print(msg, flush=True)
    ok_size = 15 < height < 130 and 20 < max(span) < 120
    if not ok_size:
        raise SystemExit('object size implausible for a test pick; not grasping')
    if not args.go:
        print('detect only (use --go to grasp)'); raise SystemExit(0)

    # 3) grasp from straight above
    grasp_tip = max(top - min(0.5 * height, 35), table_z + 8)        # fingertip height at closing
    tip = np.array([0, 0, args.gripper_mm])                            # flange -> tip (tool frame)
    off = rpy_to_R(DOWN[0], DOWN[1], yaw) @ tip                         # in base frame (~ [0,0,-230])
    fl = lambda tx, ty, tz: (tx - off[0], ty - off[1], tz - off[2])
    print('grasp plan: fingertips close at z %.0f mm (object top %.0f, table %.0f)' % (grasp_tip, top, table_z), flush=True)
    arm.set_gripper_enable(True); arm.set_gripper_mode(0); arm.set_gripper_speed(3000)
    arm.set_gripper_position(850, wait=True)                            # open
    move(*fl(cx, cy, top + 80), yaw, args.speed, 'above')
    move(*fl(cx, cy, grasp_tip), yaw, 25, 'descend')
    arm.set_gripper_speed(1500); arm.set_gripper_position(0, wait=True, timeout=6)
    time.sleep(0.5)
    code, gpos = arm.get_gripper_position()
    print('gripper closed at position %s (0 = fully closed / nothing held)' % gpos, flush=True)
    move(*fl(cx, cy, grasp_tip + args.lift_mm), yaw, 30, 'lift')
    time.sleep(3.0)
    code, gpos2 = arm.get_gripper_position()
    held = gpos2 is not None and gpos2 > 40
    print('after lift + 3 s hold: gripper at %s -> %s' % (gpos2, 'OBJECT HELD' if held else 'nothing in the gripper'), flush=True)
    move(*fl(cx, cy, grasp_tip + 3), yaw, 25, 'place')
    arm.set_gripper_position(850, wait=True)
    move(*fl(cx, cy, top + 80), yaw, 40, 'retreat')
    move(args.look[0], args.look[1], args.look[2], yaw, args.speed, 'look')
    print('PICK TEST', 'PASSED' if held else 'FAILED', flush=True)
finally:
    pipe.stop(); arm.disconnect()
