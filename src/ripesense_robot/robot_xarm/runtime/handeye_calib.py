#!/usr/bin/env python3
"""Eye-in-hand calibration for the xArm6 + wrist RealSense (runs on the Pi 5).

The arm stays in MANUAL mode: you move it by hand to ~15 different poses looking at the
ChArUco board taped to the table (charuco_6x8_25mm.pdf). Whenever the arm has been still
for ~1 s and the board is clearly visible, a sample is captured automatically.
Ctrl+C (or --samples N reached) -> solve camera->flange with OpenCV calibrateHandEye.

  ~/ripesense_env/bin/python handeye_calib.py --samples 15
Outputs ~/handeye/: samples.json, frames/*.jpg, result.yaml, live view /tmp/handeye_view.jpg
"""
import argparse, json, math, os, signal, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np, pyrealsense2 as rs
from xarm.wrapper import XArmAPI

ap = argparse.ArgumentParser()
ap.add_argument('--ip', default='192.168.1.232')
ap.add_argument('--samples', type=int, default=15)
ap.add_argument('--square_mm', type=float, default=25.0)
ap.add_argument('--marker_mm', type=float, default=18.0)
ap.add_argument('--solve_only', action='store_true', help='re-solve from saved samples.json')
ap.add_argument('--auto', choices=['plan', 'run'], help='arm moves itself around the current (top-view) pose')
ap.add_argument('--board_dist_mm', type=float, default=460, help='pivot distance below the flange (camera to board)')
ap.add_argument('--speed', type=float, default=50, help='mm/s for --auto run')
ap.add_argument('--preview', type=float, default=0, help='only stream the annotated camera for N seconds (no motion)')
ap.add_argument('--port', type=int, default=8091, help='live MJPEG stream port (view via ssh -L)')
args = ap.parse_args()
OUT = os.path.expanduser('~/handeye'); os.makedirs(OUT + '/frames', exist_ok=True)
SAMPLES = OUT + '/samples.json'

# UFactory's value for their official D435 bracket (xyz m, rpy rad), for comparison
UF_EEF_TO_COLOR = [0.067052239, -0.0311387575, 0.021611456, -0.004202176, -0.00848499, 1.5898775]


def rpy_to_R(r, p, y):
    """xArm convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def R_to_rpy(R):
    p = math.asin(-max(-1.0, min(1.0, R[2, 0])))
    return math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.ravel(t); return M


def angle_deg(Ra, Rb):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2))))


# ---------------- live stream (http://<pi>:PORT/ , tunnel to laptop localhost) ----------------
_LATEST = {'jpg': None, 'status': 'starting'}
_LOCK = threading.Lock()


def publish(img, status):
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ok:
        with _LOCK:
            _LATEST['jpg'], _LATEST['status'] = enc.tobytes(), status


def start_stream(port):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            if self.path.startswith('/stream'):
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f'); self.end_headers()
                try:
                    while True:
                        with _LOCK: j = _LATEST['jpg']
                        if j:
                            self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(j) + j + b'\r\n')
                        time.sleep(0.08)
                except (BrokenPipeError, ConnectionResetError):
                    return
            elif self.path.startswith('/status'):
                with _LOCK: b = _LATEST['status'].encode()
                self.send_response(200); self.send_header('Content-Type', 'text/plain'); self.end_headers(); self.wfile.write(b)
            else:
                html = (b'<html><head><title>xArm hand-eye calibration</title></head><body style="background:#111;color:#eee;'
                        b'font-family:sans-serif;text-align:center"><h3>xArm6 hand-eye calibration - live camera</h3>'
                        b'<img src="/stream" style="max-width:96vw;border:1px solid #444"><p id=s></p><script>'
                        b'setInterval(()=>fetch("/status").then(r=>r.text()).then(t=>s.textContent=t),500)</script></body></html>')
                self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers(); self.wfile.write(html)
    srv = ThreadingHTTPServer(('0.0.0.0', port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print('live stream on port %d' % port, flush=True)


def annotate(img, det, board, K, dist, extra=''):
    """Detect the board, draw axes + info, return (n_corners, rvec, tvec, reproj, sharpness)."""
    cc, ci, mc, mi = det.detectBoard(img)
    n = 0 if ci is None else len(ci)
    sharp = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    rvec = tvec = None; err = 99.0
    if mi is not None:
        cv2.aruco.drawDetectedMarkers(img, mc, mi)
    if n >= 6:
        objp, imgp = board.matchImagePoints(cc, ci)
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)
        if ok:
            err = float(np.linalg.norm(cv2.projectPoints(objp, rvec, tvec, K, dist)[0].reshape(-1, 2) - imgp.reshape(-1, 2), axis=1).mean())
            cv2.drawFrameAxes(img, K, dist, rvec, tvec, 0.05)
            for pt in imgp.reshape(-1, 2): cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 255, 255), -1)
    txt = 'corners %d/35  dist %s  reproj %s  sharpness %.0f  %s' % (
        n, '%.0f mm' % (1000 * np.linalg.norm(tvec)) if tvec is not None else '-', '%.2f px' % err if tvec is not None else '-', sharp, extra)
    cv2.rectangle(img, (0, 0), (640, 26), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if n >= 12 else (0, 0, 255), 1)
    return n, rvec, tvec, err, sharp, txt


def solve(samples):
    Rg, tg, Rt, tt = [], [], [], []
    for s in samples:
        x, y, z, r, p, yw = s['arm']                       # mm, rad
        Rg.append(rpy_to_R(r, p, yw)); tg.append(np.array([x, y, z]) / 1000.0)
        Rt.append(cv2.Rodrigues(np.array(s['rvec']))[0]); tt.append(np.array(s['tvec']))
    results = {}
    for name, m in [('TSAI', cv2.CALIB_HAND_EYE_TSAI), ('PARK', cv2.CALIB_HAND_EYE_PARK),
                    ('HORAUD', cv2.CALIB_HAND_EYE_HORAUD), ('DANIILIDIS', cv2.CALIB_HAND_EYE_DANIILIDIS)]:
        R, t = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=m)
        X = T(R, t)                                          # camera -> flange (cam2gripper)
        # consistency: board pose in the base frame should be identical for every sample
        boards = [T(Rg[i], tg[i]) @ X @ T(Rt[i], tt[i]) for i in range(len(samples))]
        pos = np.array([b[:3, 3] for b in boards]) * 1000
        spread_mm = float(np.linalg.norm(pos - pos.mean(0), axis=1).mean())
        rot_spread = float(np.mean([angle_deg(boards[0][:3, :3], b[:3, :3]) for b in boards[1:]])) if len(boards) > 1 else 0
        results[name] = dict(xyz_m=[float(v) for v in t.ravel()], rpy_rad=[float(v) for v in R_to_rpy(R)],
                             board_pos_spread_mm=round(spread_mm, 2), board_rot_spread_deg=round(rot_spread, 2),
                             board_in_base_mm=[round(float(v), 1) for v in pos.mean(0)])
    best = min(results, key=lambda k: results[k]['board_pos_spread_mm'])
    uf = T(rpy_to_R(*UF_EEF_TO_COLOR[3:]), UF_EEF_TO_COLOR[:3])
    b = results[best]; Xb = T(rpy_to_R(*b['rpy_rad']), b['xyz_m'])
    diff_mm = float(np.linalg.norm((Xb[:3, 3] - uf[:3, 3]) * 1000)); diff_deg = angle_deg(Xb[:3, :3], uf[:3, :3])
    print('\n==== hand-eye result (camera -> flange), %d samples ====' % len(samples))
    for k, v in results.items():
        print('%-10s xyz_mm %s  rpy_deg %s  | board spread %.1f mm / %.2f deg%s' % (
            k, [round(1000 * a, 1) for a in v['xyz_m']], [round(math.degrees(a), 2) for a in v['rpy_rad']],
            v['board_pos_spread_mm'], v['board_rot_spread_deg'], '   <-- best' if k == best else ''))
    print('UFactory bracket   xyz_mm %s  rpy_deg %s' % ([round(1000 * a, 1) for a in UF_EEF_TO_COLOR[:3]],
                                                       [round(math.degrees(a), 2) for a in UF_EEF_TO_COLOR[3:]]))
    print('difference best vs UFactory: %.1f mm, %.2f deg' % (diff_mm, diff_deg))
    with open(OUT + '/result.yaml', 'w') as f:
        f.write('# camera(color optical) -> flange, xArm rpy convention, from %d samples\n' % len(samples))
        f.write('method: %s\nEULER_EEF_TO_COLOR_OPT: [%s]\n' % (best, ', '.join('%.6f' % v for v in b['xyz_m'] + b['rpy_rad'])))
        f.write('board_pos_spread_mm: %.2f\nboard_rot_spread_deg: %.2f\n' % (b['board_pos_spread_mm'], b['board_rot_spread_deg']))
        f.write('diff_vs_ufactory_mm: %.1f\ndiff_vs_ufactory_deg: %.2f\n' % (diff_mm, diff_deg))
        f.write('all_methods: %s\n' % json.dumps(results))
    print('saved', OUT + '/result.yaml')


if args.solve_only:
    solve(json.load(open(SAMPLES))); sys.exit()

def open_camera():
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]); dist = np.array(intr.coeffs[:5])
    board = cv2.aruco.CharucoBoard((6, 8), args.square_mm / 1000, args.marker_mm / 1000,
                                   cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
    return pipe, K, dist, board, cv2.aruco.CharucoDetector(board)


if args.preview:
    start_stream(args.port)
    pipe, K, dist, board, det = open_camera()
    t_end = time.time() + args.preview; last_print = 0
    try:
        while time.time() < t_end:
            img = np.asanyarray(pipe.wait_for_frames(5000).get_color_frame().get_data()).copy()
            n, rvec, tvec, err, sharp, txt = annotate(img, det, board, K, dist, 'PREVIEW (no motion)')
            publish(img, txt)
            if time.time() - last_print > 3:
                print(txt, flush=True); cv2.imwrite('/tmp/handeye_view.jpg', img); last_print = time.time()
    finally:
        pipe.stop()
    sys.exit()

# (closer_mm, tilt_x_deg, tilt_y_deg, rot_z_deg) around the board centre; first row = the start pose itself
AUTO_POSES = [(0, 0, 0, 0), (0, 12, 0, 0), (0, -12, 0, 0), (0, 0, 12, 0), (0, 0, -12, 0),
              (0, 8, 8, 20), (0, -8, 8, -20), (0, 8, -8, -20), (0, -8, -8, 20),
              (60, 10, 0, 30), (60, -10, 0, -30), (60, 0, 10, 15),
              (-60, 0, -10, -15), (-60, 10, 10, 0), (-60, -10, -10, 25)]


def Rx(a): c, s_ = math.cos(a), math.sin(a); return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
def Ry(a): c, s_ = math.cos(a), math.sin(a); return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
def Rz(a): c, s_ = math.cos(a), math.sin(a); return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])


def plan_poses(start, d_mm):
    """Flange poses that rotate about a pivot d_mm along the tool z axis (≈ the board centre)."""
    T0 = T(rpy_to_R(*start[3:]), np.array(start[:3]))
    piv = np.array([0, 0, d_mm])
    out = []
    for closer, tx, ty, rz in AUTO_POSES:
        Rl = Rz(math.radians(rz)) @ Ry(math.radians(ty)) @ Rx(math.radians(tx))
        M = T(np.eye(3), piv) @ T(Rl, np.zeros(3)) @ T(np.eye(3), -piv) @ T(np.eye(3), np.array([0, 0, closer]))
        Tn = T0 @ M
        out.append([*Tn[:3, 3].tolist(), *R_to_rpy(Tn[:3, :3])])
    return out


if args.auto:
    arm = XArmAPI(args.ip); time.sleep(0.3)
    code, start = arm.get_position(is_radian=True)
    print('start pose (manual top view): xyz %s mm, rpy %s deg' % ([round(v, 1) for v in start[:3]], [round(math.degrees(v), 1) for v in start[3:]]))
    plan = plan_poses(start, args.board_dist_mm)
    worst = 0
    for i, (q, spec) in enumerate(zip(plan, AUTO_POSES), 1):
        dmove = np.linalg.norm(np.array(q[:3]) - np.array(start[:3])); worst = max(worst, dmove)
        print('%2d closer %+4d mm tilt (%+3d, %+3d) rot %+3d deg -> xyz %s  (%.0f mm from start)' % (
            i, spec[0], spec[1], spec[2], spec[3], [round(v, 1) for v in q[:3]], dmove))
    print('largest move from start: %.0f mm' % worst)
    assert worst < 200, 'plan moves too far; refusing'
    if args.auto == 'plan':
        arm.disconnect(); sys.exit()
    # --- run: position mode, slow, collision detection on ---
    arm.clean_warn(); arm.clean_error(); arm.motion_enable(True); arm.set_mode(0); arm.set_state(0); time.sleep(0.5)
    arm.set_collision_sensitivity(3)
    start_stream(args.port)
    pipe, K, dist, board, det = open_camera()
    samples = []
    try:
        for i, q in enumerate(plan, 1):
            ret = arm.set_position(x=q[0], y=q[1], z=q[2], roll=q[3], pitch=q[4], yaw=q[5], speed=args.speed,
                                   mvacc=300, is_radian=True, wait=False)
            t0 = time.time(); time.sleep(0.3)
            while time.time() - t0 < 30 and (arm.state == 1 or time.time() - t0 < 0.6):   # stream while moving
                img = np.asanyarray(pipe.wait_for_frames(5000).get_color_frame().get_data()).copy()
                publish(img, annotate(img, det, board, K, dist, 'MOVING to pose %d/%d' % (i, len(plan)))[5])
                if arm.error_code: break
            if ret != 0 or arm.error_code:
                print('MOVE %d failed (ret %s, error %s): stopping' % (i, ret, arm.error_code)); break
            time.sleep(1.0)
            for _ in range(10): img = np.asanyarray(pipe.wait_for_frames(5000).get_color_frame().get_data()).copy()
            code, pose = arm.get_position(is_radian=True)
            n, rvec, tvec, err, sharp, txt = annotate(img, det, board, K, dist, 'pose %d/%d STILL' % (i, len(plan)))
            publish(img, txt)
            if n < 12 or tvec is None:
                print('pose %d: board not visible enough (%d corners), skipped' % (i, n)); continue
            cv2.imwrite('%s/frames/auto_%02d.jpg' % (OUT, i), img)
            samples.append(dict(arm=[float(v) for v in pose], rvec=rvec.ravel().tolist(), tvec=tvec.ravel().tolist(),
                                corners=n, reproj_px=float(err), t=time.time()))
            json.dump(samples, open(SAMPLES, 'w'), indent=1)
            print('pose %d: CAPTURED (%d corners, %.0f mm, reproj %.2f px)' % (i, n, 1000 * np.linalg.norm(tvec), err), flush=True)
        print('returning to start pose')
        arm.set_position(x=start[0], y=start[1], z=start[2], roll=start[3], pitch=start[4], yaw=start[5],
                         speed=args.speed, is_radian=True, wait=True)
    finally:
        pipe.stop(); arm.disconnect()
    if len(samples) >= 5: solve(samples)
    else: print('only %d samples: not enough' % len(samples))
    sys.exit()

arm = XArmAPI(args.ip); time.sleep(0.3)
print('arm mode %s state %s error %s (move it by hand in manual mode)' % (arm.mode, arm.state, arm.error_code), flush=True)
pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
prof = pipe.start(cfg)
intr = prof.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]); dist = np.array(intr.coeffs[:5])
print('colour intrinsics fx %.1f fy %.1f cx %.1f cy %.1f' % (intr.fx, intr.fy, intr.ppx, intr.ppy), flush=True)
board = cv2.aruco.CharucoBoard((6, 8), args.square_mm / 1000, args.marker_mm / 1000,
                               cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
det = cv2.aruco.CharucoDetector(board)

samples = json.load(open(SAMPLES)) if os.path.exists(SAMPLES) and '--resume' in sys.argv else []
history, stop = [], False
signal.signal(signal.SIGINT, lambda *a: globals().__setitem__('stop', True))
last_cap = None
while not stop and len(samples) < args.samples:
    img = np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data()).copy()
    code, pose = arm.get_position(is_radian=True)
    now = time.time()
    history = [(t, p) for t, p in history if now - t < 1.0] + [(now, pose)]
    still = len(history) > 10 and max(np.linalg.norm(np.array(p[:3]) - np.array(pose[:3])) for _, p in history) < 0.5 \
        and max(np.abs(np.array(p[3:]) - np.array(pose[3:])).max() for _, p in history) < math.radians(0.2)
    cc, ci, mc, mi = det.detectBoard(img)
    n = 0 if ci is None else len(ci)
    ok = False
    if n >= 12:
        objp, imgp = board.matchImagePoints(cc, ci)
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)
        if ok:
            cv2.drawFrameAxes(img, K, dist, rvec, tvec, 0.05)
            err = np.linalg.norm(cv2.projectPoints(objp, rvec, tvec, K, dist)[0].reshape(-1, 2) - imgp.reshape(-1, 2), axis=1).mean()
    far_enough = last_cap is None or (np.linalg.norm(np.array(pose[:3]) - np.array(last_cap[:3])) > 30 or
                                      angle_deg(rpy_to_R(*pose[3:]), rpy_to_R(*last_cap[3:])) > 8)
    status = 'samples %d/%d | board corners %d | %s | %s' % (len(samples), args.samples, n,
             'STILL' if still else 'moving', 'new pose' if far_enough else 'too close to last sample')
    cv2.putText(img, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if ok else (0, 0, 255), 2)
    if ok and still and far_enough and code == 0 and err < 1.0:
        samples.append(dict(arm=[float(v) for v in pose], rvec=rvec.ravel().tolist(), tvec=tvec.ravel().tolist(),
                            corners=n, reproj_px=float(err), t=now))
        cv2.imwrite('%s/frames/sample_%02d.jpg' % (OUT, len(samples)), img)
        json.dump(samples, open(SAMPLES, 'w'), indent=1)
        last_cap = pose
        print('CAPTURED sample %d: arm xyz %s mm, board %d corners at %.0f mm, reproj %.2f px' % (
            len(samples), [round(v, 1) for v in pose[:3]], n, 1000 * np.linalg.norm(tvec), err), flush=True)
    if int(now * 2) % 2 == 0:
        cv2.imwrite('/tmp/handeye_view.jpg', img)
        open('/tmp/handeye_status.txt', 'w').write(status + '\n')
pipe.stop(); arm.disconnect()
print('collected %d samples' % len(samples), flush=True)
if len(samples) >= 5:
    solve(samples)
else:
    print('need at least 5 samples (15 recommended)')
