#!/usr/bin/env python3
"""One collection round: everything on the table is ONE class (e.g. all 7 ripe oranges).
The arm sweeps a grid of viewpoints; with <= 3 objects it also orbits each object.
Every object fully in view is cropped and saved to ~/rs_collect/<label>/<session>__vNN_oNN.jpg

  ~/ripesense_env/bin/python collect_class.py orange_fresh
Live stream on port 8091.
"""
import json, math, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np, pyrealsense2 as rs, yaml
from xarm.wrapper import XArmAPI

LABEL = sys.argv[1]
SMALL = len(sys.argv) > 2 and sys.argv[2] == 'small'      # small fruit (cherries): smaller crops, low grid
OUT = os.path.expanduser('~/rs_collect/' + LABEL); os.makedirs(OUT, exist_ok=True)
SESSION = time.strftime('%H%M%S')
CFG = yaml.safe_load(open(os.path.expanduser('~/config.yaml')))
XV = CFG['camera']['eef_to_color']; DK = CFG['camera'].get('depth_scale_correction', 1.0)
TABLE_Z = CFG['camera'].get('table_z_mm', -265.0)
SWEEP = [(250, 0, 600), (250, 220, 560), (250, -220, 560), (380, 200, 500), (380, -200, 500), (400, 0, 520),
         (200, 320, 520), (200, -320, 520), (300, 120, 430), (300, -120, 430), (420, 120, 400), (420, -120, 400),
         (330, 0, 380), (230, 200, 420), (230, -200, 420), (450, 0, 430)]


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.ravel(t); return M


X = T(rpy_to_R(*XV[3:]), XV[:3])


def fit_plane(Pmm, valid):
    Q = Pmm[valid]; keep = np.abs(Q[:, 2] - np.median(Q[:, 2])) < 40
    for _ in range(6):
        A = np.c_[Q[keep, 0], Q[keep, 1], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(A, Q[keep, 2], rcond=None)
        res = Q[:, 2] - (coef[0] * Q[:, 0] + coef[1] * Q[:, 1] + coef[2])
        keep = np.abs(res) < max(4.0, 2.5 * np.std(res[keep]))
    return coef


LATEST = {'jpg': None, 'status': 'collecting ' + LABEL}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith('/stream'):
            self.send_response(200); self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f'); self.end_headers()
            try:
                while True:
                    j = LATEST['jpg']
                    if j: self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(j) + j + b'\r\n')
                    time.sleep(0.08)
            except (BrokenPipeError, ConnectionResetError):
                return
        elif self.path.startswith('/status'):
            self.send_response(200); self.send_header('Content-Type', 'text/plain'); self.end_headers(); self.wfile.write(LATEST['status'].encode())
        else:
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
            self.wfile.write(b'<html><body style="background:#111;color:#eee;font-family:sans-serif;text-align:center"><h3>xArm training-photo collection</h3>'
                             b'<img src="/stream" style="max-width:96vw"><pre id=s style="font-size:16px"></pre>'
                             b'<script>setInterval(()=>fetch("/status").then(r=>r.text()).then(t=>s.textContent=t),700)</script></body></html>')
threading.Thread(target=ThreadingHTTPServer(('0.0.0.0', 8091), H).serve_forever, daemon=True).start()
def show(img):
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75]); LATEST['jpg'] = enc.tobytes() if ok else LATEST['jpg']

arm = XArmAPI('192.168.1.232'); time.sleep(0.3)
arm.clean_warn(); arm.clean_error(); arm.motion_enable(True); arm.set_mode(0); arm.set_state(0); time.sleep(0.5)
arm.set_collision_sensitivity(3)
pipe = rs.pipeline(); cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30); cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
prof = pipe.start(cfg); align = rs.align(rs.stream.color)
intr = prof.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
sc = prof.get_device().first_depth_sensor().get_depth_scale()
spat, hole = rs.spatial_filter(), rs.hole_filling_filter()
us, vs = np.meshgrid(np.arange(640), np.arange(480))
saved = 0


LIMITS = [(-360, 360), (-117.06, 116.02), (-219.04, 10.03), (-360, 360), (-97.0, 180.0), (-360, 360)]
MARGIN = 10.0
MAX_DJ = 60.0      # deg: refuse poses needing a big joint change from the current one
MIN_R = 220.0      # mm: flange never closer than this to the base axis
MAX_DJ_REACH = 110.0   # deg: shoulder/elbow/wrist-pitch may change more (normal up/down/tilt moves)
REFUSED = [0]


def go(q):
    """q = (x, y, z) tool down, or (x, y, z, roll, pitch, yaw) radians.  Safety gates before any motion:
    reachable; every joint >= MARGIN deg inside its limit; no joint moves more than MAX_DJ deg from now
    (no wide swings or configuration flips); flange >= MIN_R mm from the base axis.  Straight-line move."""
    if len(q) == 3: q = (q[0], q[1], q[2], math.pi, 0.0, 0.0)
    if math.hypot(q[0], q[1]) < MIN_R: return False
    code, ang = arm.get_inverse_kinematics([q[0], q[1], q[2], math.degrees(q[3]), math.degrees(q[4]), math.degrees(q[5])],
                                           input_is_radian=False, return_is_radian=False)
    if code != 0: return False
    if any(not (lo + MARGIN <= v <= hi - MARGIN) for v, (lo, hi) in zip(ang[:6], LIMITS)): return False
    cur = arm.angles[:6]
    dj = [abs(a - b) for a, b in zip(ang[:6], cur)]
    if max(dj[0], dj[3], dj[5]) > MAX_DJ or max(dj[1], dj[2], dj[4]) > MAX_DJ_REACH:
        REFUSED[0] += 1; return False
    if arm.set_position(x=q[0], y=q[1], z=q[2], roll=q[3], pitch=q[4], yaw=q[5], speed=100, mvacc=500, is_radian=True, wait=False) != 0:
        return False
    t0 = time.time(); time.sleep(0.3)
    while (arm.state == 1 or time.time() - t0 < 0.6) and time.time() - t0 < 40:
        show(np.asanyarray(align.process(pipe.wait_for_frames(5000)).get_color_frame().get_data()))
        if arm.error_code: raise SystemExit('ARM ERROR %s' % arm.error_code)
    return True


def capture(tag):
    """Crop every object fully in view; return their base-frame centres."""
    global saved
    time.sleep(0.35)
    code, pose = arm.get_position(is_radian=True)
    Tbc = T(rpy_to_R(*pose[3:]), np.array(pose[:3]) / 1000) @ X
    for _ in range(3): f = align.process(pipe.wait_for_frames(5000))
    color = np.asanyarray(f.get_color_frame().get_data()).copy(); vis = color.copy()
    D = np.asanyarray(hole.process(spat.process(f.get_depth_frame())).get_data()).astype(np.float32) * sc * DK
    valid = (D > 0.12) & (D < 1.3)
    Pmm = (np.stack([(us - intr.ppx) / intr.fx * D, (vs - intr.ppy) / intr.fy * D, D, np.ones_like(D)], -1).reshape(-1, 4) @ Tbc.T)[:, :3].reshape(480, 640, 3) * 1000
    if valid.sum() < 5000: return []
    coef = fit_plane(Pmm, valid)
    zb = Pmm[..., 2] - (coef[0] * Pmm[..., 0] + coef[1] * Pmm[..., 1] + coef[2])
    m = cv2.morphologyEx((valid & (zb > (6 if SMALL else 8)) & (zb < 160)).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3) if SMALL else (5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    centres = []
    for j in range(1, n):
        x, y, w, h, area = st[j]
        if area < (40 if SMALL else 150) or x <= 2 or y <= 2 or x + w >= 638 or y + h >= 478: continue
        hgt = float(np.percentile(zb[lab == j], 95))
        if not ((6 if SMALL else 8) < hgt < 140): continue
        pad = int(0.12 * max(w, h)); crop = color[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        if min(crop.shape[:2]) < (14 if SMALL else 28): continue
        cv2.imwrite(os.path.join(OUT, '%s__%s__%03d.jpg' % (SESSION, tag, saved)), crop); saved += 1
        pts = Pmm[lab == j]; centres.append((float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))))
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(vis, '%s: %d crops saved' % (LABEL, saved), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    show(vis); LATEST['status'] = 'collecting %s: %d crops saved (%s)' % (LABEL, saved, tag)
    return centres


def flange_for_camera_over(px, py, cam_h, tilt_deg=0.0, tilt_dir=0.0):
    R = rpy_to_R(math.pi, 0.0, 0.0)
    if tilt_deg:
        a = math.radians(tilt_deg); ax = np.array([math.cos(tilt_dir), math.sin(tilt_dir), 0.0])
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = (np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K) @ R
    TX = T(R, np.zeros(3)) @ X
    cam_off, cam_dir = TX[:3, 3] * 1000, TX[:3, 2]
    cam = np.array([px, py, TABLE_Z]) - cam_dir * (cam_h / max(0.3, -cam_dir[2]))
    fl = cam - cam_off
    return (fl[0], fl[1], fl[2], math.atan2(R[2, 1], R[2, 2]), math.asin(-max(-1, min(1, R[2, 0]))), math.atan2(R[1, 0], R[0, 0]))


try:
    code, cur = arm.get_position()
    if cur[2] < 380: arm.set_position(z=480, relative=False, x=cur[0], y=cur[1], roll=180, pitch=0, yaw=0, speed=60, wait=True)
    seen = []
    for vi, v in enumerate(SWEEP, 1):
        if go(v):
            for c in capture('v%02d' % vi):
                if all(math.hypot(c[0] - s[0], c[1] - s[1]) > 50 for s in seen): seen.append(c)
    print('%s: sweep done, %d objects on the table, %d crops' % (LABEL, len(seen), saved), flush=True)
    # tilted views (~25 deg) from 4 directions over the middle of the objects, to see their sides
    if seen:
        mx, my = float(np.mean([c[0] for c in seen])), float(np.mean([c[1] for c in seen]))
        for hgt in (420, 560):
            for k, ang in enumerate((0, 90, 180, 270)):
                q = flange_for_camera_over(mx, my, hgt, tilt_deg=25, tilt_dir=math.radians(ang))
                if go(q): capture('t%d_%d' % (hgt, k))
        print('%s: tilted views done, %d crops' % (LABEL, saved), flush=True)
    if 0 < len(seen) <= (8 if SMALL else 3):        # few objects: dense straight-down grid around each one
        for oi, (px, py) in enumerate(seen, 1):
            for hgt in ((250, 330) if SMALL else (330, 440, 560)):
                for dx in (-110, 0, 110):
                    for dy in (-110, 0, 110):
                        q = flange_for_camera_over(px + dx, py + dy, hgt)
                        if go(q): capture('g%d_%d_%d_%d' % (oi, hgt, dx, dy))
        print('%s: dense grid done, %d crops' % (LABEL, saved), flush=True)
    go((250, 0, 600))
    print('ROUND DONE: %s -> %d crops in %s (%d poses refused by the swing gate)' % (LABEL, saved, OUT, REFUSED[0]), flush=True)
    LATEST['status'] = 'ROUND DONE: %s, %d crops' % (LABEL, saved); time.sleep(2)
finally:
    pipe.stop(); arm.disconnect()
