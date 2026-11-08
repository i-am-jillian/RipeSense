#!/usr/bin/env python3
"""Search the table from several high viewpoints (arm only, rail not moved) and build one map of
all fruits in the arm's base frame. Detections closer than 5 cm are merged; each fruit keeps the
label from its best view (not cut off by the image edge, highest confidence).
  ~/ripesense_env/bin/python search.py            # runs the search, prints the map, saves /tmp/search_*.jpg
Live stream on port 8091 during the search.
"""
import json, math, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np, pyrealsense2 as rs, yaml
from xarm.wrapper import XArmAPI

CFG = yaml.safe_load(open(os.path.expanduser('~/config.yaml')))
XV = CFG['camera']['eef_to_color']; DK = CFG['camera'].get('depth_scale_correction', 1.0)
MD = os.path.expanduser('~/models'); net = cv2.dnn.readNetFromONNX(MD + '/robot_fruit_model.onnx')
CLASSES = json.load(open(MD + '/robot_fruit_model_classes.json'))
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
# flange viewpoints (x, y, z mm), tool straight down; unreachable ones are skipped
VIEWS = [(250, 0, 600), (250, 220, 560), (250, -220, 560), (380, 200, 500), (380, -200, 500),
         (400, 0, 520), (200, 320, 520), (200, -320, 520)]


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.ravel(t); return M


def classify(bgr):
    h, w = bgr.shape[:2]; s = 240.0 / min(h, w)
    img = cv2.resize(bgr, (max(224, round(w * s)), max(224, round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]; y0, x0 = (h - 224) // 2, (w - 224) // 2
    x = cv2.cvtColor(img[y0:y0 + 224, x0:x0 + 224], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    net.setInput(((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32))
    z = net.forward()[0]; p = np.exp(z - z.max()); p /= p.sum(); k = int(p.argmax())
    return CLASSES[k], float(p[k])


def fit_plane(Pmm, valid):
    X = Pmm[valid]; keep = np.abs(X[:, 2] - np.median(X[:, 2])) < 40
    for _ in range(6):
        A = np.c_[X[keep, 0], X[keep, 1], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(A, X[keep, 2], rcond=None)
        res = X[:, 2] - (coef[0] * X[:, 0] + coef[1] * X[:, 1] + coef[2])
        keep = np.abs(res) < max(4.0, 2.5 * np.std(res[keep]))
    return coef


LATEST = {'jpg': None, 'status': 'searching'}
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
            self.wfile.write(b'<html><body style="background:#111;color:#eee;font-family:sans-serif;text-align:center"><h3>xArm table search</h3>'
                             b'<img src="/stream" style="max-width:96vw"><pre id=s style="font-size:15px"></pre>'
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
spat, temp_f, hole = rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()
us, vs = np.meshgrid(np.arange(640), np.arange(480))


def go(x, y, z):
    ret = arm.set_position(x=x, y=y, z=z, roll=math.pi, pitch=0, yaw=0, speed=80, mvacc=400, is_radian=True, wait=False)
    if ret != 0: return False
    t0 = time.time(); time.sleep(0.3)
    while (arm.state == 1 or time.time() - t0 < 0.6) and time.time() - t0 < 40:
        show(np.asanyarray(align.process(pipe.wait_for_frames(5000)).get_color_frame().get_data()))
        if arm.error_code: raise SystemExit('ARM ERROR %s' % arm.error_code)
    return True


def look(vi):
    time.sleep(0.6)
    code, pose = arm.get_position(is_radian=True)
    Tbc = T(rpy_to_R(*pose[3:]), np.array(pose[:3]) / 1000) @ T(rpy_to_R(*XV[3:]), XV[:3])
    deps = []
    for _ in range(10):
        f = align.process(pipe.wait_for_frames(5000))
        deps.append(np.asanyarray(hole.process(temp_f.process(spat.process(f.get_depth_frame()))).get_data()).astype(np.float32) * sc * DK)
        color = np.asanyarray(f.get_color_frame().get_data()).copy()
    D = np.median(np.array(deps), 0); valid = (D > 0.15) & (D < 1.3)
    Pmm = (np.stack([(us - intr.ppx) / intr.fx * D, (vs - intr.ppy) / intr.fy * D, D, np.ones_like(D)], -1).reshape(-1, 4) @ Tbc.T)[:, :3].reshape(480, 640, 3) * 1000
    coef = fit_plane(Pmm, valid)
    zb = Pmm[..., 2] - (coef[0] * Pmm[..., 0] + coef[1] * Pmm[..., 1] + coef[2])
    m = cv2.morphologyEx((valid & (zb > 12) & (zb < 150)).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    dets = []
    for j in range(1, n):
        x, y, w, h, area = st[j]
        if area < 300: continue
        hgt = float(np.percentile(zb[lab == j], 95))
        if not (12 < hgt < 130): continue
        edge = bool(x <= 2 or y <= 2 or x + w >= 638 or y + h >= 478)
        pts = Pmm[lab == j]; pad = int(0.12 * max(w, h))
        cls, conf = classify(color[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad])
        dets.append(dict(x=float(np.median(pts[:, 0])), y=float(np.median(pts[:, 1])), height=hgt, cls=cls, conf=conf, edge=edge, view=vi))
        col = (150, 150, 150) if edge else (0, 220, 0)
        cv2.rectangle(color, (x, y), (x + w, y + h), col, 2)
        cv2.putText(color, '%s %.0f%%' % (cls, 100 * conf), (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.putText(color, 'view %d' % vi, (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    show(color); cv2.imwrite('/tmp/search_view%d.jpg' % vi, color)
    return dets


try:
    code, cur = arm.get_position()
    if cur[2] < 450: go(cur[0], cur[1], 520)
    all_dets = []
    for vi, v in enumerate(VIEWS, 1):
        code, ang = arm.get_inverse_kinematics([v[0], v[1], v[2], 180, 0, 0], input_is_radian=False, return_is_radian=False)
        if code != 0:
            print('view %d %s: unreachable, skipped' % (vi, v), flush=True); continue
        if not go(*v):
            print('view %d %s: move rejected, skipped' % (vi, v), flush=True); continue
        d = look(vi); all_dets += d
        msg = 'view %d at flange %s: %d objects' % (vi, v, len(d)); print(msg, flush=True); LATEST['status'] = msg
    go(*VIEWS[0])
    # merge detections of the same fruit (within 5 cm); best view = not at edge, then highest confidence
    fruits = []
    for d in sorted(all_dets, key=lambda d: (d['edge'], -d['conf'])):
        for f in fruits:
            if math.hypot(f['x'] - d['x'], f['y'] - d['y']) < 50:
                f['seen'] += 1; break
        else:
            fruits.append(dict(d, seen=1))
    fruits.sort(key=lambda f: (f['x'], f['y']))
    lines = ['TABLE MAP: %d fruits/objects (arm base frame, rail %s mm)' % (len(fruits), arm.get_linear_track_pos()[1])]
    for i, f in enumerate(fruits, 1):
        lines.append('%d. %-17s %3.0f%%  at (%4.0f, %4.0f) mm  %3.0f mm tall  seen in %d view(s)%s' % (
            i, f['cls'], 100 * f['conf'], f['x'], f['y'], f['height'], f['seen'], '  [only partly visible]' if f['edge'] else ''))
    print('\n'.join(lines)); LATEST['status'] = '\n'.join(lines)
    json.dump(fruits, open('/tmp/table_map.json', 'w'), indent=1)
    time.sleep(3)
finally:
    pipe.stop(); arm.disconnect()
