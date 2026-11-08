#!/usr/bin/env python3
"""Continuous live view of the wrist RealSense on port 8091 (tunnel to laptop localhost:8091):
colour video at ~15 fps with every object on the table boxed and classified (refreshed every ~1.5 s),
plus the arm pose and rail position.  Other scripts that need the camera stop this first (pkill) and
restart it afterwards.
  ~/ripesense_env/bin/python live_view.py
"""
import json, math, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np, pyrealsense2 as rs, yaml
from xarm.wrapper import XArmAPI

PORT = 8091
CFG = yaml.safe_load(open(os.path.expanduser('~/config.yaml')))
XV = CFG['camera']['eef_to_color']; DK = CFG['camera'].get('depth_scale_correction', 1.0)
MD = os.path.expanduser('~/models'); net = cv2.dnn.readNetFromONNX(MD + '/robot_fruit_model.onnx')
CLASSES = json.load(open(MD + '/robot_fruit_model_classes.json'))
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
LATEST = {'jpg': None, 'status': 'starting'}; LOCK = threading.Lock()


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
    for _ in range(5):
        A = np.c_[X[keep, 0], X[keep, 1], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(A, X[keep, 2], rcond=None)
        res = X[:, 2] - A.dot(coef) if False else X[:, 2] - (coef[0] * X[:, 0] + coef[1] * X[:, 1] + coef[2])
        keep = np.abs(res) < max(4.0, 2.5 * np.std(res[keep]))
    return coef


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith('/stream'):
            self.send_response(200); self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f'); self.end_headers()
            try:
                while True:
                    with LOCK: j = LATEST['jpg']
                    if j: self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(j) + j + b'\r\n')
                    time.sleep(0.066)
            except (BrokenPipeError, ConnectionResetError):
                return
        elif self.path.startswith('/status'):
            with LOCK: b = LATEST['status'].encode()
            self.send_response(200); self.send_header('Content-Type', 'text/plain'); self.end_headers(); self.wfile.write(b)
        else:
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
            self.wfile.write(b'<html><head><title>xArm live view</title></head><body style="background:#111;color:#eee;'
                             b'font-family:sans-serif;text-align:center"><h3>xArm6 wrist camera - live</h3>'
                             b'<img src="/stream" style="max-width:96vw;border:1px solid #444"><pre id=s style="font-size:15px"></pre>'
                             b'<script>setInterval(()=>fetch("/status").then(r=>r.text()).then(t=>s.textContent=t),700)</script></body></html>')


threading.Thread(target=ThreadingHTTPServer(('0.0.0.0', PORT), H).serve_forever, daemon=True).start()
arm = XArmAPI('192.168.1.232'); time.sleep(0.3)
pipe = rs.pipeline(); cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15); cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
prof = pipe.start(cfg); align = rs.align(rs.stream.color)
intr = prof.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
sc = prof.get_device().first_depth_sensor().get_depth_scale()
spat, hole = rs.spatial_filter(), rs.hole_filling_filter()
boxes, last_det = [], 0.0
us, vs = np.meshgrid(np.arange(640), np.arange(480))
print('live view on port %d' % PORT, flush=True)
try:
    while True:
        f = align.process(pipe.wait_for_frames(5000))
        color = np.asanyarray(f.get_color_frame().get_data()).copy()
        code, pose = arm.get_position(is_radian=True)
        rail = arm.get_linear_track_pos()[1] if hasattr(arm, 'get_linear_track_pos') else None
        if time.time() - last_det > 1.5 and code == 0 and abs(abs(pose[3]) - math.pi) < 0.35 and abs(pose[4]) < 0.35:
            last_det = time.time()
            D = np.asanyarray(hole.process(spat.process(f.get_depth_frame())).get_data()).astype(np.float32) * sc * DK
            valid = (D > 0.15) & (D < 1.3)
            Tbc = T(rpy_to_R(*pose[3:]), np.array(pose[:3]) / 1000) @ T(rpy_to_R(*XV[3:]), XV[:3])
            Pmm = (np.stack([(us - intr.ppx) / intr.fx * D, (vs - intr.ppy) / intr.fy * D, D, np.ones_like(D)], -1).reshape(-1, 4)
                   @ Tbc.T)[:, :3].reshape(480, 640, 3) * 1000
            if valid.sum() > 5000:
                coef = fit_plane(Pmm, valid)
                zb = Pmm[..., 2] - (coef[0] * Pmm[..., 0] + coef[1] * Pmm[..., 1] + coef[2])
                m = cv2.morphologyEx((valid & (zb > 12) & (zb < 150)).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
                n, lab, st, _ = cv2.connectedComponentsWithStats(m)
                boxes = []
                for j in range(1, n):
                    x, y, w, h, area = st[j]
                    if area < 300: continue
                    edge = x <= 2 or y <= 2 or x + w >= 638 or y + h >= 478
                    pts = Pmm[lab == j]; hgt = float(np.percentile(zb[lab == j], 95))
                    if not (12 < hgt < 130): continue
                    pad = int(0.12 * max(w, h)); cls, conf = classify(color[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad])
                    boxes.append((x, y, w, h, cls, conf, np.median(pts[:, 0]), np.median(pts[:, 1]), hgt, edge))
        lines = ['arm flange (%.0f, %.0f, %.0f) mm   rail %s mm' % (pose[0], pose[1], pose[2], rail)]
        for i, (x, y, w, h, cls, conf, bx, by, hgt, edge) in enumerate(boxes, 1):
            col = (150, 150, 150) if edge else (0, 0, 255) if 'rotten' in cls else (0, 200, 255) if 'unripe' in cls else (0, 220, 0)
            cv2.rectangle(color, (x, y), (x + w, y + h), col, 2)
            cv2.putText(color, '%d %s %.0f%%' % (i, cls, 100 * conf), (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            lines.append('%d. %-17s %3.0f%%  at (%4.0f, %4.0f) mm  %3.0f mm tall%s' % (i, cls, 100 * conf, bx, by, hgt, '  [cut off at edge]' if edge else ''))
        ok, enc = cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with LOCK:
            if ok: LATEST['jpg'] = enc.tobytes()
            LATEST['status'] = '\n'.join(lines)
finally:
    pipe.stop(); arm.disconnect()
