#!/usr/bin/env python3
"""Find objects on the table with the RealSense depth (arm stays where it is) and classify each one
with the RipeSense robot_fruit_model.  Writes /tmp/classify_view.jpg.
  ~/ripesense_env/bin/python classify_view.py
"""
import json, math, os, time
import cv2, numpy as np, pyrealsense2 as rs, yaml
from xarm.wrapper import XArmAPI

CFG = yaml.safe_load(open(os.path.expanduser('~/config.yaml')))
XV = CFG['camera']['eef_to_color']; DK = CFG['camera'].get('depth_scale_correction', 1.0)
MDIR = os.path.expanduser('~/models')
net = cv2.dnn.readNetFromONNX(MDIR + '/robot_fruit_model.onnx')
CLASSES = json.load(open(MDIR + '/robot_fruit_model_classes.json'))
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
VERDICT = lambda c: ('not a fruit' if c == 'not_fruit' else 'other fruit' if c == 'other_fruit'
                     else c.replace('_', ' '))


def classify(bgr):
    h, w = bgr.shape[:2]; s = 240.0 / min(h, w)
    img = cv2.resize(bgr, (max(224, round(w * s)), max(224, round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]; y0, x0 = (h - 224) // 2, (w - 224) // 2
    x = cv2.cvtColor(img[y0:y0 + 224, x0:x0 + 224], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    net.setInput(((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32))
    z = net.forward()[0]; p = np.exp(z - z.max()); p /= p.sum()
    return p


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.ravel(t); return M



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


arm = XArmAPI('192.168.1.232'); time.sleep(0.3); code, pose = arm.get_position(is_radian=True); arm.disconnect()
Tbc = T(rpy_to_R(*pose[3:]), np.array(pose[:3]) / 1000) @ T(rpy_to_R(*XV[3:]), XV[:3])
pipe = rs.pipeline(); cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30); cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
prof = pipe.start(cfg); align = rs.align(rs.stream.color)
intr = prof.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
sc = prof.get_device().first_depth_sensor().get_depth_scale()
spat, temp_f, hole = rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()
deps = []
for _ in range(15):
    f = align.process(pipe.wait_for_frames(5000))
    deps.append(np.asanyarray(hole.process(temp_f.process(spat.process(f.get_depth_frame()))).get_data()).astype(np.float32) * sc * DK)
    color = np.asanyarray(f.get_color_frame().get_data()).copy()
pipe.stop()
D = np.median(np.array(deps), 0); H, W = D.shape
us, vs = np.meshgrid(np.arange(W), np.arange(H)); valid = (D > 0.15) & (D < 1.2)
P = np.stack([(us - intr.ppx) / intr.fx * D, (vs - intr.ppy) / intr.fy * D, D, np.ones_like(D)], -1).reshape(-1, 4) @ Tbc.T
Pmm = P[:, :3].reshape(H, W, 3) * 1000
coef = fit_table_plane(Pmm, valid)
zb = Pmm[..., 2] - (coef[0] * Pmm[..., 0] + coef[1] * Pmm[..., 1] + coef[2])     # height above the table plane
table = float(np.median((Pmm[..., 2] - zb)[valid]))
print('table plane tilt: %.1f mm/m in x, %.1f mm/m in y' % (1000 * coef[0], 1000 * coef[1]))
m = cv2.morphologyEx((valid & (zb > 12) & (zb < 150)).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats(m)
vis = color.copy(); found = 0
for j in range(1, n):
    x, y, w, h, area = st[j]
    if area < 300 or x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
        continue
    pad = int(0.12 * max(w, h)); crop = color[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
    p = classify(crop); top = np.argsort(p)[::-1][:3]
    pts = P.reshape(H, W, 4)[lab == j][:, :3] * 1000
    hgt = float(np.percentile(zb[lab == j], 95))
    found += 1
    print('object %d at (%.0f, %.0f) mm, %.0f mm tall: %s  | top-3: %s' % (
        found, np.median(pts[:, 0]), np.median(pts[:, 1]), hgt, VERDICT(CLASSES[top[0]]).upper(),
        ', '.join('%s %.0f%%' % (CLASSES[k], 100 * p[k]) for k in top)), flush=True)
    col = (0, 200, 0) if 'fresh' in CLASSES[top[0]] or 'ripe' == CLASSES[top[0]].split('_')[-1] else (0, 0, 255) if 'rotten' in CLASSES[top[0]] else (0, 200, 255)
    cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
    cv2.putText(vis, '%s %.0f%%' % (CLASSES[top[0]], 100 * p[top[0]]), (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
if not found:
    print('no objects fully in view (table z %.0f mm)' % table)
cv2.imwrite('/tmp/classify_view.jpg', vis)
