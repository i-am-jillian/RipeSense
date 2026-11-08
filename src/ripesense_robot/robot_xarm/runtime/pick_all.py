#!/usr/bin/env python3
"""Pick every object on the table one after another: look from high up, find all objects with depth
(height above a fitted table plane), then for each: open, descend, close, lift, hold, put back, release.
Long objects (banana) get the wrist turned so the fingers close across them; if a grasp closes on air
the arm retries once with the wrist turned 90 degrees.  Classifies each object with robot_fruit_model.

  ~/ripesense_env/bin/python pick_all.py            # detect + list only
  ~/ripesense_env/bin/python pick_all.py --go       # pick them all
  ~/ripesense_env/bin/python pick_all.py --inspect 2 # ask Tiger which fruit to check, fly the camera over the top 2

Every look is logged to Tiger Data (scan_history.py): overview scans for all objects, close-up
'inspect' scans, and 'pick' outcomes.  --no_log turns that off.
"""
import argparse, json, math, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np, pyrealsense2 as rs, yaml
from xarm.wrapper import XArmAPI

ap = argparse.ArgumentParser()
ap.add_argument('--go', action='store_true')
ap.add_argument('--look', type=float, nargs=3, default=[263.7, 36.7, 459.0])
ap.add_argument('--lift_mm', type=float, default=100)
ap.add_argument('--hold_s', type=float, default=2.0)
ap.add_argument('--speed', type=float, default=80)
ap.add_argument('--port', type=int, default=8091)
ap.add_argument('--inspect', type=int, default=0, help='visit the N highest-priority fruits from the scan history')
ap.add_argument('--inspect_dist', type=float, default=280, help='camera height above the fruit top, mm')
ap.add_argument('--no_log', action='store_true')
ap.add_argument('--to_bin', action='store_true', help='only pick the --target fruit and drop it at the taught bin pose')
ap.add_argument('--target', choices=['ripe', 'spoiled'], default='ripe', help='--to_bin: which fruit to pick')
ap.add_argument('--min_conf', type=float, default=0.5, help='--to_bin: minimum confidence to count as ripe')
ap.add_argument('--hold', default=os.environ.get('HOLD_KINDS', ''), help='comma list of fruit kinds never put in the bin (e.g. strawberry)')
args = ap.parse_args()
CFG = yaml.safe_load(open(os.path.expanduser('~/config.yaml')))
XV = CFG['camera']['eef_to_color']; DK = CFG['camera'].get('depth_scale_correction', 1.0)
GRIP = CFG['arm'].get('gripper_length_mm', 230)
MD = os.path.expanduser('~/models'); net = cv2.dnn.readNetFromONNX(MD + '/robot_fruit_model.onnx')
CLASSES = json.load(open(MD + '/robot_fruit_model_classes.json'))
MODEL = 'robot_fruit_model@' + time.strftime('%m%d-%H%M', time.localtime(os.path.getmtime(MD + '/robot_fruit_model.onnx')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import scan_history as SH
SESSION = time.strftime('%Y%m%d-%H%M%S'); CROPS = os.path.expanduser('~/scan_crops/' + SESSION)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


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
    return CLASSES[k], float(p[k]), p


def fit_table_plane(Pmm, valid):
    X = Pmm[valid]; keep = np.abs(X[:, 2] - np.median(X[:, 2])) < 40
    for _ in range(6):
        A = np.c_[X[keep, 0], X[keep, 1], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(A, X[keep, 2], rcond=None)
        res = X[:, 2] - (coef[0] * X[:, 0] + coef[1] * X[:, 1] + coef[2])
        keep = np.abs(res) < max(4.0, 2.5 * np.std(res[keep]))
    return coef


LATEST = {'jpg': None}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if not self.path.startswith('/stream'):
            self.send_response(200); self.send_header('Content-Type', 'text/html'); self.end_headers()
            self.wfile.write(b'<html><body style="background:#111;color:#eee;text-align:center;font-family:sans-serif">'
                             b'<h3>xArm pick-all</h3><img src="/stream" style="max-width:96vw"></body></html>'); return
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
pipe = rs.pipeline(); cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30); cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
prof = pipe.start(cfg); align = rs.align(rs.stream.color)
intr = prof.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
dscale = prof.get_device().first_depth_sensor().get_depth_scale()
spat, temp_f, hole = rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()
code, cur = arm.get_position(is_radian=True)
YAW0 = cur[5] if abs(cur[4]) < 0.3 and abs(abs(cur[3]) - math.pi) < 0.3 else 0.0   # tool not pointing down -> use yaw 0


def move(x, y, z, yaw, speed, what):
    ret = arm.set_position(x=x, y=y, z=z, roll=math.pi, pitch=0.0, yaw=yaw, speed=speed, mvacc=400, is_radian=True, wait=False)
    if ret != 0:
        raise SystemExit('move %s rejected (code %s)' % (what, ret))
    t0 = time.time(); time.sleep(0.3)
    while (arm.state == 1 or time.time() - t0 < 0.6) and time.time() - t0 < 40:
        show(np.asanyarray(align.process(pipe.wait_for_frames(5000)).get_color_frame().get_data()))
        if arm.error_code: break
    if arm.error_code:
        raise SystemExit('ARM ERROR %s during %s' % (arm.error_code, what))


def tip_to_flange(tx, ty, tz, yaw):
    off = rpy_to_R(math.pi, 0.0, yaw) @ np.array([0, 0, GRIP])
    return tx - off[0], ty - off[1], tz - off[2]


def rail_mm():
    try:
        code, pos = arm.get_linear_track_pos()
        return float(pos) if code == 0 and pos is not None else None
    except Exception:  # noqa: BLE001
        return None


def detect(source='overview'):
    code, pose = arm.get_position(is_radian=True)
    Tbc = T(rpy_to_R(*pose[3:]), np.array(pose[:3]) / 1000) @ T(rpy_to_R(*XV[3:]), XV[:3])
    deps = []
    for _ in range(15):
        f = align.process(pipe.wait_for_frames(5000))
        deps.append(np.asanyarray(hole.process(temp_f.process(spat.process(f.get_depth_frame()))).get_data()).astype(np.float32) * dscale * DK)
        color = np.asanyarray(f.get_color_frame().get_data()).copy()
    D = np.median(np.array(deps), 0); Hh, Ww = D.shape
    us, vs = np.meshgrid(np.arange(Ww), np.arange(Hh)); valid = (D > 0.15) & (D < 1.3)
    Pmm = (np.stack([(us - intr.ppx) / intr.fx * D, (vs - intr.ppy) / intr.fy * D, D, np.ones_like(D)], -1).reshape(-1, 4) @ Tbc.T)[:, :3].reshape(Hh, Ww, 3) * 1000
    coef = fit_table_plane(Pmm, valid)
    zb = Pmm[..., 2] - (coef[0] * Pmm[..., 0] + coef[1] * Pmm[..., 1] + coef[2])
    m = cv2.morphologyEx((valid & (zb > 12) & (zb < 150)).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    objs, vis = [], color.copy()
    for j in range(1, n):
        x, y, w, h, area = st[j]
        if area < 300 or x <= 2 or y <= 2 or x + w >= Ww - 2 or y + h >= Hh - 2:
            continue
        pts = Pmm[lab == j]; hgt = float(np.percentile(zb[lab == j], 95))
        if not (12 < hgt < 130): continue
        cx, cy = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
        xy = pts[:, :2] - [cx, cy]; ev, evec = np.linalg.eigh(np.cov(xy.T))
        long_ang = math.atan2(evec[1, 1], evec[0, 1])                                   # long axis direction
        length = float(np.ptp(xy @ evec[:, 1])); width = float(np.ptp(xy @ evec[:, 0]))
        tz = float(coef[0] * cx + coef[1] * cy + coef[2])
        pad = int(0.12 * max(w, h)); crop = color[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        cls, conf, p = classify(crop)
        hue, sat, val, dark = SH.colour_stats(color, (lab == j).astype(np.uint8))
        os.makedirs(CROPS, exist_ok=True); cp = '%s/%s_%s_%d.jpg' % (CROPS, time.strftime('%H%M%S'), source, len(objs) + 1)
        cv2.imwrite(cp, crop)
        top3 = np.argsort(p)[::-1][:3]
        scan = dict(session=SESSION, source=source, x_mm=cx, y_mm=cy, top_z_mm=tz + hgt, table_z_mm=tz, rail_mm=RAIL,
                    cam_pose=dict(zip(['x', 'y', 'z'], [round(v, 1) for v in pose[:3]]), **dict(zip(['roll', 'pitch', 'yaw'], [round(math.degrees(v), 2) for v in pose[3:]]))),
                    cam_dist_mm=float(Tbc[2, 3] * 1000 - (tz + hgt)), height_mm=hgt, length_mm=length, width_mm=width,
                    area_px=int(area), hue_deg=hue, sat=sat, val=val, dark_frac=dark, spectral=None, model=MODEL,
                    cls=cls, conf=conf, p_rotten=float(sum(p[i] for i, c in enumerate(CLASSES) if c.endswith('_rotten'))),
                    probs={CLASSES[i]: round(float(p[i]), 4) for i in top3}, crop_path=cp)
        objs.append(dict(cx=cx, cy=cy, height=hgt, table=tz, top=tz + hgt, long_ang=long_ang, length=length, width=width, cls=cls, conf=conf, scan=scan))
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, '%d %s %.0f%%' % (len(objs), cls, 100 * conf), (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    show(vis); cv2.imwrite('/tmp/pick_all.jpg', vis)
    return objs


# Gentle grasp.  The UFACTORY gripper has no force setting, but it is position-controlled: the grip force
# grows with how far past the fruit surface we command it.  So: find contact by creeping closed, then
# squeeze only a little past contact (per fruit type; spoiled fruit softer).  Units: 10 per mm of opening.
SQUEEZE_MM = {'banana': 1.5, 'strawberry': 1.0, 'mango': 2.0, 'avocado': 2.0, 'lemon': 2.0, 'apple': 2.5, 'orange': 2.5}
MAX_EXTRA_MM = 2.0          # extra squeeze allowed in total if the fruit slips while lifting


def squeeze_for(cls):
    mm = SQUEEZE_MM.get(cls.split('_')[0], 2.0)
    return mm * (0.6 if cls.endswith('_rotten') else 1.0)


def across_mm(o, rotated):
    """Fruit size across the fingers (depth blob), mm."""
    elong = o['length'] / max(1.0, o['width'])
    if elong > 1.6:
        return o['length'] if rotated else o['width']
    return min(o['length'], o['width'])


def grip_pos():
    code, g = arm.get_gripper_position()
    return g if code == 0 and g is not None else None


def gentle_close(o, rotated):
    """Close until contact, then squeeze a little.  Returns (contact, final) gripper units, or (None, g) on air."""
    w = across_mm(o, rotated)
    start = 850 if not (20 < w < 85) else min(850, int((w + 6) * 10))
    arm.set_gripper_speed(2500); arm.set_gripper_position(start, wait=True, timeout=4)
    arm.set_gripper_speed(800)
    prev = grip_pos(); cmd = start; contact = None
    while cmd > 0:
        cmd = max(0, cmd - 20)                                    # 2 mm steps
        arm.set_gripper_position(cmd, wait=True, timeout=1.5); time.sleep(0.15)
        g = grip_pos()
        if g is None: break
        if g - cmd > 12 or (prev is not None and prev - g < 6):   # fingers stopped short of the command -> contact
            contact = g; break
        prev = g
    if contact is None:
        return None, grip_pos()
    final = max(0, int(contact - 10 * squeeze_for(o['cls'])))
    arm.set_gripper_position(final, wait=True, timeout=2); time.sleep(0.3)
    print('   contact at %.0f mm, squeezed %.1f mm -> %s (fruit ~%.0f mm across by depth)' % (
        contact / 10, squeeze_for(o['cls']), final, w), flush=True)
    return contact, final


DROP = json.load(open(os.path.expanduser('~/drop_poses.json'))) if os.path.exists(os.path.expanduser('~/drop_poses.json')) else {}
SAFE_Z = 200.0
CARRY_R = 450.0       # pull the fruit in to this radius before rising (far fruit + high z is out of reach: C21)
PAD_ABOVE_TIP_MM = 10.0   # finger-pad contact centre above the fingertip        # flange height for carrying fruit to the bin


def is_target(o):
    if args.target == 'spoiled':
        return o['conf'] >= args.min_conf and o['cls'].endswith('_rotten')
    return is_ripe(o)


BIN_NAME = 'spoiled_bin' if args.target == 'spoiled' else 'ripe_bin'


def is_ripe(o):
    c = o['cls']
    if c.split('_')[0] in [k.strip() for k in args.hold.split(',') if k.strip()]:
        return False
    return o['conf'] >= args.min_conf and c not in ('not_fruit', 'other_fruit') and not c.endswith(('_rotten', '_unripe', '_overripe'))


def joint_move(angles, what, speed=30):
    if arm.set_servo_angle(angle=list(angles), speed=speed, mvacc=150, wait=False) != 0:
        raise SystemExit('joint move %s rejected' % what)
    t0 = time.time(); time.sleep(0.3)
    while (arm.state == 1 or time.time() - t0 < 0.6) and time.time() - t0 < 60:
        show(np.asanyarray(align.process(pipe.wait_for_frames(5000)).get_color_frame().get_data()))
        if arm.error_code: break
    if arm.error_code:
        raise SystemExit('ARM ERROR %s during %s' % (arm.error_code, what))


def line_move(x, y, z, roll, pitch, yaw, what, speed=40):
    """Straight-line move to a pose given in mm / degrees."""
    if arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, speed=speed, mvacc=300, is_radian=False, wait=False) != 0:
        raise SystemExit('move %s rejected' % what)
    t0 = time.time(); time.sleep(0.3)
    while (arm.state == 1 or time.time() - t0 < 0.6) and time.time() - t0 < 40:
        show(np.asanyarray(align.process(pipe.wait_for_frames(5000)).get_color_frame().get_data()))
        if arm.error_code: break
    if arm.error_code:
        raise SystemExit('ARM ERROR %s during %s' % (arm.error_code, what))


def look_joints():
    code, ang = arm.get_inverse_kinematics([args.look[0], args.look[1], args.look[2], 180, 0, math.degrees(YAW0)],
                                           input_is_radian=False, return_is_radian=False)
    if code != 0: raise SystemExit('look pose IK failed')
    return ang[:6]


def drop_in_bin(o, y_try, g):
    """Fruit is held and lifted: rise straight up, swing to the taught bin pose, open gently, swing back."""
    b = DROP[BIN_NAME]
    code, p = arm.get_position(is_radian=True)
    r = math.hypot(p[0], p[1])
    if r > CARRY_R:                                   # far out: bring it in first, at the current (lifted) height
        k = CARRY_R / r
        move(p[0] * k, p[1] * k, p[2], y_try, 30, 'pull in')
        code, p = arm.get_position(is_radian=True)
    if p[2] < SAFE_Z:
        move(p[0], p[1], SAFE_Z, y_try, 30, 'rise to carry height')
    cur = arm.angles[:6]; dj = [abs(x - y) for x, y in zip(b['joints'], cur)]
    if dj[0] > 200 or dj[3] > 120 or dj[5] > 120:
        print('   swing to the bin refused (joint change %s deg) -> putting it back' % [round(d) for d in dj], flush=True)
        return False
    # approach from ABOVE: swing to a point over the bin, then straight down, so the fruit hanging below the
    # fingers clears the rim
    bx, by, bz, br, bp, byaw = b['pose']
    up = max(120.0, o['height'] + 60.0)
    from above import above_joints                    # same arm configuration as the taught bin pose (no wrist flip)
    above = above_joints(arm, b['joints'], up); code = 0 if above else 1
    if code != 0:
        print('   no reachable pose %.0f mm above the bin -> putting it back' % up, flush=True); return False
    print('   carrying to %.0f mm above the bin (base turns %.0f deg)...' % (up, dj[0]), flush=True)
    joint_move(above[:6], 'above bin')
    drop_z = bz + max(0.0, o['height'] - 40.0)        # tall fruit hang lower: release higher (big apple hit C31)
    line_move(bx, by, drop_z, br, bp, byaw, 'down into bin')
    arm.set_gripper_speed(1500); arm.set_gripper_position(850, wait=True); time.sleep(0.8)
    if (g or 0) > 700:                                # wide fruit: shake the wrist so a wedged fruit falls out
        c0 = arm.angles[:6]
        for dj in (20, -20, 0):
            arm.set_servo_angle(angle=c0[:5] + [c0[5] + dj], speed=90, mvacc=600, wait=True)
        print('   shook the wrist to free a wide fruit', flush=True)
    g3 = grip_pos()
    print('   released in the %s (gripper %s)' % ('bin' if args.target == 'ripe' else 'spoiled bin', g3), flush=True)
    line_move(bx, by, bz + up, br, bp, byaw, 'up out of bin')
    joint_move(look_joints(), 'back from bin')
    return True


def refine(o):
    """Close look right before gripping: camera straight above the fruit (~25 cm), re-detect, re-centre.
    The wide overhead view is least accurate near its edges (an orange at the edge was missed twice)."""
    tcam = rpy_to_R(math.pi, 0.0, YAW0) @ (np.array(XV[:3]) * 1000)
    fx, fy, fz = o['cx'] - tcam[0], o['cy'] - tcam[1], o['top'] + 250 - tcam[2]
    code, _ = arm.get_inverse_kinematics([fx, fy, fz, 180, 0, math.degrees(YAW0)], input_is_radian=False, return_is_radian=False)
    if code != 0 or math.hypot(fx, fy) > 640:
        return
    try:
        move(fx, fy, fz, YAW0, args.speed, 'close look'); time.sleep(0.4)
        near = [d for d in detect('refine') if math.hypot(d['cx'] - o['cx'], d['cy'] - o['cy']) < 70]
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print('   close look failed (%s), using the overview position' % e, flush=True); return
    if not near:
        print('   close look: fruit not found near its position -> using the overview position', flush=True); return
    d = min(near, key=lambda d: math.hypot(d['cx'] - o['cx'], d['cy'] - o['cy']))
    print('   close look: re-centred by %.0f mm' % math.hypot(d['cx'] - o['cx'], d['cy'] - o['cy']), flush=True)
    for k in ('cx', 'cy', 'top', 'table', 'height', 'length', 'width', 'long_ang'):
        o[k] = d[k]


def pick(o, n_total, idx):
    refine(o)
    elong = o['length'] / max(1.0, o['width'])
    yaw = YAW0
    if elong > 1.6:                                       # long fruit: fingers across it (assume they close along tool y)
        yaw = o['long_ang']
        while yaw - YAW0 > math.pi / 2: yaw -= math.pi
        while yaw - YAW0 < -math.pi / 2: yaw += math.pi
    # fingertips 10 mm below the fruit's middle: the finger pads touch ~10 mm above the tips, so the pads
    # close on the widest part (old rule aimed the tips at the middle -> pads gripped above the equator)
    grasp = max(o['top'] - 0.5 * o['height'] - PAD_ABOVE_TIP_MM, o['table'] + 8)
    # fruit wider than the gripper can comfortably open around (max 85 mm): grip it higher, where the
    # cross-section is ~75 mm, so the fingers can open past it and it drops freely (a 97 mm apple wedged)
    wide = min(o['length'], o['width'])
    if wide > 78:
        R = wide / 2.0; d = 0.5 * o['height'] * math.sqrt(max(0.0, 1 - (37.5 / R) ** 2))   # 75 mm section height above centre (ellipse: fruit are flatter than wide)
        grasp = min(o['top'] - 12, (o['top'] - 0.5 * o['height']) + d - PAD_ABOVE_TIP_MM)
        print('   wide fruit (%.0f mm): gripping higher, at the 75 mm section' % wide, flush=True)
    print('[%d/%d] %s (%.0f%%) at (%.0f, %.0f), %.0f mm tall, %.0f x %.0f mm%s -> close at z %.0f' % (
        idx, n_total, o['cls'], 100 * o['conf'], o['cx'], o['cy'], o['height'], o['length'], o['width'],
        ', wrist %.0f deg' % math.degrees(yaw - YAW0) if elong > 1.6 else '', grasp), flush=True)
    # reach check before any motion: the approach and the grasp pose must both be solvable, else skip the fruit
    for zz, what in ((o['top'] + 70, 'above'), (grasp, 'grasp')):
        fx, fy, fz = tip_to_flange(o['cx'], o['cy'], zz, yaw)
        code, _ = arm.get_inverse_kinematics([fx, fy, fz, 180, 0, math.degrees(yaw)], input_is_radian=False, return_is_radian=False)
        if code != 0 or math.hypot(fx, fy) > 640:
            print('   OUT OF REACH (%s pose, %.0f mm from the base) -> skipped; move it closer or use the rail' % (what, math.hypot(fx, fy)), flush=True)
            return False
    for attempt in (0, 1):
        y_try = yaw + (math.pi / 2 if attempt else 0)
        arm.set_gripper_speed(3000); arm.set_gripper_position(850, wait=True)
        move(*tip_to_flange(o['cx'], o['cy'], o['top'] + 70, y_try), y_try, args.speed, 'above')
        move(*tip_to_flange(o['cx'], o['cy'], grasp, y_try), y_try, 30, 'descend')
        contact, g = gentle_close(o, attempt == 1)
        if contact is not None and contact > 40:
            break
        print('   closed on nothing (gripper %s)%s' % (g, ', retrying with the wrist turned 90 deg' if attempt == 0 else ''), flush=True)
        arm.set_gripper_position(850, wait=True)
        move(*tip_to_flange(o['cx'], o['cy'], o['top'] + 70, y_try), y_try, 40, 'up')
    else:
        return False
    extra = 0.0
    while True:                                         # lift slowly; if the fruit slips, squeeze 1 mm more and retry
        move(*tip_to_flange(o['cx'], o['cy'], grasp + 25, y_try), y_try, 15, 'test lift')
        time.sleep(0.5); g1 = grip_pos()
        slipped = g1 is None or g1 < 40 or g1 < g - 15
        if not slipped or extra >= MAX_EXTRA_MM:
            break
        extra += 1.0; print('   slipping (gripper %s) -> squeeze 1 mm more' % g1, flush=True)
        move(*tip_to_flange(o['cx'], o['cy'], grasp, y_try), y_try, 15, 'reseat')
        g = max(0, int(g - 10)); arm.set_gripper_position(g, wait=True, timeout=2); time.sleep(0.3)
    move(*tip_to_flange(o['cx'], o['cy'], grasp + args.lift_mm, y_try), y_try, 25, 'lift')
    time.sleep(args.hold_s)
    g2 = grip_pos(); held = g2 is not None and g2 > 40
    print('   lifted %.0f mm, gripper %s -> %s' % (args.lift_mm, g2, 'HELD' if held else 'DROPPED'), flush=True)
    if args.to_bin and held and drop_in_bin(o, y_try, g2):
        return held
    move(*tip_to_flange(o['cx'], o['cy'], grasp + 3, y_try), y_try, 15, 'place')
    arm.set_gripper_speed(1500); arm.set_gripper_position(min(850, g + 200), wait=True)
    move(*tip_to_flange(o['cx'], o['cy'], o['top'] + 70, y_try), y_try, 40, 'retreat')
    return held


def inspect(n):
    """Ask the scan history which fruit to check, put the camera straight above each, scan close up, log it."""
    try:
        todo = SH.inspect_next(n)
    except Exception as e:  # noqa: BLE001
        print('inspect: cannot reach Tiger (%s)' % e); return
    tcam = rpy_to_R(math.pi, 0.0, YAW0) @ (np.array(XV[:3]) * 1000)
    for k, r in enumerate(todo, 1):
        print('\n[inspect %d/%d] fruit #%d %s at (%.0f, %.0f), priority %.2f -- %s' % (
            k, len(todo), r['fruit_id'], r['cls'], r['x_mm'], r['y_mm'], r['priority'], SH.reasons(r)), flush=True)
        zc = -265 + 80 + args.inspect_dist                                     # table ~ -265, fruit top ~ +80
        move(r['x_mm'] - tcam[0], r['y_mm'] - tcam[1], zc - tcam[2], YAW0, 60, 'inspect')
        time.sleep(0.6)
        near = [o for o in detect('inspect') if math.hypot(o['cx'] - r['x_mm'], o['cy'] - r['y_mm']) < 60]
        if not near:
            print('   fruit not found at its last position (moved or taken away)'); continue
        o = min(near, key=lambda o: math.hypot(o['cx'] - r['x_mm'], o['cy'] - r['y_mm']))
        if not args.no_log:
            SH.log_scans([dict(o['scan'], fruit_id=r['fruit_id'])], match=False)
        print('   close-up: %s %.0f%% (spoiled %.0f%%, dark %.0f%%, hue %.0f deg)' % (
            o['cls'], 100 * o['conf'], 100 * o['scan']['p_rotten'], 100 * (o['scan']['dark_frac'] or 0), o['scan']['hue_deg'] or 0))
        try:
            ch = [c for c in SH.changes() if c['fruit_id'] == r['fruit_id']]
            if ch: print('   since last scan: ' + SH.describe_change(ch[0]))
        except Exception:  # noqa: BLE001
            pass
    move(*args.look, YAW0, 60, 'look')


try:
    arm.set_gripper_enable(True); arm.set_gripper_mode(0); arm.set_gripper_speed(3000)
    if cur[2] < args.look[2] - 5: move(cur[0], cur[1], args.look[2], YAW0, 60, 'up')
    move(*args.look, YAW0, 60, 'look'); time.sleep(0.8)
    RAIL = rail_mm()
    objs = detect('overview')
    if not args.no_log:
        SH.log_scans([o['scan'] for o in objs])
        for o in objs: o['fid'] = o['scan'].get('fruit_id')
    print('found %d objects:' % len(objs))
    for i, o in enumerate(objs, 1):
        print('  %d. %-16s %3.0f%%  at (%4.0f, %4.0f)  %3.0f mm tall  %3.0f x %3.0f mm%s' % (
            i, o['cls'], 100 * o['conf'], o['cx'], o['cy'], o['height'], o['length'], o['width'],
            '   fruit #%s' % o['fid'] if o.get('fid') else ''))
    if args.inspect:
        inspect(args.inspect)
    if not args.go:
        raise SystemExit(0)
    results = []
    if args.to_bin:
        if BIN_NAME not in DROP: raise SystemExit('no %s pose taught (manual.py save %s)' % (BIN_NAME, BIN_NAME))
        for o in objs:
            if not is_target(o): print('  leaving %s %.0f%% at (%.0f, %.0f) on the table' % (o['cls'], 100 * o['conf'], o['cx'], o['cy']))
        objs = [o for o in objs if is_target(o)]
        print('%s fruit to put in the %s: %d' % (args.target, 'bin' if args.target == 'ripe' else 'spoiled bin', len(objs)), flush=True)
    for i, o in enumerate(sorted(objs, key=lambda o: math.hypot(o['cx'], o['cy'])), 1):
        ok = pick(o, len(objs), i); results.append((o['cls'], ok))
        if not args.no_log and o.get('fid'):
            SH.log_scans([dict({k: v for k, v in o['scan'].items() if k != 'ts'}, source='pick', fruit_id=o['fid'],
                               action=(('to_spoiled_bin' if args.target == 'spoiled' else 'to_bin') if args.to_bin else 'picked') if ok else 'pick_failed')], match=False)
    move(*args.look, YAW0, 60, 'look')
    print('\nRESULT: %d of %d picked and held' % (sum(r for _, r in results), len(results)))
    for c, r in results: print('  %-16s %s' % (c, 'OK' if r else 'FAILED'))
finally:
    pipe.stop(); arm.disconnect()
