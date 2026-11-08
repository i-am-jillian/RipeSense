#!/usr/bin/env python3
"""Row search: FRONT first (camera looking forward, away from the rail, at bush / tree height), THEN the
FLOOR (camera straight down).  Rail is not moved.  Every object is classified with robot_fruit_model,
merged across views, logged to Tiger Data (scan_history.py) and shown, sorted by category, on a live page:

  http://<pi>:8091          live camera + categorized list + "inspect next" from Tiger

  ~/ripesense_env/bin/python row_search.py [--no_log] [--front_only] [--floor_only]
"""
import argparse, base64, json, math, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2, numpy as np, pyrealsense2 as rs, yaml
from xarm.wrapper import XArmAPI

ap = argparse.ArgumentParser()
ap.add_argument('--no_log', action='store_true')
ap.add_argument('--front_only', action='store_true')
ap.add_argument('--floor_only', action='store_true')
ap.add_argument('--port', type=int, default=8091)
ap.add_argument('--exit', action='store_true', help='exit when done (dashboard jobs)')
args = ap.parse_args()

CFG = yaml.safe_load(open(os.path.expanduser('~/config.yaml')))
XV = CFG['camera']['eef_to_color']; DK = CFG['camera'].get('depth_scale_correction', 1.0)
TABLE_Z = CFG['camera'].get('table_z_mm', -264.8)
MD = os.path.expanduser('~/models'); net = cv2.dnn.readNetFromONNX(MD + '/robot_fruit_model.onnx')
CLASSES = json.load(open(MD + '/robot_fruit_model_classes.json'))
MODEL = 'robot_fruit_model@' + time.strftime('%m%d-%H%M', time.localtime(os.path.getmtime(MD + '/robot_fruit_model.onnx')))
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import scan_history as SH
SESSION = time.strftime('%Y%m%d-%H%M%S'); CROPS = os.path.expanduser('~/scan_crops/' + SESSION)

# FRONT views: flange at radius R_FRONT, height Z_FRONT, tool pitched forward (camera ~15 deg below
# horizontal) and turned to each heading (deg, 0 = straight ahead = +x, away from the rail).
# (IK-checked 2026-11-08: needs the wrist pitch to turn ~133 deg from the straight-down pose)
R_FRONT, Z_FRONT, FRONT_PITCH = 450.0, 250.0, -60.0
FRONT_HEADINGS = [30.0, 0.0, -30.0]
FLIP_DJ = 145.0   # wrist-pitch allowance for entering/leaving the front views only
# FLOOR views: flange xyz, tool straight down
FLOOR_VIEWS = [(263.7, 36.7, 459.0), (300.0, 200.0, 459.0), (300.0, -160.0, 459.0)]

# motion gates (same as the collection program)
LIMITS = [(-360, 360), (-117.06, 116.02), (-219.04, 10.03), (-360, 360), (-97.0, 180.0), (-360, 360)]
MARGIN, MAX_DJ, MAX_DJ_REACH, MIN_R = 10.0, 60.0, 110.0, 220.0

MIN_CONF = 0.5          # below this -> 'Not sure' instead of a verdict
FRONT_REACH_MM = 900    # front views: ignore anything farther than this from the arm base (room, people, other tables)
CATS = [('unsure', 'Not sure -> look closer'), ('ripe', 'Ripe -> basket'), ('spoiled', 'Spoiled -> bin'), ('unripe', 'Unripe -> leave on plant'),
        ('other', 'Other fruit'), ('notfruit', 'Not fruit')]


def category(cls, conf=1.0):
    if conf < MIN_CONF: return 'unsure'
    if cls == 'not_fruit': return 'notfruit'
    if cls == 'other_fruit': return 'other'
    if cls.endswith('_rotten'): return 'spoiled'
    if cls.endswith('_unripe'): return 'unripe'
    return 'ripe'


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
    z = net.forward()[0]; p = np.exp(z - z.max()); p /= p.sum()
    return p


def fit_plane(Pmm, valid):
    X = Pmm[valid]; keep = np.abs(X[:, 2] - np.median(X[:, 2])) < 40
    for _ in range(6):
        A = np.c_[X[keep, 0], X[keep, 1], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(A, X[keep, 2], rcond=None)
        res = X[:, 2] - (coef[0] * X[:, 0] + coef[1] * X[:, 1] + coef[2])
        keep = np.abs(res) < max(4.0, 2.5 * np.std(res[keep]))
    return coef


# ---------------------------------------------------------------- live page
STATE = {'phase': 'starting', 'view': '', 'items': [], 'next': [], 'log': [], 'done': False, 'session': SESSION}
LATEST = {'jpg': None}
LOCK = threading.Lock()
PAGE = r"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Row Search</title><style>
:root{--bg:#101214;--card:#1a1d21;--fg:#e9ecef;--mut:#9aa3ab;--ripe:#3fb950;--spoiled:#f85149;--unripe:#d29922;--other:#58a6ff;--unsure:#bc8cff;--notfruit:#8b949e}
body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}
header{padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #2a2f35}
h1{font-size:18px;margin:0}.phase{padding:3px 10px;border-radius:12px;background:#264;font-weight:600}
.phase.floor{background:#246}.phase.done{background:#444}
main{display:grid;grid-template-columns:minmax(0,640px) minmax(0,1fr);gap:14px;padding:14px 16px}
@media(max-width:900px){main{grid-template-columns:1fr}}
img.cam{width:100%;border-radius:8px;background:#000}
.cat{background:var(--card);border-radius:8px;padding:8px 10px;margin-bottom:10px;border-left:5px solid}
.cat h2{font-size:15px;margin:2px 0 6px;display:flex;justify-content:space-between}
.it{display:flex;gap:8px;align-items:center;padding:4px 0;border-top:1px solid #262a2f}
.it img{width:44px;height:44px;object-fit:cover;border-radius:4px}.it small{color:var(--mut)}
.next{background:var(--card);border-radius:8px;padding:8px 10px}.next li{margin:4px 0}
.mut{color:var(--mut)}</style></head><body>
<header><h1>xArm Row Search</h1><span id=ph class=phase>…</span><span id=vw class=mut></span></header>
<main><div><img class=cam src="/stream"><div class="next"><b>Inspect next (Tiger Data)</b><ol id=nx class=mut><li>after the search</li></ol></div>
<pre id=lg class=mut style="white-space:pre-wrap;font-size:12px"></pre></div><div id=cats></div></main>
<script>
const CATS=[["unsure","Not sure → look closer"],["ripe","Ripe → basket"],["spoiled","Spoiled → bin"],["unripe","Unripe → leave on plant"],["other","Other fruit"],["notfruit","Not fruit"]];
function esc(s){return String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
async function tick(){try{const s=await (await fetch("/state")).json();
ph.textContent=s.phase.toUpperCase();ph.className="phase "+(s.done?"done":s.phase.startsWith("floor")?"floor":"");vw.textContent=s.view;
cats.innerHTML=CATS.map(([k,t])=>{const it=s.items.filter(i=>i.cat==k);return `<div class=cat style="border-color:var(--${k})"><h2><span>${t}</span><span>${it.length}</span></h2>`+
(it.length?it.map(i=>`<div class=it>${i.thumb?`<img src="data:image/jpeg;base64,${i.thumb}">`:""}<div><b>${i.fid?"#"+i.fid+" ":""}${esc(i.cls.replace(/_/g," "))}</b> ${Math.round(i.conf*100)}%<br><small>${i.where} · (${i.x}, ${i.y}, ${i.z}) mm · seen ${i.seen}×</small></div></div>`).join(""):"<small class=mut>none</small>")+"</div>"}).join("");
if(s.next.length)nx.innerHTML=s.next.map(n=>`<li><b>#${n.fid} ${esc(n.cls.replace(/_/g," "))}</b> — ${esc(n.why)}<br><small>since last scan: ${esc(n.change)}</small></li>`).join("");
lg.textContent=s.log.slice(-8).join("\n");}catch(e){} setTimeout(tick,700)}tick();
</script></body></html>"""


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
        body, ct = (json.dumps(STATE, default=str).encode(), 'application/json') if self.path.startswith('/state') else (PAGE.encode(), 'text/html')
        self.send_response(200); self.send_header('Content-Type', ct); self.send_header('Cache-Control', 'no-store'); self.end_headers()
        self.wfile.write(body)


threading.Thread(target=ThreadingHTTPServer(('0.0.0.0', args.port), H).serve_forever, daemon=True).start()


def show(img):
    ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75]); LATEST['jpg'] = enc.tobytes() if ok else LATEST['jpg']


def say(msg):
    print(msg, flush=True); STATE['log'].append(time.strftime('%H:%M:%S ') + msg)


# ---------------------------------------------------------------- hardware
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
try:
    r = arm.get_linear_track_registers()[1]; RAIL = float(r['pos']) if r.get('error') == 0 and r.get('on_zero') else None
except Exception:  # noqa: BLE001
    RAIL = None


def wait_motion():
    t0 = time.time(); time.sleep(0.3)
    while (arm.state == 1 or time.time() - t0 < 0.6) and time.time() - t0 < 60:
        show(np.asanyarray(align.process(pipe.wait_for_frames(5000)).get_color_frame().get_data()))
        if arm.error_code: raise SystemExit('ARM ERROR %s' % arm.error_code)


def go(x, y, z, roll, pitch, yaw, reach_dj=MAX_DJ_REACH, wrist_only=False):
    """Gated joint-space move to a flange pose (deg).  Returns False (no motion) if refused."""
    if math.hypot(x, y) < MIN_R: return False
    code, ang = arm.get_inverse_kinematics([x, y, z, roll, pitch, yaw], input_is_radian=False, return_is_radian=False)
    if code != 0: return False
    if any(not (lo + MARGIN <= v <= hi - MARGIN) for v, (lo, hi) in zip(ang[:6], LIMITS)): return False
    cur = arm.angles[:6]; dj = [abs(a - b) for a, b in zip(ang[:6], cur)]
    if wrist_only:                                   # in-place wrist untwist: only J6 may change
        if max(dj[:5]) > 8 or dj[5] > 185:
            say('  wrist untwist refused (joint change %s deg)' % [round(d) for d in dj]); return False
    elif max(dj[0], dj[3], dj[5]) > MAX_DJ or max(dj[1], dj[2], dj[4]) > reach_dj:
        say('  pose refused by swing gate (joint change %s deg)' % [round(d) for d in dj]); return False
    if arm.set_servo_angle(angle=list(ang[:6]), speed=25, mvacc=200, wait=False) != 0: return False
    wait_motion(); return True


def capture():
    code, pose = arm.get_position(is_radian=True)
    Tbc = T(rpy_to_R(*pose[3:]), np.array(pose[:3]) / 1000) @ T(rpy_to_R(*XV[3:]), XV[:3])
    deps = []
    for _ in range(10):
        f = align.process(pipe.wait_for_frames(5000))
        deps.append(np.asanyarray(hole.process(temp_f.process(spat.process(f.get_depth_frame()))).get_data()).astype(np.float32) * sc * DK)
        color = np.asanyarray(f.get_color_frame().get_data()).copy()
    D = np.median(np.array(deps), 0)
    Pmm = (np.stack([(us - intr.ppx) / intr.fx * D, (vs - intr.ppy) / intr.fy * D, D, np.ones_like(D)], -1).reshape(-1, 4) @ Tbc.T)[:, :3].reshape(480, 640, 3) * 1000
    return pose, Tbc, D, Pmm, color


def blobs_floor(D, Pmm):
    valid = (D > 0.15) & (D < 1.3)
    coef = fit_plane(Pmm, valid)
    zb = Pmm[..., 2] - (coef[0] * Pmm[..., 0] + coef[1] * Pmm[..., 1] + coef[2])
    m = valid & (zb > 12) & (zb < 150)
    return m, zb, lambda x, y: float(coef[0] * x + coef[1] * y + coef[2])


def blobs_front(D, Pmm):
    """Forward view: anything within 1.1 m that sticks up from the table/floor, cut at depth edges so a fruit
    is separated from the bush, wall or stand behind it."""
    valid = (D > 0.2) & (D < 1.4)
    zb = Pmm[..., 2] - TABLE_Z
    gy, gx = np.gradient(cv2.medianBlur((D * 1000).astype(np.float32), 5))
    edge = np.hypot(gx, gy) > 12                                        # > 12 mm depth jump per pixel
    m = valid & (zb > 12) & ~edge
    return m, zb, lambda x, y: TABLE_Z


def detect(where, label):
    pose, Tbc, D, Pmm, color = capture()
    m, zb, base_z = (blobs_front if where == 'front' else blobs_floor)(D, Pmm)
    m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    dets, vis = [], color.copy()
    for j in range(1, n):
        x, y, w, h, area = st[j]
        if area < 300 or area > 60000: continue
        pts = Pmm[lab == j]
        ext = np.ptp(pts, axis=0)
        if ext.max() > 160 or ext.max() < 15: continue                 # walls / table edges / specks
        hgt = float(np.percentile(zb[lab == j], 95))
        if where == 'floor' and not (12 < hgt < 130): continue
        if where == 'front' and math.hypot(*np.median(pts[:, :2], 0)) > FRONT_REACH_MM: continue
        edge = bool(x <= 2 or y <= 2 or x + w >= 638 or y + h >= 478)
        pad = int(0.12 * max(w, h)); crop = color[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        p = classify(crop); k = int(p.argmax()); cls, conf = CLASSES[k], float(p[k])
        cx, cy, cz = (float(v) for v in np.median(pts, 0))
        hue, sat, val, dark = SH.colour_stats(color, (lab == j).astype(np.uint8))
        os.makedirs(CROPS, exist_ok=True); cp = '%s/%s_%s_%d.jpg' % (CROPS, time.strftime('%H%M%S'), where, len(dets) + 1)
        cv2.imwrite(cp, crop)
        ok, th = cv2.imencode('.jpg', cv2.resize(crop, (64, 64)), [cv2.IMWRITE_JPEG_QUALITY, 70])
        top3 = np.argsort(p)[::-1][:3]
        scan = dict(session=SESSION, source=where, x_mm=cx, y_mm=cy, top_z_mm=float(np.percentile(pts[:, 2], 95)),
                    table_z_mm=base_z(cx, cy), rail_mm=RAIL,
                    cam_pose=dict(zip(['x', 'y', 'z'], [round(v, 1) for v in pose[:3]]), **dict(zip(['roll', 'pitch', 'yaw'], [round(math.degrees(v), 2) for v in pose[3:]]))),
                    cam_dist_mm=float(np.median(D[lab == j]) * 1000), height_mm=hgt,
                    length_mm=float(max(ext[0], ext[1])), width_mm=float(min(ext[0], ext[1])), area_px=int(area),
                    hue_deg=hue, sat=sat, val=val, dark_frac=dark, spectral=None, model=MODEL, cls=cls, conf=conf,
                    p_rotten=float(sum(p[i] for i, c in enumerate(CLASSES) if c.endswith('_rotten'))),
                    probs={CLASSES[i]: round(float(p[i]), 4) for i in top3}, crop_path=cp)
        dets.append(dict(x=cx, y=cy, z=cz, cls=cls, conf=conf, edge=edge, where=where, scan=scan,
                         thumb=base64.b64encode(th.tobytes()).decode() if ok else None))
        col = {'ripe': (80, 185, 63), 'spoiled': (73, 81, 248), 'unripe': (34, 153, 210), 'other': (255, 166, 88), 'unsure': (255, 140, 188)}.get(category(cls, conf), (150, 150, 150))
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
        cv2.putText(vis, '%s %.0f%%' % (cls, 100 * conf), (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.putText(vis, label, (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    show(vis); cv2.imwrite('/tmp/row_search_%s.jpg' % label.replace(' ', '_').replace('/', '-'), vis)
    return dets


MERGED = []


def merge(dets):
    for d in sorted(dets, key=lambda d: (d['edge'], -d['conf'])):
        for f in MERGED:
            if math.dist((f['x'], f['y'], f['z']), (d['x'], d['y'], d['z'])) < 50:
                f['seen'] += 1
                if (not d['edge'] and f['edge']) or (d['edge'] == f['edge'] and d['conf'] > f['conf']):
                    f.update({k: v for k, v in d.items() if k != 'seen'})
                break
        else:
            MERGED.append(dict(d, seen=1))
    publish()


def publish():
    STATE['items'] = [dict(cat=category(f['cls'], f['conf']), cls=f['cls'], conf=round(f['conf'], 3), where=f['where'],
                           x=round(f['x']), y=round(f['y']), z=round(f['z']), seen=f['seen'], fid=f.get('fid'), thumb=f['thumb'])
                      for f in sorted(MERGED, key=lambda f: -f['conf'])]


try:
    code, cur = arm.get_position()
    say('start at flange (%.0f, %.0f, %.0f); rail %s' % (cur[0], cur[1], cur[2], '%.0f mm' % RAIL if RAIL is not None else 'not homed (fixed)'))
    if cur[2] < 300 and not go(cur[0], cur[1], 459, 180, 0, cur[5]):
        raise SystemExit('cannot rise to a safe height from here')
    # start from the standard look pose with the standard wrist configuration (a previous run may have left the
    # wrist rolled ~180 deg, which makes every front pose fail the swing gate)
    code, home = arm.get_inverse_kinematics([*FLOOR_VIEWS[0], 180, 0, 0], input_is_radian=False, return_is_radian=False)
    code2, cur = arm.get_position()
    cj = arm.angles[:6]; dj = [abs(x - y) for x, y in zip(home[:6], cj)] if code == 0 else [0] * 6
    if code == 0 and max(dj) > 5 and cur[2] > 400 and max(dj[1], dj[2]) < 45 and dj[0] < 90:
        say('returning to the start pose (joint change %s deg)' % [round(d) for d in dj])
        if arm.set_servo_angle(angle=list(home[:6]), speed=20, mvacc=150, wait=False) == 0: wait_motion()
    code, cur = arm.get_position()
    if abs(abs(cur[3]) - 180) < 10 and abs(cur[4]) < 10 and abs(cur[5]) > 5:
        say('untwisting the wrist from %.0f to 0 deg (in place)' % cur[5])
        if not go(cur[0], cur[1], cur[2], 180, 0, 0, wrist_only=True):
            raise SystemExit('cannot untwist the wrist here')
    if not args.floor_only:
        for i, hd in enumerate(FRONT_HEADINGS, 1):
            STATE['phase'] = 'front search'; STATE['view'] = 'view %d/%d, heading %+.0f deg' % (i, len(FRONT_HEADINGS), hd)
            a = math.radians(hd)
            if not go(R_FRONT * math.cos(a), R_FRONT * math.sin(a), Z_FRONT, 180, FRONT_PITCH, hd, FLIP_DJ if i == 1 else MAX_DJ_REACH):
                say('front heading %+.0f: unreachable or refused, skipped' % hd); continue
            time.sleep(0.6); d = detect('front', 'FRONT %+.0f' % hd); merge(d)
            say('front heading %+.0f: %d objects (%s)' % (hd, len(d), ', '.join(x['cls'] for x in d) or '-'))
        # back to a top-down pose before the floor views (two gated steps: rise, then turn the wrist down)
        if not go(*FLOOR_VIEWS[0], 180, 0, 0, FLIP_DJ):
            raise SystemExit('cannot return to the floor view')
    if not args.front_only:
        for i, v in enumerate(FLOOR_VIEWS, 1):
            STATE['phase'] = 'floor search'; STATE['view'] = 'view %d/%d' % (i, len(FLOOR_VIEWS))
            if not go(*v, 180, 0, 0) and not (go(*FLOOR_VIEWS[0], 180, 0, 0) and go(*v, 180, 0, 0)):
                say('floor view %d: unreachable or refused, skipped' % i); continue
            time.sleep(0.6); d = detect('floor', 'FLOOR %d' % i); merge(d)
            say('floor view %d: %d objects (%s)' % (i, len(d), ', '.join(x['cls'] for x in d) or '-'))
        go(*FLOOR_VIEWS[0], 180, 0, 0)
    STATE['phase'] = 'logging'; STATE['view'] = ''
    if not args.no_log and MERGED:
        rows = SH.log_scans([dict(f['scan']) for f in MERGED])
        for f, r in zip(MERGED, rows): f['fid'] = r.get('fruit_id')
        publish()
        try:
            STATE['next'] = [dict(fid=r['fruit_id'], cls=r['cls'], why=SH.reasons(r), change=SH.describe_change(r))
                             for r in SH.inspect_next(3)]
        except Exception as e:  # noqa: BLE001
            say('Tiger query failed: %s' % e)
    counts = {t: sum(1 for f in MERGED if category(f['cls'], f['conf']) == k) for k, t in CATS}
    say('DONE: ' + ', '.join('%s %d' % (t, c) for t, c in counts.items()))
    for f in sorted(MERGED, key=lambda f: (category(f['cls'], f['conf']), -f['conf'])):
        say('  #%s %-17s %3.0f%%  %-5s (%4.0f, %4.0f, %4.0f) mm  seen %d' % (
            f.get('fid') or '-', f['cls'], 100 * f['conf'], f['where'], f['x'], f['y'], f['z'], f['seen']))
    for n in STATE['next']:
        say('INSPECT NEXT #%s %s -- %s | %s' % (n['fid'], n['cls'], n['why'], n['change']))
    STATE['phase'] = 'done'; STATE['done'] = True; STATE['finished'] = time.time()
    json.dump(STATE, open('/tmp/row_state.tmp', 'w'), default=str); os.replace('/tmp/row_state.tmp', '/tmp/row_state.json')
    json.dump([{k: v for k, v in f.items() if k not in ('thumb', 'scan')} for f in MERGED], open('/tmp/row_map.json', 'w'), indent=1, default=str)
    # keep the page and the camera live after the search
    if args.exit:
        raise SystemExit(0)
    say('page stays live; Ctrl-C to stop')
    while True:
        show(np.asanyarray(align.process(pipe.wait_for_frames(5000)).get_color_frame().get_data())); time.sleep(0.05)
finally:
    pipe.stop(); arm.disconnect()
