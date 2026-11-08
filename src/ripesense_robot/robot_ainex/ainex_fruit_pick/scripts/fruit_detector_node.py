#!/usr/bin/env python3
# encoding: utf-8
"""RipeSense fruit detector for the Hiwonder AiNex.

Finds fruit on a plain, light floor by colour (fruit is saturated or dark, the
floor is not), runs the RipeSense image model on each fruit crop and publishes
the best *pickable* fruit in the same ObjectsInfo format the stock colour
detector uses, so Hiwonder's approach/pick logic can follow it unchanged.

Uses only our own trained models (ripesense models/artifacts -> ~/ripesense_models):
  * banana_model.onnx (5 classes, notebooks/train_banana_model_colab.ipynb) when present;
  * otherwise image_model.onnx (fruit type + fresh/rotten) + stage_model.onnx (banana stages).
Each blob gets a verdict and an action:
  ripe -> 'basket', rotten/spoiled -> 'bin', overripe -> ~overripe_action (bin), unripe -> 'leave'.
The target is the closest blob whose action is basket or bin.
An optional CLIP "is it a fruit?" check exists (~use_gate, off by default).

Topics
  sub  /camera/image_raw                      sensor_msgs/Image (rgb8)
  pub  /fruit_detection/target                ainex_interfaces/ObjectsInfo (0 or 1 item)
  pub  /fruit_detection/all                   ainex_interfaces/ObjectsInfo (every fruit, label=verdict)
  pub  /fruit_detection/image_result          sensor_msgs/Image (annotated, view on :8080)
Services
  /fruit_detection/start, /fruit_detection/stop   std_srvs/Empty
HTTP (port ~http_port, default 8090) for the laptop RipeSense /robot page
  /status      JSON: detections, target, pick-node phase, fps
  /target.jpg  padded crop of the current target fruit
"""
import json
import os
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import queue
import threading
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_srvs.srv import Empty, EmptyResponse
from std_msgs.msg import String
from ainex_interfaces.msg import ObjectInfo, ObjectsInfo

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def softmax(x):
    p = np.exp(x - x.max())
    return p / p.sum()


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-9)


def parse_class(name):
    """'rotten_lime' -> ('lime', True); 'fresh_lime' -> ('lime', False)."""
    if name.startswith('rotten_'):
        return name[len('rotten_'):], True
    if name.startswith('fresh_'):
        return name[len('fresh_'):], False
    return name, None


class Classifier:
    """MobileNetV3 ONNX from RipeSense train_image.py, run with cv2.dnn
    (verified identical to onnxruntime on the robot: max prob diff 3.5e-4)."""

    def __init__(self, onnx_path, classes_path):
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        with open(classes_path) as f:
            self.classes = json.load(f)

    @staticmethod
    def prepare(bgr):
        h, w = bgr.shape[:2]
        s = 240.0 / min(h, w)
        img = cv2.resize(bgr, (max(224, round(w * s)), max(224, round(h * s))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        y0, x0 = (h - 224) // 2, (w - 224) // 2
        img = img[y0:y0 + 224, x0:x0 + 224]
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)

    def probs(self, bgr):
        self.net.setInput(self.prepare(bgr))
        return softmax(self.net.forward()[0])


class ClipGate:
    """Zero-shot "fruit vs. not fruit" on a crop. Text embeddings are precomputed on
    the laptop (clip_gate_text.npz: fruit names + negatives such as person, hand,
    phone screen, box, carpet), so only the vision tower runs here."""
    MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

    def __init__(self, mdir, threads=2):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        o.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(os.path.join(mdir, 'vision_model_quantized.onnx'), o,
                                         providers=['CPUExecutionProvider'])
        d = np.load(os.path.join(mdir, 'clip_gate_text.npz'))
        self.text, self.labels, self.n_fruit = d['text'], [str(x) for x in d['labels']], int(d['n_fruit'])

    def restrict(self, fruits):
        """Gate on these fruit names only (e.g. ['apple']): "is it an apple?" instead of "any fruit?"."""
        idx = [i for i, l in enumerate(self.labels[:self.n_fruit]) if l in fruits]
        if idx:
            self.fruit_idx = np.array(idx)

    fruit_idx = None

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        s = 224.0 / min(h, w)
        im = cv2.resize(bgr, (max(224, round(w * s)), max(224, round(h * s))), interpolation=cv2.INTER_CUBIC)
        h, w = im.shape[:2]
        y, x = (h - 224) // 2, (w - 224) // 2
        im = im[y:y + 224, x:x + 224]
        a = ((cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - self.MEAN) / self.STD)
        v = self.sess.run(['image_embeds'], {'pixel_values': a.transpose(2, 0, 1)[None]})[0][0]
        sim = self.text @ (v / np.linalg.norm(v))
        cand = self.fruit_idx if self.fruit_idx is not None else np.arange(self.n_fruit)
        fi, ni = int(cand[sim[cand].argmax()]), self.n_fruit + int(sim[self.n_fruit:].argmax())
        return {'margin': float(sim[fi] - sim[ni]), 'clip_fruit': self.labels[fi], 'clip_other': self.labels[ni]}


class FruitDetectorNode:
    def __init__(self, name):
        rospy.init_node(name)
        self.name = name
        mdir = os.path.expanduser(rospy.get_param('~models_dir', '~/ripesense_models'))
        self.rot_thresh = rospy.get_param('~rot_thresh', 0.35)      # P(rotten) below this = fresh
        self.conf_thresh = rospy.get_param('~conf_thresh', 0.70)    # fruit-type confidence needed
        self.sat_thresh = rospy.get_param('~sat_thresh', 70)        # HSV S above this = not floor
        self.dark_thresh = rospy.get_param('~dark_thresh', 0)       # HSV V below this = candidate (catches brown/black spoiled fruit; CLIP gate rejects other dark things)
        self.stage_thresh = rospy.get_param('~stage_thresh', 0.6)   # banana stage confidence needed, else 'unsure'
        self.overripe_action = rospy.get_param('~overripe_action', 'bin')   # bin | basket | leave
        self.elong_min = rospy.get_param('~elong_min', 1.6)         # long/short ratio above which we align to the long axis
        self.max_elong = rospy.get_param('~max_elong', 0.0)          # >0: drop blobs longer than this ratio (round fruit like apples)
        self.border = rospy.get_param('~border_px', 6)               # drop blobs cut off by the image edge
        self.min_area = rospy.get_param('~display_min_area', 300)   # px in 640x480: smallest blob classified + shown
        self.target_min_area = rospy.get_param('~min_area', 300)    # smallest blob the robot may go for
        self.max_area_frac = rospy.get_param('~max_area_frac', 0.5)
        self.reclassify_s = rospy.get_param('~reclassify_s', 0.3)
        self.max_classify = rospy.get_param('~max_classify_per_frame', 4)
        self.allowed = [f.strip() for f in rospy.get_param('~fruits', '').split(',') if f.strip()]  # '' = any
        self.ignore_bottom = rospy.get_param('~ignore_bottom_frac', 0.0)  # mask robot's own feet/hands
        # training-data collection: save every blob crop as class <collect_label>
        self.collect_label = rospy.get_param('~collect_label', '').strip()
        self.collect_every = rospy.get_param('~collect_every_s', 0.7)
        self.collect_max = rospy.get_param('~collect_max', 400)
        self.collect_n, self.collect_t = 0, 0.0
        if self.collect_label:
            self.collect_session = time.strftime('%Y%m%d-%H%M%S')
            self.collect_dir = os.path.expanduser(os.path.join(rospy.get_param('~collect_dir', '~/ripesense_collect'),
                                                               self.collect_label))
            os.makedirs(self.collect_dir, exist_ok=True)
            rospy.logwarn('%s: COLLECT MODE -> every blob saved as "%s" in %s', name, self.collect_label, self.collect_dir)

        self.image_model = Classifier(os.path.join(mdir, 'image_model.onnx'),
                                      os.path.join(mdir, 'image_model_classes.json'))
        self.parsed = [parse_class(c) for c in self.image_model.classes]
        self.stage_model = None
        sp = os.path.join(mdir, 'stage_model.onnx')
        if os.path.exists(sp):
            self.stage_model = Classifier(sp, os.path.join(mdir, 'stage_model_classes.json'))
        # Our own 5-class model from notebooks/train_banana_model_colab.ipynb (preferred when present):
        # not_banana / overripe / ripe / rotten / unripe
        self.banana_model = None
        bp = os.path.join(mdir, 'banana_model.onnx')
        if os.path.exists(bp) and rospy.get_param('~use_banana_model', True):
            self.banana_model = Classifier(bp, os.path.join(mdir, 'banana_model_classes.json'))
        rospy.loginfo('%s: models from %s: %s', name, mdir,
                      'banana_model %s' % self.banana_model.classes if self.banana_model else
                      'image_model + stage_model' if self.stage_model else 'image_model')

        # CLIP fruit gate (background thread)
        self.gate_margin = rospy.get_param('~gate_margin', 0.005)   # CLIP fruit-vs-other margin needed
        self.gate_ttl = rospy.get_param('~gate_ttl_s', 4.0)          # re-check a tracked blob after this
        self.gate = None
        if rospy.get_param('~use_gate', False):   # optional CLIP check (downloaded model); off: we use our own trained models
            try:
                self.gate = ClipGate(mdir, rospy.get_param('~gate_threads', 2))
                if self.allowed:
                    self.gate.restrict(self.allowed)
                rospy.loginfo('%s: CLIP fruit gate on (margin %.3f)', name, self.gate_margin)
            except Exception as e:  # noqa: BLE001
                rospy.logerr('%s: CLIP gate unavailable (%s) -- NOTHING will be pickable', name, e)
        self.gate_results = {}       # track id -> {'ok', 'margin', 'clip_fruit', 'clip_other', 't'}
        self.gate_job = None         # (track id, crop) waiting for the worker
        self.gate_busy = False
        self.gate_cv = threading.Condition()
        self.next_id = 0
        self.gate_done = 0
        if self.gate is not None:
            threading.Thread(target=self.gate_worker, daemon=True).start()

        self.tracks = []          # [{'c': (x, y), 't': time, 'res': dict}]
        self.last_target = None
        self.lock_id, self.lock_act, self.lock_last, self.lock_seen = None, None, None, 0.0
        self.vote_n = rospy.get_param('~vote_n', 7)          # classifications remembered per fruit
        self.vote_min = rospy.get_param('~vote_min', 3)      # votes needed before acting
        self.track_keep_s = rospy.get_param('~track_keep_s', 1.0)
        self.hold_s = rospy.get_param('~hold_s', 1.0)        # keep publishing a lost target this long
        self.lock_radius = rospy.get_param('~lock_radius_px', 150)
        self.detect_on = rospy.get_param('~start', True)
        self.lock = threading.Lock()
        self.q = queue.Queue(maxsize=1)

        self.target_pub = rospy.Publisher('/fruit_detection/target', ObjectsInfo, queue_size=1)
        self.all_pub = rospy.Publisher('/fruit_detection/all', ObjectsInfo, queue_size=1)
        self.image_pub = rospy.Publisher('/fruit_detection/image_result', Image, queue_size=1)
        rospy.Service('/fruit_detection/start', Empty, self.start_srv)
        rospy.Service('/fruit_detection/stop', Empty, self.stop_srv)
        rospy.Subscriber(rospy.get_param('~image_topic', '/camera/image_raw'), Image, self.image_cb,
                         queue_size=1, buff_size=2 ** 24)

        # status for the laptop dashboard
        self.status = {'detections': [], 'target': None, 'pick': None, 'detecting': self.detect_on, 'fps': 0.0, 't': 0.0}
        self.target_jpg = None
        self.fps_t, self.fps_n = time.time(), 0
        rospy.Subscriber('/fruit_pick/status', String, self.pick_status_cb, queue_size=5)
        self.start_http(rospy.get_param('~http_port', 8090))

    def pick_status_cb(self, msg):
        try:
            self.status['pick'] = json.loads(msg.data)
        except ValueError:
            pass

    def start_http(self, port):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith('/status'):
                    body, ctype = json.dumps(node.status).encode(), 'application/json'
                elif self.path.startswith('/target.jpg') and node.target_jpg is not None:
                    body, ctype = node.target_jpg, 'image/jpeg'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', ctype)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = ThreadingHTTPServer(('0.0.0.0', port), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        rospy.loginfo('%s: status server on :%d', self.name, port)

    # ---------- services ----------
    def start_srv(self, _):
        with self.lock:
            self.detect_on = True
            self.tracks, self.last_target = [], None
            self.lock_id, self.lock_last = None, None
        self.status['detecting'] = True
        rospy.loginfo('%s start', self.name)
        return EmptyResponse()

    def stop_srv(self, _):
        with self.lock:
            self.detect_on = False
        self.status.update(detecting=False, detections=[], target=None)
        self.target_pub.publish(ObjectsInfo([]))
        rospy.loginfo('%s stop', self.name)
        return EmptyResponse()

    def image_cb(self, msg):
        rgb = np.ndarray((msg.height, msg.width, 3), np.uint8, msg.data)
        bgr = rgb[:, :, ::-1].copy() if msg.encoding == 'rgb8' else rgb.copy()
        try:
            self.q.get_nowait()
        except queue.Empty:
            pass
        self.q.put_nowait(bgr)

    # ---------- perception ----------
    def segment(self, bgr):
        """Fruit-candidate blobs on a plain light floor -> list of (contour, rect)."""
        H, W = bgr.shape[:2]
        small = cv2.resize(bgr, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = ((hsv[..., 1] > self.sat_thresh) | (hsv[..., 2] < self.dark_thresh)).astype(np.uint8) * 255
        if self.ignore_bottom > 0:
            mask[int(mask.shape[0] * (1 - self.ignore_bottom)):] = 0
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        blobs = []
        for c in cnts:
            c = c * 2  # back to full resolution
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area_frac * W * H:
                continue
            x, y, w, h = cv2.boundingRect(c)
            blobs.append((c, (x, y, w, h), area))
        blobs.sort(key=lambda b: -b[2])
        return blobs[:10]

    def classify(self, bgr, rect):
        """RipeSense verdict for one crop -> res with 'verdict' and 'action'.
        verdict: ripe / unripe / overripe / spoiled / fresh (non-banana) / unsure
        action : basket / bin / leave / None (unsure or not an allowed fruit)"""
        x, y, w, h = rect
        H, W = bgr.shape[:2]
        px, py = int(w * 0.12), int(h * 0.12)
        crop = bgr[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)]
        if self.banana_model is not None:
            return self.classify_banana(crop)
        p = self.image_model.probs(crop)
        fruit_p = {}
        for (f, _), pi in zip(self.parsed, p):
            fruit_p[f] = fruit_p.get(f, 0.0) + float(pi)
        fruit = max(fruit_p, key=fruit_p.get)
        fp = [(rot, float(pi)) for (f, rot), pi in zip(self.parsed, p) if f == fruit and rot is not None]
        tot = sum(pi for _, pi in fp)
        rot_prob = sum(pi for rot, pi in fp if rot) / tot if tot > 0 else 1.0
        res = {'fruit': fruit, 'conf': fruit_p[fruit], 'rot': rot_prob, 'stage': None, 'stage_p': None}
        if fruit == 'banana' and self.stage_model is not None:
            sp = self.stage_model.probs(crop)
            res['stage'] = self.stage_model.classes[int(sp.argmax())]
            res['stage_p'] = float(sp.max())
        if res['conf'] < self.conf_thresh:
            verdict = 'unsure'
        elif rot_prob >= self.rot_thresh:
            verdict = 'spoiled'
        elif res['stage'] is not None:
            verdict = res['stage'] if res['stage_p'] >= self.stage_thresh else 'unsure'
        else:
            verdict = 'fresh'
        res['verdict'] = verdict
        res['action'] = {'ripe': 'basket', 'fresh': 'basket', 'spoiled': 'bin', 'unripe': 'leave',
                         'overripe': self.overripe_action}.get(verdict)
        res['target_ok'] = not (self.allowed and fruit not in self.allowed)
        return res

    def classify_banana(self, crop):
        p = self.banana_model.probs(crop)
        cls = self.banana_model.classes[int(p.argmax())]
        conf = float(p.max())
        probs = dict(zip(self.banana_model.classes, map(float, p)))
        res = {'fruit': 'banana' if cls != 'not_banana' else 'other', 'conf': conf,
               'rot': probs.get('rotten', 0.0), 'stage': None if cls in ('not_banana', 'rotten') else cls,
               'stage_p': conf}
        if cls == 'not_banana':
            verdict = 'not fruit'
        elif conf < self.stage_thresh:
            verdict = 'unsure'
        else:
            verdict = 'spoiled' if cls == 'rotten' else cls
        res['verdict'] = verdict
        res['action'] = {'ripe': 'basket', 'spoiled': 'bin', 'unripe': 'leave',
                         'overripe': self.overripe_action}.get(verdict)
        if verdict == 'not fruit':
            res['gate'] = 'not_fruit'
        return res

    def gate_worker(self):
        while not rospy.is_shutdown():
            with self.gate_cv:
                while self.gate_job is None:
                    self.gate_cv.wait(0.5)
                    if rospy.is_shutdown():
                        return
                tid, crop, box = self.gate_job
                self.gate_job, self.gate_busy = None, True
            t0 = time.time()
            try:
                g = self.gate(crop)
                g['ok'] = g['margin'] > self.gate_margin
                g['t'] = time.time()
                g['box'] = box
                self.gate_results[tid] = g
                self.gate_done += 1
                self.status['gate'] = {'done': self.gate_done, 'ms': round(1000 * (g['t'] - t0)),
                                       'last': '%s %+.3f' % (g['clip_fruit'] if g['ok'] else g['clip_other'], g['margin'])}
            except Exception as e:  # noqa: BLE001
                rospy.logwarn_throttle(10, 'gate failed: %s' % e)
            finally:
                self.gate_busy = False

    def lookup(self, c, now, size=0):
        best, bd = None, max(50.0, 0.6 * size)
        for t in self.tracks:
            d = np.hypot(t['c'][0] - c[0], t['c'][1] - c[1])
            if d < bd:
                best, bd = t, d
        return best

    @staticmethod
    def long_axis_angle(cnt):
        """Angle (deg) of the blob's long axis vs. image x-axis, in the same minAreaRect
        convention Hiwonder's block code uses; 0 = lying crosswise in front of the robot."""
        (_, _), (rw, rh), a = cv2.minAreaRect(cnt)
        ang = a if rw >= rh else a - 90
        if ang > 90:
            ang -= 180
        elif ang <= -90:
            ang += 180
        elong = max(rw, rh) / max(1.0, min(rw, rh))
        return int(round(ang)), elong

    def process(self, bgr):
        H, W = bgr.shape[:2]
        now = time.time()
        blobs = self.segment(bgr)
        n_classified = 0
        dets, new_tracks, gate_wanted = [], [], []
        for cnt, rect, area in blobs:
            (cx, cy), _, _ = cv2.minAreaRect(cnt)
            t = self.lookup((cx, cy), now, max(rect[2], rect[3]))
            if t is None and self.lock_id is not None and self.lock_last is not None and \
                    np.hypot(cx - self.lock_last[0], cy - self.lock_last[1]) < self.lock_radius:
                t = next((tr for tr in self.tracks if tr['id'] == self.lock_id), None)   # locked fruit jumped (robot stepped)
            if t is not None and now - t['t'] < self.reclassify_s:
                res, ts = t['res'], t['t']
            elif n_classified < self.max_classify:
                res, ts = self.classify(bgr, rect), now
                n_classified += 1
            elif t is not None:
                res, ts = t['res'], t['t']
            else:
                continue
            if t is not None:
                tid, votes = t['id'], t['votes']
            else:
                tid, self.next_id, votes = self.next_id, self.next_id + 1, []
            if ts == now:                                   # classified this frame: add a vote
                votes = (votes + [(res['fruit'], res['verdict'], res['action'])])[-self.vote_n:]
            new_tracks.append({'c': (cx, cy), 't': ts, 'res': res, 'id': tid, 'votes': votes, 'seen': now,
                               'rect': rect})
            g = self.gate_results.get(tid)
            if g is None:
                for og in list(self.gate_results.values()):
                    if iou(og['box'], rect) > 0.4:
                        g = self.gate_results[tid] = og
                        break
            res = dict(res)
            # majority vote over the last few classifications of this track, so the verdict
            # cannot flip fresh <-> spoiled from frame to frame while the robot walks
            if votes:
                (vf, vv, va), k = Counter(votes).most_common(1)[0]
                if k >= 0.6 * len(votes) and len(votes) >= self.vote_min:
                    res['fruit'], res['verdict'], res['action'] = vf, vv, va
                else:
                    res['verdict'], res['action'] = 'deciding', None
            res['tid'] = tid
            res['angle'], res['elong'] = self.long_axis_angle(cnt)
            why = []
            if not res.get('target_ok', True):
                why.append('not %s' % '/'.join(self.allowed))
            if self.max_elong > 0 and res['elong'] > self.max_elong:
                why.append('wrong shape')
            if area < self.target_min_area:
                why.append('too small')
            bx, by, bw, bh = rect
            if self.border and (bx < self.border or by < self.border or bx + bw > W - self.border or by + bh > H - self.border):
                why.append('at edge')                  # partly out of view: shown, never targeted
            res['not_target'] = ', '.join(why)
            res['act'] = None                       # action actually allowed this frame
            if self.gate is None:
                res.setdefault('gate', 'off')
                res['act'] = None if res['not_target'] else res['action']
            elif g is None or now - g['t'] > self.gate_ttl:
                res['gate'] = 'checking'
                gate_wanted.append((0 if res['action'] in ('basket', 'bin') else 1, -area, tid, rect))
                if g is not None and g['ok']:      # stale but was fruit: keep acting on it while re-checking
                    res['gate'], res['act'] = 'fruit', (None if res['not_target'] else res['action'])
            elif not g['ok']:
                res.update(gate='not_fruit', verdict='not fruit', action=None, clip=g['clip_other'])
            else:
                res.update(gate='fruit', clip=g['clip_fruit'], act=None if res['not_target'] else res['action'])
            dets.append((int(cx), int(cy), rect, res))
        seen_ids = {t['id'] for t in new_tracks}
        new_tracks += [t for t in self.tracks if t['id'] not in seen_ids and now - t['seen'] < self.track_keep_s]
        self.tracks = new_tracks
        live = {t['id'] for t in new_tracks}
        self.gate_results = {k: v for k, v in self.gate_results.items() if k in live}
        if gate_wanted and self.gate is not None and not self.gate_busy and self.gate_job is None:
            _, _, tid, (x, y, w, h) = min(gate_wanted)
            px, py = int(w * 0.12), int(h * 0.12)
            crop = bgr[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)].copy()
            with self.gate_cv:
                self.gate_job = (tid, crop, (x, y, w, h))
                self.gate_cv.notify()

        if self.collect_label and self.collect_n < self.collect_max and now - self.collect_t >= self.collect_every:
            self.collect_t = now
            for d in dets:
                if 'at edge' in d[3].get('not_target', ''):
                    continue                           # partly out of view (and the robot's own feet/hands)
                x, y, w, h = d[2]
                px, py = int(w * 0.12), int(h * 0.12)
                crop = bgr[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)]
                cv2.imwrite(os.path.join(self.collect_dir, '%s__%04d.jpg' % (self.collect_session, self.collect_n)), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                self.collect_n += 1
            self.status['collect'] = {'label': self.collect_label, 'saved': self.collect_n, 'max': self.collect_max}

        # target: keep the previous target if still there, else the closest (lowest in image)
        # fruit that has somewhere to go (ripe -> basket, spoiled -> bin). Unripe stays put.
        movable = [d for d in dets if d[3]['act'] in ('basket', 'bin')]
        target, held = None, False
        if self.lock_id is not None:
            mine = [d for d in dets if d[3]['tid'] == self.lock_id]
            if mine and mine[0][3]['verdict'] not in ('not fruit', 'unripe'):
                target = mine[0]
                if target[3]['act'] is None:                # still voting: keep the locked action
                    target[3]['act'] = self.lock_act
            elif not mine and self.lock_last is not None and now - self.lock_seen < self.hold_s:
                target, held = self.lock_last, True        # brief dropout (step shake/blur): repeat last position
            else:
                self.lock_id = None
        if target is None and movable:
            target = max(movable, key=lambda d: d[1])      # closest = lowest in the image
            self.lock_id, self.lock_act = target[3]['tid'], target[3]['act']
        if target is not None and not held:
            self.lock_seen, self.lock_last = now, target
        if target is None:
            self.lock_id = None
        self.last_target = (target[0], target[1]) if target else None
        if target:
            x, y, w, h = target[2]
            px, py = int(w * 0.12), int(h * 0.12)
            crop = bgr[max(0, y - py):min(H, y + h + py), max(0, x - px):min(W, x + w + px)]
            ok, enc = cv2.imencode('.jpg', crop)
            if ok:
                self.target_jpg = enc.tobytes()

        def jd(d):
            r = d[3]
            return {'x': d[0], 'y': d[1], 'box': list(map(int, d[2])), 'fruit': r['fruit'],
                    'verdict': r['verdict'], 'conf': round(r['conf'], 3), 'rot': round(r['rot'], 3),
                    'stage': r['stage'], 'stage_p': r['stage_p'] and round(r['stage_p'], 3),
                    'action': r['action'], 'act': r['act'], 'pick': r['act'] in ('basket', 'bin'),
                    'gate': r.get('gate'), 'clip': r.get('clip'), 'angle': r['angle'], 'tid': r.get('tid'),
                    'not_target': r.get('not_target', '')}
        self.fps_n += 1
        if now - self.fps_t >= 2.0:
            self.status['fps'] = round(self.fps_n / (now - self.fps_t), 1)
            self.fps_t, self.fps_n = now, 0
        self.status.update(detections=[jd(d) for d in dets], target=jd(target) if target else None,
                           detecting=True, t=now)

        def info(d):
            r = d[3]
            o = ObjectInfo()
            o.label = '%s:%s_%s' % (r['act'] or 'none', r['verdict'], r['fruit'])   # e.g. "bin:spoiled_banana"
            o.type = 'rect'
            o.x, o.y = d[0], d[1]
            o.width, o.height = W, H     # stock convention: frame size, not object size
            o.radius = int(max(d[2][2], d[2][3]) / 2)
            # elongated fruit (banana): turn so it lies crosswise between the hands; round fruit: no turning
            o.angle = r['angle'] if r['elong'] >= self.elong_min else 0
            return o

        self.all_pub.publish(ObjectsInfo([info(d) for d in dets]))
        self.target_pub.publish(ObjectsInfo([info(target)] if target else []))

        # annotated image
        ACOL = {'basket': (0, 200, 0), 'bin': (0, 0, 255), 'leave': (0, 200, 255)}
        for d in dets:
            x, y, w, h = d[2]
            r = d[3]
            if r.get('gate') == 'not_fruit':
                col, txt = (140, 140, 140), 'not fruit: %s' % r.get('clip', '?').replace('a ', '', 1)
            elif r.get('gate') == 'checking':
                col, txt = (170, 170, 170), 'checking... (%s %s?)' % (r['verdict'], r['fruit'])
            elif r.get('not_target'):
                col = (255, 160, 60)                       # classified, but not something the robot goes for
                txt = '%s %s %.0f%% (%s)' % (r['verdict'].upper(), r['fruit'], 100 * r['conf'], r['not_target'])
            else:
                col = ACOL.get(r['act'], (0, 200, 255))
                where = {'basket': '-> basket', 'bin': '-> bin', 'leave': 'leave'}.get(r['act'], '')
                txt = '%s %s %.0f%% %s' % (r['verdict'].upper(), r['fruit'], 100 * r['conf'], where)
            th = 4 if d is target else (1 if r.get('gate') in ('not_fruit', 'checking') else 2)
            cv2.rectangle(bgr, (x, y), (x + w, y + h), col, th)
            cv2.putText(bgr, txt, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if target:
            cv2.drawMarker(bgr, (target[0], target[1]), ACOL.get(target[3]['act'], (0, 255, 0)), cv2.MARKER_CROSS, 24, 2)
        return bgr

    def publish_image(self, bgr):
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.height, msg.width = bgr.shape[:2]
        msg.encoding = 'rgb8'
        msg.step = msg.width * 3
        msg.data = np.ascontiguousarray(bgr[:, :, ::-1]).tobytes()
        self.image_pub.publish(msg)

    def run(self):
        while not rospy.is_shutdown():
            try:
                bgr = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            with self.lock:
                on = self.detect_on
            focus = self.measure_focus(bgr)
            if on:
                bgr = self.process(bgr)
            self.draw_focus(bgr, focus)
            self.publish_image(bgr)

    def measure_focus(self, bgr):
        """Sharpness for focusing the manual lens: variance of the Laplacian over the centre
        half of the frame (higher = sharper). Smoothed a little so the number is readable."""
        H, W = bgr.shape[:2]
        g = cv2.cvtColor(bgr[H // 4:3 * H // 4, W // 4:3 * W // 4], cv2.COLOR_BGR2GRAY)
        v = float(cv2.Laplacian(g, cv2.CV_64F).var())
        self.focus_ema = v if getattr(self, 'focus_ema', None) is None else 0.6 * self.focus_ema + 0.4 * v
        self.status['focus'] = round(self.focus_ema, 1)
        return self.focus_ema

    @staticmethod
    def draw_focus(bgr, focus):
        H, W = bgr.shape[:2]
        cv2.rectangle(bgr, (W // 4, H // 4), (3 * W // 4, 3 * H // 4), (255, 255, 255), 1)
        cv2.putText(bgr, 'focus %.0f' % focus, (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(bgr, 'focus %.0f' % focus, (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


if __name__ == '__main__':
    FruitDetectorNode('fruit_detection').run()
