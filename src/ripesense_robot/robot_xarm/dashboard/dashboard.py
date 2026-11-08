#!/usr/bin/env python3
"""RipeSense xArm dashboard (laptop):  http://localhost:5050

  Live wrist camera (Pi, via the localhost:8091 tunnel) - arm / rail / model / Tiger status
  Controls: Row search (front first, then floor), Inspect next, Pick & place test, STOP
  What the robot sees (categorized), Tiger Data: inspect-next ranking, what changed, fruit history charts.

  ~/UB_Hack/ripesense/.venv/bin/python dashboard.py
"""
from __future__ import annotations

import json, os, re, subprocess, sys, threading, time, urllib.request
from datetime import datetime
from decimal import Decimal

from flask import Flask, Response, abort, jsonify, request

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'runtime'))
import scan_history as SH  # noqa: E402

PI = os.environ.get('PI_SSH', 'sujal@128.205.223.114')   # ssh target of the Pi on the arm
CAM = 'http://localhost:8091'           # ssh tunnel to the Pi's camera page
PY = '/home/sujal/ripesense_env/bin/python'
CTL = os.path.expanduser('~/.ssh/cm-xarm-%r@%h')
CACHE = os.path.join(HERE, '.dash_cache'); os.makedirs(CACHE, exist_ok=True)
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6', '-o', 'ControlMaster=auto',
       '-o', 'ControlPath=' + CTL, '-o', 'ControlPersist=120', PI]

JOBS = {   # name -> (label, command on the Pi, needs the "front zone clear" confirmation)
    'demo':         ('Full demo: search, then ripe -> bin', '/home/sujal/demo.sh full', True),
    'auto':         ('Autopilot', '/home/sujal/autopilot.sh', False),
    'search':       ('Row search (front, then floor)', PY + ' /home/sujal/row_search.py --exit', True),
    'search_front': ('Front search only',              PY + ' /home/sujal/row_search.py --exit --front_only', True),
    'search_floor': ('Floor search only',              PY + ' /home/sujal/row_search.py --exit --floor_only', False),
    'inspect':      ('Inspect top 2 (Tiger)',          PY + ' /home/sujal/pick_all.py --inspect 2', False),
    'pick':         ('Pick & place test (gentle)',     PY + ' /home/sujal/pick_all.py --go', False),
    'sort':         ('Ripe fruit -> bin (whole table)', '/home/sujal/demo.sh pick', False),
}

app = Flask(__name__)


def ssh(cmd, timeout=25):
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def jsonable(o):
    if isinstance(o, Decimal): return float(o)
    if isinstance(o, datetime): return o.isoformat()
    return str(o)


def J(obj):
    return Response(json.dumps(obj, default=jsonable), mimetype='application/json')


_cache = {}
def cached(key, ttl, fn):
    now = time.time(); hit = _cache.get(key)
    if hit and now - hit[0] < ttl: return hit[1]
    val = fn(); _cache[key] = (now, val); return val


def q(sql, **kw):
    with SH.connect() as con:
        cur = con.execute(sql, kw)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


# ------------------------------------------------------------------ Pi / arm
@app.get('/api/status')
def status():
    def get():
        try:
            rc, out, err = ssh(PY + ' /home/sujal/arm_status.py', timeout=15)
            return json.loads(out.strip().splitlines()[-1]) if rc == 0 and out.strip() else {'pi': False, 'err': err[-200:]}
        except Exception as e:  # noqa: BLE001
            return {'pi': False, 'err': str(e)}
    return J(cached('status', 4, get))


@app.get('/api/log')
def log():
    def get():
        try:
            rc, out, _ = ssh("grep -vE 'SDK_VERSION|ROBOT_IP|protocol|Warn|warn|^\\s*$' /tmp/job.log 2>/dev/null | tail -n 400")
            return {'log': out}
        except Exception as e:  # noqa: BLE001
            return {'log': 'log unavailable: %s' % e}
    return J(cached('log', 1.5, get))


@app.post('/api/run/<name>')
def run(name):
    if name not in JOBS: abort(404)
    label, cmd, needs_clear = JOBS[name]
    body = request.get_json(silent=True) or {}
    if needs_clear and not body.get('front_clear'):
        return J({'ok': False, 'msg': 'Confirm that the area in front of the arm is clear of people first.'}), 400
    st = cached('status', 4, lambda: {})
    if isinstance(st, dict) and st.get('busy'):
        return J({'ok': False, 'msg': 'Another arm job is running. Stop it first.'}), 409
    if name == 'auto':      # front views in autopilot only when the operator confirmed the front zone is clear
        cmd = 'env FRONT=%d %s' % (1 if body.get('front_clear') else 0, cmd)
    rc, out, err = ssh('(setsid nohup /home/sujal/job.sh %s %s > /tmp/job_launch.log 2>&1 < /dev/null &); sleep 0.5; cat /tmp/job_launch.log' % (name, cmd))
    _cache.pop('status', None)
    if 'busy' in out:
        return J({'ok': False, 'msg': out.strip()}), 409
    return J({'ok': rc == 0, 'msg': 'started: ' + label})


@app.post('/api/stop')
def stop():
    rc, out, err = ssh('echo "STOPPED BY USER" >> /tmp/job.log; pkill -f "^bash /home/sujal/(autopilot|demo)\\.sh"; ' + PY + ' /home/sujal/arm_stop.py; '
                       'pkill -INT -f "^/home/sujal/ripesense_env/bin/python /home/sujal/(row_search|pick_all)\\.py"; sleep 1; '
                       'pkill -KILL -f "^/home/sujal/ripesense_env/bin/python /home/sujal/(row_search|pick_all)\\.py"; echo killed')
    _cache.pop('status', None)
    return J({'ok': rc == 0, 'msg': (out or err).strip()[-300:]})


@app.post('/api/auto_off')
def auto_off():
    rc, out, err = ssh('touch /tmp/autopilot.stop; echo ok')
    _cache.pop('status', None)
    return J({'ok': rc == 0, 'msg': 'Autopilot will stop after the current fruit.'})


@app.get('/api/search')
def search_state():
    """Live categorized state from a running row search, else the last finished one."""
    path = os.path.join(CACHE, 'row_state.json')
    try:
        with urllib.request.urlopen(CAM + '/state', timeout=1.2) as r:
            if 'json' in r.headers.get('Content-Type', ''):
                data = r.read(); open(path, 'wb').write(data)
                return Response(data, mimetype='application/json')
    except Exception:  # noqa: BLE001
        pass
    def from_pi():
        rc, out, _ = ssh('cat /tmp/row_state.json 2>/dev/null')
        if rc == 0 and out.strip():
            open(path, 'w').write(out); return json.loads(out)
        return json.load(open(path)) if os.path.exists(path) else {'items': [], 'phase': 'no search yet'}
    return J(cached('rowstate', 5, from_pi))


@app.get('/api/crop')
def crop():
    p = request.args.get('p', '')
    if not re.fullmatch(r'/home/sujal/scan_crops/[\w\-]+/[\w\-+]+\.jpg', p): abort(400)
    local = os.path.join(CACHE, p.replace('/', '_'))
    if not os.path.exists(local):
        r = subprocess.run(SSH + ['cat', p], capture_output=True, timeout=15)
        if r.returncode != 0 or not r.stdout: abort(404)
        open(local, 'wb').write(r.stdout)
    return Response(open(local, 'rb').read(), mimetype='image/jpeg', headers={'Cache-Control': 'max-age=86400'})


# ------------------------------------------------------------------ Tiger Data
@app.get('/api/next')
def nxt():
    def get():
        rows = SH.inspect_next(6)
        return [dict(fruit_id=r['fruit_id'], cls=r['cls'], conf=r['conf'], p_rotten=r['p_rotten'], priority=float(r['priority']),
                     x=r['x_mm'], y=r['y_mm'], why=SH.reasons(r), change=SH.describe_change(r), n_scans=r['n_scans'],
                     age_min=float(r['age_min'])) for r in rows]
    try:
        return J(cached('next', 3, get))
    except Exception as e:  # noqa: BLE001
        return J({'error': str(e)}), 502


FRUITS = """
WITH latest AS (SELECT session FROM fruit_scans WHERE source IN ('overview','front','floor') ORDER BY ts DESC LIMIT 1)
SELECT DISTINCT ON (fruit_id) fruit_id, ts, cls, conf, p_rotten, source, action, x_mm, y_mm, top_z_mm, dark_frac, hue_deg, crop_path,
       count(*) OVER (PARTITION BY fruit_id) AS n_scans,
       bool_or(session = (SELECT session FROM latest)) OVER (PARTITION BY fruit_id) AS present
FROM fruit_scans
WHERE ts > now() - interval '12 hours' AND fruit_id IS NOT NULL
ORDER BY fruit_id, ts DESC
"""


@app.get('/api/fruits')
def fruits():
    try:
        return J(cached('fruits', 3, lambda: q(FRUITS)))
    except Exception as e:  # noqa: BLE001
        return J({'error': str(e)}), 502


@app.get('/api/history/<int:fid>')
def history(fid):
    try:
        return J(q("SELECT ts, source, cls, conf, p_rotten, dark_frac, hue_deg, cam_dist_mm, crop_path, model "
                   "FROM fruit_scans WHERE fruit_id = %(f)s ORDER BY ts LIMIT 300", f=fid))
    except Exception as e:  # noqa: BLE001
        return J({'error': str(e)}), 502


@app.get('/api/stats')
def stats():
    def get():
        s = q("SELECT count(*) AS scans, count(DISTINCT fruit_id) AS fruits, count(DISTINCT session) AS sessions, "
              "max(ts) AS last FROM fruit_scans")[0]
        s['buckets'] = q("SELECT time_bucket('10 minutes', ts) AS b, count(*) AS n FROM fruit_scans "
                         "WHERE ts > now() - interval '6 hours' GROUP BY b ORDER BY b")
        s['to_bin'] = q("SELECT count(*) AS n FROM fruit_scans WHERE action = 'to_bin' AND ts > now() - interval '12 hours'")[0]['n']
        s['hypertable'] = bool(q("SELECT 1 AS x FROM timescaledb_information.hypertables WHERE hypertable_name = 'fruit_scans'"))
        return s
    try:
        return J(cached('stats', 5, get))
    except Exception as e:  # noqa: BLE001
        return J({'error': str(e)}), 502


@app.get('/api/bin')
def bin_pose():
    def get():
        rc, out, _ = ssh('cat ~/drop_poses.json 2>/dev/null')
        return json.loads(out) if rc == 0 and out.strip() else {}
    return J(cached('bin', 60, get))


@app.get('/classic')
def classic():
    return Response(open(os.path.join(HERE, 'dashboard_classic.html')).read(), mimetype='text/html')


@app.get('/')
def index():
    return Response(open(os.path.join(HERE, 'dashboard.html')).read(), mimetype='text/html',
                    headers={'Cache-Control': 'no-store, max-age=0'})


if __name__ == '__main__':
    threading.Thread(target=lambda: ssh('true'), daemon=True).start()     # warm the ssh master connection
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5050)), threaded=True)
