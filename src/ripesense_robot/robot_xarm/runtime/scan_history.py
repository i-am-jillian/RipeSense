#!/usr/bin/env python3
"""Robot scan history on Tiger Data (TimescaleDB).

Every time the arm looks at a fruit, one row goes into the `fruit_scans` hypertable: where (fruit
position in the arm base frame, rail position, camera pose), when, what the sensors measured (RealSense
size + colour stats, AS7341 spectral channels when that sensor is fitted) and what the model predicted.

Fruits carry no tags, so identity = position: a detection within MATCH_MM of a fruit seen in the last
ACTIVE_H hours is the same fruit (greedy nearest match, one detection per fruit).

  python scan_history.py next        # which fruit should we inspect now (ranked, with reasons)
  python scan_history.py changes     # what changed since each fruit's previous scan
  python scan_history.py history 3   # all scans of fruit 3
  python scan_history.py flush       # push scans buffered while offline

Connection string: env TIGER_DATABASE_URL, else ~/.tiger_env or ../.env (TIGER_DATABASE_URL=...).
A failed write never stops the arm: the row is appended to ~/scan_history_pending.jsonl instead.
"""
from __future__ import annotations

import json, math, os, sys, time
from datetime import datetime, timezone

MATCH_MM = 40.0
ACTIVE_H = 12
PENDING = os.path.expanduser('~/scan_history_pending.jsonl')

DDL = """
CREATE TABLE IF NOT EXISTS fruits (
    fruit_id   SERIAL PRIMARY KEY,
    first_seen TIMESTAMPTZ NOT NULL,
    last_seen  TIMESTAMPTZ NOT NULL,
    x_mm DOUBLE PRECISION, y_mm DOUBLE PRECISION,
    label TEXT
);
CREATE TABLE IF NOT EXISTS fruit_scans (
    ts          TIMESTAMPTZ NOT NULL,
    session     TEXT,
    source      TEXT,              -- overview | inspect | pick
    fruit_id    INTEGER,
    -- where
    x_mm DOUBLE PRECISION, y_mm DOUBLE PRECISION, top_z_mm DOUBLE PRECISION, table_z_mm DOUBLE PRECISION,
    rail_mm DOUBLE PRECISION,
    cam_pose    JSONB,             -- flange x,y,z mm + roll,pitch,yaw deg when the frame was taken
    cam_dist_mm DOUBLE PRECISION,  -- camera to fruit top
    -- what the sensors measured
    height_mm DOUBLE PRECISION, length_mm DOUBLE PRECISION, width_mm DOUBLE PRECISION,
    area_px INTEGER,
    hue_deg DOUBLE PRECISION, sat DOUBLE PRECISION, val DOUBLE PRECISION,
    dark_frac DOUBLE PRECISION,    -- share of fruit pixels that are very dark (bruise / rot cue)
    spectral    JSONB,             -- AS7341 channels, NULL until the sensor is fitted
    -- what the model predicted
    model       TEXT,
    cls         TEXT,
    conf        DOUBLE PRECISION,
    p_rotten    DOUBLE PRECISION,  -- summed probability of all *_rotten classes
    action      TEXT,              -- basket | bin | ignore
    probs       JSONB,
    crop_path   TEXT
);
CREATE INDEX IF NOT EXISTS fruit_scans_fruit_ts ON fruit_scans (fruit_id, ts DESC);
"""
HYPERTABLE = "SELECT create_hypertable('fruit_scans', by_range('ts'), if_not_exists => TRUE)"

# latest + previous scan per fruit (fruits seen in the active window), shared by `next` and `changes`
PAIRS = """
WITH s AS (
  SELECT fs.*, row_number() OVER (PARTITION BY fruit_id ORDER BY ts DESC) AS rn
  FROM fruit_scans fs
  WHERE ts > now() - make_interval(hours => %(h)s) AND fruit_id IS NOT NULL
)
SELECT c.fruit_id, c.ts, c.cls, c.conf, c.p_rotten, c.action, c.hue_deg, c.dark_frac, c.height_mm,
       c.x_mm, c.y_mm, c.source,
       p.ts AS p_ts, p.cls AS p_cls, p.conf AS p_conf, p.p_rotten AS p_p_rotten, p.hue_deg AS p_hue,
       p.dark_frac AS p_dark, p.height_mm AS p_height,
       (SELECT count(*) FROM s s2 WHERE s2.fruit_id = c.fruit_id) AS n_scans,
       extract(epoch FROM now() - c.ts) / 60.0 AS age_min
FROM s c LEFT JOIN s p ON p.fruit_id = c.fruit_id AND p.rn = 2
WHERE c.rn = 1
"""

# priority: likely spoiled, unsure, getting worse, label flipped, or simply not looked at for a while
NEXT = """
SELECT *,
  round((0.40 * p_rotten
       + 0.20 * (1 - conf)
       + 0.20 * least(greatest(coalesce(p_rotten - p_p_rotten, 0), 0) * 4, 1)
       + 0.10 * (CASE WHEN p_cls IS NOT NULL AND p_cls <> cls THEN 1 ELSE 0 END)
       + 0.10 * least(age_min / 30.0, 1))::numeric, 3) AS priority
FROM (""" + PAIRS + """) t
WHERE cls <> 'not_fruit'
  -- only fruits that are still there: seen in the most recent search session (stale / mis-detected ones drop out)
  AND fruit_id IN (SELECT fruit_id FROM fruit_scans
                   WHERE session = (SELECT session FROM fruit_scans WHERE source IN ('overview', 'front', 'floor')
                                    ORDER BY ts DESC LIMIT 1))
ORDER BY priority DESC
"""


def _url():
    u = os.environ.get('TIGER_DATABASE_URL')
    if u:
        return u
    for f in ('~/.tiger_env', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')):
        f = os.path.expanduser(f)
        if os.path.exists(f):
            for line in open(f):
                if line.startswith('TIGER_DATABASE_URL='):
                    return line.split('=', 1)[1].strip()
    return None


def connect():
    import psycopg
    url = _url()
    if not url:
        raise RuntimeError('TIGER_DATABASE_URL not set')
    return psycopg.connect(url, connect_timeout=6)


def ensure_schema(con):
    con.execute(DDL)
    try:
        con.execute(HYPERTABLE)
    except Exception as e:  # noqa: BLE001
        print('scan_history: hypertable not created (%s), plain table' % e, file=sys.stderr)
        con.rollback(); con.execute(DDL)
    con.commit()


def action_for(cls):
    if cls == 'not_fruit':
        return 'ignore'
    return 'bin' if cls.endswith('_rotten') else 'basket'


def colour_stats(bgr, mask):
    """Circular-mean hue (deg), mean saturation/value (0-1) and dark-pixel share inside the object mask."""
    import cv2, numpy as np
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[mask > 0].astype(np.float32)
    if len(hsv) < 20:
        return None, None, None, None
    ang = hsv[:, 0] * 2 * math.pi / 180.0
    hue = math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360
    return hue, float(hsv[:, 1].mean() / 255), float(hsv[:, 2].mean() / 255), float((hsv[:, 2] < 60).mean())


def _match(con, rows, now):
    """Assign fruit_id by position; create new fruits for unmatched detections."""
    known = con.execute("SELECT fruit_id, x_mm, y_mm, label FROM fruits WHERE last_seen > now() - make_interval(hours => %s)",
                        (ACTIVE_H,)).fetchall()
    # same place AND same kind of fruit (apple / orange / ...) -> same fruit; a different fruit put in the
    # same spot is a new fruit (ripeness may change between scans, the fruit type may not)
    kind = lambda c: (c or '').split('_')[0]
    pairs = sorted(((math.hypot(r['x_mm'] - kx, r['y_mm'] - ky), i, fid)
                    for i, r in enumerate(rows) for fid, kx, ky, kl in known
                    if kind(kl) == kind(r['cls']) or r.get('conf', 1) < 0.5), key=lambda t: t[0])
    used_r, used_f = set(), set()
    for d, i, fid in pairs:
        if d > MATCH_MM or i in used_r or fid in used_f:
            continue
        rows[i]['fruit_id'] = fid; used_r.add(i); used_f.add(fid)
    for i, r in enumerate(rows):
        if r.get('fruit_id') is None and r['cls'] != 'not_fruit':
            r['fruit_id'] = con.execute("INSERT INTO fruits (first_seen, last_seen, x_mm, y_mm, label) VALUES "
                                        "(%s, %s, %s, %s, %s) RETURNING fruit_id",
                                        (now, now, r['x_mm'], r['y_mm'], r['cls'])).fetchone()[0]
        elif r.get('fruit_id') is not None:
            con.execute("UPDATE fruits SET last_seen = %s, x_mm = %s, y_mm = %s, label = %s WHERE fruit_id = %s",
                        (now, r['x_mm'], r['y_mm'], r['cls'], r['fruit_id']))


COLS = ['ts', 'session', 'source', 'fruit_id', 'x_mm', 'y_mm', 'top_z_mm', 'table_z_mm', 'rail_mm', 'cam_pose',
        'cam_dist_mm', 'height_mm', 'length_mm', 'width_mm', 'area_px', 'hue_deg', 'sat', 'val', 'dark_frac',
        'spectral', 'model', 'cls', 'conf', 'p_rotten', 'action', 'probs', 'crop_path']


def _insert(con, rows):
    from psycopg.types.json import Jsonb
    sql = 'INSERT INTO fruit_scans (%s) VALUES (%s)' % (', '.join(COLS), ', '.join(['%s'] * len(COLS)))
    for r in rows:
        con.execute(sql, [Jsonb(r.get(c)) if c in ('cam_pose', 'spectral', 'probs') and r.get(c) is not None
                          else r.get(c) for c in COLS])


def log_scans(rows, match=True):
    """rows: list of dicts with COLS keys (ts optional). Returns rows with fruit_id filled. Never raises."""
    now = datetime.now(timezone.utc)
    for r in rows:
        r.setdefault('ts', now)
        r.setdefault('action', action_for(r['cls']))
    try:
        with connect() as con:
            ensure_schema(con)
            if match:
                _match(con, rows, now)
            else:
                for r in rows:
                    if r.get('fruit_id') is not None:
                        con.execute("UPDATE fruits SET last_seen = %s, x_mm = %s, y_mm = %s WHERE fruit_id = %s",
                                    (now, r['x_mm'], r['y_mm'], r['fruit_id']))
            _insert(con, rows)
            con.commit()
    except Exception as e:  # noqa: BLE001
        print('scan_history: Tiger write failed (%s); buffered to %s' % (e, PENDING), file=sys.stderr)
        with open(PENDING, 'a') as f:
            for r in rows:
                f.write(json.dumps({**r, 'ts': r['ts'].isoformat()}, default=str) + '\n')
    return rows


def flush():
    if not os.path.exists(PENDING):
        print('nothing pending'); return
    rows = [json.loads(l) for l in open(PENDING)]
    for r in rows:
        r['ts'] = datetime.fromisoformat(r['ts'])
    os.rename(PENDING, PENDING + '.sending')
    with connect() as con:
        ensure_schema(con)
        for ts in sorted({r['ts'] for r in rows}):
            batch = [r for r in rows if r['ts'] == ts]
            if any(r.get('fruit_id') is None for r in batch):
                _match(con, batch, ts)
            _insert(con, batch)
        con.commit()
    os.remove(PENDING + '.sending'); print('flushed %d scans' % len(rows))


def _q(sql, **kw):
    with connect() as con:
        ensure_schema(con)
        cur = con.execute(sql, {'h': ACTIVE_H, **kw})
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def describe_change(r):
    """Human sentence: what changed between this fruit's previous scan and its latest one."""
    if r['p_ts'] is None:
        return 'first scan'
    bits = []
    if r['p_cls'] != r['cls']:
        bits.append('%s -> %s' % (r['p_cls'], r['cls']))
    dp = (r['p_rotten'] or 0) - (r['p_p_rotten'] or 0)
    if abs(dp) >= 0.05:
        bits.append('spoiled prob %.0f%% -> %.0f%%' % (100 * r['p_p_rotten'], 100 * r['p_rotten']))
    if r['dark_frac'] is not None and r['p_dark'] is not None and abs(r['dark_frac'] - r['p_dark']) >= 0.02:
        bits.append('dark area %.0f%% -> %.0f%%' % (100 * r['p_dark'], 100 * r['dark_frac']))
    if r['hue_deg'] is not None and r['p_hue'] is not None:
        dh = (r['hue_deg'] - r['p_hue'] + 180) % 360 - 180
        if abs(dh) >= 4:
            bits.append('hue %+.0f deg' % dh)
    if r['height_mm'] is not None and r['p_height'] is not None and abs(r['height_mm'] - r['p_height']) >= 6:
        bits.append('height %+.0f mm' % (r['height_mm'] - r['p_height']))
    gap = (r['ts'] - r['p_ts']).total_seconds() / 60
    return ('; '.join(bits) if bits else 'no significant change') + ' (over %.0f min)' % gap


def reasons(r):
    out = []
    if r['p_rotten'] >= 0.5: out.append('likely spoiled (%.0f%%)' % (100 * r['p_rotten']))
    if r['conf'] < 0.6: out.append('model unsure (%.0f%%)' % (100 * r['conf']))
    if r['p_p_rotten'] is not None and r['p_rotten'] - r['p_p_rotten'] >= 0.1: out.append('getting worse')
    if r['p_cls'] is not None and r['p_cls'] != r['cls']: out.append('label changed')
    if r['age_min'] >= 30: out.append('not checked for %.0f min' % r['age_min'])
    return ', '.join(out) or 'routine check'


def inspect_next(limit=5):
    return _q(NEXT + ' LIMIT %(n)s', n=limit)


def changes():
    return _q(PAIRS + ' ORDER BY c.fruit_id')


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'next'
    if cmd == 'next':
        rows = inspect_next()
        if not rows: print('no fruit scanned in the last %d h' % ACTIVE_H); return
        print('Inspect next:')
        for i, r in enumerate(rows, 1):
            print(' %d. fruit #%d %-17s at (%4.0f, %4.0f)  priority %.2f  -- %s' % (
                i, r['fruit_id'], r['cls'], r['x_mm'], r['y_mm'], r['priority'], reasons(r)))
            print('      since last scan: %s' % describe_change(r))
    elif cmd == 'changes':
        for r in changes():
            print('fruit #%-3d %-17s %3d scans  %s' % (r['fruit_id'], r['cls'], r['n_scans'], describe_change(r)))
    elif cmd == 'history':
        for r in _q("SELECT ts, source, cls, conf, p_rotten, hue_deg, dark_frac, cam_dist_mm FROM fruit_scans "
                    "WHERE fruit_id = %(f)s ORDER BY ts", f=int(sys.argv[2])):
            print('%s %-8s %-17s conf %.2f  rotten %.2f  hue %5.1f  dark %.2f  cam %4.0f mm' % (
                r['ts'].strftime('%H:%M:%S'), r['source'], r['cls'], r['conf'], r['p_rotten'],
                r['hue_deg'] or 0, r['dark_frac'] or 0, r['cam_dist_mm'] or 0))
    elif cmd == 'flush':
        flush()
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
