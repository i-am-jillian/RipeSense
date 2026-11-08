"""Optional store log on Tiger Data (TimescaleDB / Postgres) for the manager
dashboard. Set TIGER_DATABASE_URL and cloud.tiger_enabled: true.

Every write runs on a background thread and failures are only logged, so a
flaky network never blocks a scan."""
from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS ripesense_scans (
    ts            TIMESTAMPTZ NOT NULL,
    device        TEXT NOT NULL,
    scan_id       INTEGER,
    fruit_type    TEXT,
    stage         TEXT,
    confidence    DOUBLE PRECISION,
    days_p50      DOUBLE PRECISION,
    markdown_lo   INTEGER,
    markdown_hi   INTEGER,
    price         DOUBLE PRECISION,
    base_price    DOUBLE PRECISION
);
"""
HYPERTABLE = "SELECT create_hypertable('ripesense_scans', by_range('ts'), if_not_exists => TRUE);"

WEEK_BY_TIER = """
SELECT time_bucket('1 day', ts) AS day, stage, count(*) AS n,
       sum(coalesce(base_price,0) - coalesce(price, base_price, 0)) AS markdown_dollars
FROM ripesense_scans
WHERE ts > now() - interval '7 days'
GROUP BY day, stage ORDER BY day, stage;
"""


def _url():
    return os.environ.get("TIGER_DATABASE_URL")


def enabled(cfg) -> bool:
    return bool(cfg["cloud"].get("tiger_enabled") and _url())


def _connect():
    import psycopg
    return psycopg.connect(_url(), connect_timeout=5)


_ready = False


def _ensure():
    global _ready
    if _ready:
        return
    with _connect() as con:
        con.execute(DDL)
        try:
            con.execute(HYPERTABLE)
        except Exception as e:  # noqa: BLE001  (plain Postgres: no timescaledb extension)
            log.info("hypertable not created (%s); using a plain table", e)
            con.rollback()
        con.commit()
    _ready = True


def log_scan(cfg, row: dict):
    if not enabled(cfg):
        return

    def work():
        try:
            _ensure()
            with _connect() as con:
                con.execute(
                    "INSERT INTO ripesense_scans (ts, device, scan_id, fruit_type, stage, confidence, days_p50,"
                    " markdown_lo, markdown_hi, price, base_price) VALUES (%(ts)s, %(device)s, %(scan_id)s,"
                    " %(fruit_type)s, %(stage)s, %(confidence)s, %(days_p50)s, %(md_lo)s, %(md_hi)s, %(price)s,"
                    " %(base_price)s)", row)
                con.commit()
        except Exception as e:  # noqa: BLE001
            log.warning("Tiger log failed: %s", e)

    threading.Thread(target=work, daemon=True).start()


def week_by_tier(cfg) -> list[dict] | None:
    if not enabled(cfg):
        return None
    try:
        _ensure()
        with _connect() as con:
            cur = con.execute(WEEK_BY_TIER)
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as e:  # noqa: BLE001
        log.warning("Tiger query failed: %s", e)
        return None
