"""Optional: copy each scan to PostgreSQL / Tiger Data for a store dashboard (MLH Tiger Data prize)."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS ripesense_scans (
  time TIMESTAMPTZ NOT NULL DEFAULT now(),
  scan_id INTEGER, fruit TEXT, stage TEXT, days_left REAL, confidence REAL,
  markdown_low REAL, markdown_high REAL, mode TEXT
);
"""


class PostgresSink:
    def __init__(self, url: str):
        import psycopg
        self.conn = psycopg.connect(url, autocommit=True)
        with self.conn.cursor() as c:
            c.execute(DDL)
            try:  # Tiger Data / TimescaleDB: make it a hypertable; harmless if unavailable
                c.execute("SELECT create_hypertable('ripesense_scans', 'time', if_not_exists => TRUE)")
            except Exception:
                pass

    def write(self, scan_id: int, v: dict, pred) -> None:
        md = v.get("markdown") or [None, None]
        with self.conn.cursor() as c:
            c.execute("INSERT INTO ripesense_scans (scan_id, fruit, stage, days_left, confidence, markdown_low, "
                      "markdown_high, mode) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                      (scan_id, v["fruit"], pred.stage, pred.days_mid if pred.days_mid == pred.days_mid else None,
                       pred.confidence, md[0], md[1], v["mode"]))


def make_sink(cfg):
    url = cfg["timeseries"].get("postgres_url")
    if not url:
        return None
    try:
        return PostgresSink(url)
    except Exception as e:
        log.error("time-series sink disabled: %s", e)
        return None
