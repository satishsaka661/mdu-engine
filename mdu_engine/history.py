# mdu_engine/history.py
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List

DB_PATH = os.path.join("reports", "history.db")


def _ensure_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at_utc TEXT NOT NULL,
            type TEXT,
            platform TEXT,
            action TEXT,
            status TEXT,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def log_decision(payload: Dict[str, Any]) -> None:
    conn = _ensure_db()
    record = dict(payload)
    record.setdefault("logged_at_utc", datetime.now(timezone.utc).isoformat())

    logged_at = record.get("logged_at_utc")
    d_type = record.get("type")
    platform = record.get("platform")
    action = record.get("action")
    status = record.get("status")

    conn.execute(
        "INSERT INTO decisions (logged_at_utc, type, platform, action, status, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
        (logged_at, d_type, platform, action, status, json.dumps(record, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def read_history(limit: int = 10) -> List[Dict[str, Any]]:
    if not os.path.exists(DB_PATH):
        return []

    conn = _ensure_db()
    cur = conn.execute(
        "SELECT payload_json FROM decisions ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()

    out: List[Dict[str, Any]] = []
    for (payload_json,) in rows:
        try:
            out.append(json.loads(payload_json))
        except Exception:
            continue
    return out