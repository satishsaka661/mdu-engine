"""
MDU Engine — Decision History
Persists decision history to Google Cloud Storage,
which survives Cloud Run container restarts.

Falls back to local SQLite if GCS is not configured
(useful for local development and testing).

Environment variables:
    MDU_FEEDBACK_BUCKET  — GCS bucket name (reuses same bucket as feedback)
    MDU_HISTORY_BLOB     — GCS object path (default: "history/mdu_history.json")
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Configuration ──────────────────────────────────────────
HISTORY_BUCKET = os.environ.get("MDU_FEEDBACK_BUCKET", "")
HISTORY_BLOB   = os.environ.get("MDU_HISTORY_BLOB", "history/mdu_history.json")
DB_PATH        = os.path.join("reports", "history.db")


# ── GCS helpers ────────────────────────────────────────────

def _gcs_available() -> bool:
    if not HISTORY_BUCKET:
        return False
    try:
        from google.cloud import storage  # noqa: F401
        return True
    except ImportError:
        return False


def _gcs_read() -> List[Dict[str, Any]]:
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(HISTORY_BUCKET)
    blob   = bucket.blob(HISTORY_BLOB)
    if not blob.exists():
        return []
    raw = blob.download_as_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return []


def _gcs_write(records: List[Dict[str, Any]]) -> None:
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(HISTORY_BUCKET)
    blob   = bucket.blob(HISTORY_BLOB)
    blob.upload_from_string(
        json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8"),
        content_type="application/json",
    )


# ── Local SQLite helpers (fallback) ────────────────────────

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


def _local_read(limit: int = 100) -> List[Dict[str, Any]]:
    if not os.path.exists(DB_PATH):
        return []
    conn = _ensure_db()
    cur = conn.execute(
        "SELECT payload_json FROM decisions ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    out = []
    for (payload_json,) in rows:
        try:
            out.append(json.loads(payload_json))
        except Exception:
            continue
    return out


def _local_append(record: Dict[str, Any]) -> None:
    conn = _ensure_db()
    conn.execute(
        "INSERT INTO decisions (logged_at_utc, type, platform, action, status, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
        (
            record.get("logged_at_utc"),
            record.get("type"),
            record.get("platform"),
            record.get("action"),
            record.get("status"),
            json.dumps(record, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


# ── Public API ─────────────────────────────────────────────

def log_decision(payload: Dict[str, Any]) -> None:
    """
    Append one decision record to history.
    Uses GCS if configured, otherwise falls back to local SQLite.
    Never raises.
    """
    try:
        record = dict(payload)
        record.setdefault("logged_at_utc", datetime.now(timezone.utc).isoformat())

        if _gcs_available():
            records = _gcs_read()
            records.append(record)
            # Keep last 500 records max
            if len(records) > 500:
                records = records[-500:]
            _gcs_write(records)
        else:
            _local_append(record)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("history: log_decision failed: %s", exc)


def read_history(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Read decision history, newest first.
    Uses GCS if configured, otherwise falls back to local SQLite.
    Never raises.
    """
    try:
        if _gcs_available():
            records = _gcs_read()
            return list(reversed(records))[:limit]
        else:
            return _local_read(limit=limit)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("history: read_history failed: %s", exc)
        return []


def get_latest_decision() -> Optional[Dict[str, Any]]:
    """
    Return the most recent decision record, or None.
    """
    items = read_history(limit=1)
    return items[0] if items else None


def history_to_json_bytes(limit: int = 500) -> bytes:
    """
    Return full history as JSON bytes for download.
    Never raises.
    """
    try:
        records = read_history(limit=limit)
        return json.dumps(records, indent=2, ensure_ascii=False).encode("utf-8")
    except Exception:
        return b"[]"