"""
MDU Engine — Feedback Capture
Persists practitioner feedback to Google Cloud Storage,
which survives Cloud Run container restarts.

Falls back to local file if GCS is not configured
(useful for local development and testing).

Environment variables:
    MDU_FEEDBACK_BUCKET  — GCS bucket name (e.g. "mdu-engine-data")
                           If not set, falls back to local CSV file.
    MDU_FEEDBACK_BLOB    — GCS object path (default: "feedback/mdu_feedback.csv")
    MDU_FEEDBACK_FILE    — Local fallback path (default: "mdu_feedback.csv")
"""

import csv
import io
import os
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────
FEEDBACK_BUCKET = os.environ.get("MDU_FEEDBACK_BUCKET", "")
FEEDBACK_BLOB   = os.environ.get("MDU_FEEDBACK_BLOB", "feedback/mdu_feedback.csv")
FEEDBACK_FILE   = os.environ.get("MDU_FEEDBACK_FILE", "mdu_feedback.csv")

FEEDBACK_FIELDNAMES = [
    "submitted_at_utc",
    "name",
    "role",
    "company",
    "use_case",
    "was_useful",
    "decision_outcome",
    "platform",
    "days_of_data",
    "spend_total",
    "confidence_tier",
    "engine_version",
    "ruleset_version",
    "input_hash",
]


# ── GCS helpers ────────────────────────────────────────────

def _gcs_available() -> bool:
    """True only when the bucket env var is set and google-cloud-storage is installed."""
    if not FEEDBACK_BUCKET:
        return False
    try:
        from google.cloud import storage  # noqa: F401
        return True
    except ImportError:
        return False


def _gcs_read_csv() -> list[dict]:
    """
    Download the CSV blob from GCS and return as list of dicts.
    Returns [] if the blob does not exist yet.
    """
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(FEEDBACK_BUCKET)
    blob   = bucket.blob(FEEDBACK_BLOB)

    if not blob.exists():
        return []

    raw = blob.download_as_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    return list(reader)


def _gcs_write_csv(rows: list[dict]) -> None:
    """
    Upload the full list of rows as a CSV blob to GCS.
    Overwrites the existing blob (read-modify-write pattern).
    """
    from google.cloud import storage

    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=FEEDBACK_FIELDNAMES,
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)

    client = storage.Client()
    bucket = client.bucket(FEEDBACK_BUCKET)
    blob   = bucket.blob(FEEDBACK_BLOB)
    blob.upload_from_string(
        buf.getvalue().encode("utf-8"),
        content_type="text/csv",
    )


# ── Local file helpers (fallback) ──────────────────────────

def _local_read_csv() -> list[dict]:
    path = Path(FEEDBACK_FILE)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        logger.warning("feedback: local read failed: %s", exc)
        return []


def _local_append_row(record: dict) -> None:
    path = Path(FEEDBACK_FILE)
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FEEDBACK_FIELDNAMES,
            extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


# ── Public API ─────────────────────────────────────────────

def write_feedback(record: dict) -> bool:
    """
    Append one feedback record.
    Uses GCS if configured, otherwise falls back to local file.
    Returns True on success, False on failure. Never raises.
    """
    try:
        if _gcs_available():
            # Read-modify-write so we never lose existing rows
            rows = _gcs_read_csv()
            # Fill missing fields with empty string
            full_record = {k: record.get(k, "") for k in FEEDBACK_FIELDNAMES}
            rows.append(full_record)
            _gcs_write_csv(rows)
        else:
            _local_append_row(record)
        return True
    except Exception as exc:
        logger.error("feedback: write_feedback failed: %s", exc)
        return False


def read_feedback(limit: int = 100) -> list[dict]:
    """
    Read feedback records, newest first.
    Uses GCS if configured, otherwise falls back to local file.
    Never raises.
    """
    try:
        if _gcs_available():
            rows = _gcs_read_csv()
        else:
            rows = _local_read_csv()
        return list(reversed(rows))[:limit]
    except Exception as exc:
        logger.error("feedback: read_feedback failed: %s", exc)
        return []


def feedback_to_csv_bytes() -> bytes:
    """
    Return the full feedback CSV as bytes for Streamlit download_button.
    Never raises.
    """
    try:
        if _gcs_available():
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(FEEDBACK_BUCKET)
            blob   = bucket.blob(FEEDBACK_BLOB)
            if not blob.exists():
                return b""
            return blob.download_as_bytes()
        else:
            path = Path(FEEDBACK_FILE)
            if not path.exists():
                return b""
            with open(path, "rb") as f:
                return f.read()
    except Exception as exc:
        logger.error("feedback: feedback_to_csv_bytes failed: %s", exc)
        return b""