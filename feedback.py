"""
MDU Engine — Feedback Capture
Stores practitioner feedback to a local CSV file,
following the same file-based pattern as history.py.
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

# Storage location — same directory pattern as your history logs
FEEDBACK_FILE = os.environ.get("MDU_FEEDBACK_FILE", "mdu_feedback.csv")

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


def write_feedback(record: dict) -> bool:
    """
    Appends a feedback record to the CSV file.
    Returns True on success, False on failure.
    Never raises — safe to call in production.
    """
    try:
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
        return True
    except Exception:
        return False


def read_feedback(limit: int = 100) -> list[dict]:
    """
    Reads feedback records from the CSV file.
    Returns a list of dicts, newest first.
    Never raises.
    """
    try:
        path = Path(FEEDBACK_FILE)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return list(reversed(rows))[:limit]
    except Exception:
        return []


def feedback_to_csv_bytes() -> bytes:
    """
    Returns the full feedback CSV as bytes for download.
    """
    try:
        path = Path(FEEDBACK_FILE)
        if not path.exists():
            return b""
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return b""
