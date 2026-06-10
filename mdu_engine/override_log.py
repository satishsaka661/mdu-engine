"""
override_log.py — MDU Engine Override Logging
Tracks whether SCALE/HOLD/REDUCE/BLOCK recommendations were followed or overridden,
and what happened to campaign performance 72 hours later.
Persisted to GCS alongside decision history.
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional

# GCS import with graceful fallback for local dev
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

BUCKET_NAME = os.environ.get("MDU_FEEDBACK_BUCKET", "")
OVERRIDE_LOG_PREFIX = "override_logs/"
LOCAL_FALLBACK_FILE = "override_log_local.json"


# ── GCS helpers ──────────────────────────────────────────────────────────────

def _get_bucket():
    if not GCS_AVAILABLE or not BUCKET_NAME:
        return None
    try:
        client = storage.Client()
        return client.bucket(BUCKET_NAME)
    except Exception:
        return None


def _load_all_logs() -> list:
    bucket = _get_bucket()
    if bucket:
        try:
            blobs = bucket.list_blobs(prefix=OVERRIDE_LOG_PREFIX)
            logs = []
            for blob in blobs:
                if blob.name.endswith(".json"):
                    data = json.loads(blob.download_as_text())
                    logs.append(data)
            return logs
        except Exception:
            return []
    else:
        # Local fallback
        if os.path.exists(LOCAL_FALLBACK_FILE):
            with open(LOCAL_FALLBACK_FILE, "r") as f:
                return json.load(f)
        return []


def _save_log(log_entry: dict):
    bucket = _get_bucket()
    log_id = log_entry["log_id"]
    if bucket:
        try:
            blob = bucket.blob(f"{OVERRIDE_LOG_PREFIX}{log_id}.json")
            blob.upload_from_string(
                json.dumps(log_entry, indent=2),
                content_type="application/json"
            )
        except Exception as e:
            print(f"GCS save failed: {e}")
    else:
        # Local fallback — append to list
        logs = _load_all_logs()
        # Replace if exists, else append
        existing = next((i for i, l in enumerate(logs) if l["log_id"] == log_id), None)
        if existing is not None:
            logs[existing] = log_entry
        else:
            logs.append(log_entry)
        with open(LOCAL_FALLBACK_FILE, "w") as f:
            json.dump(logs, f, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def create_override_log(
    decision_id: str,
    recommendation: str,       # SCALE / HOLD / REDUCE / BLOCK
    confidence_score: float,
    campaign_name: str = "",
    risk_profile: str = "",
) -> str:
    """
    Called immediately after MDU Engine outputs a recommendation.
    Creates a pending log entry awaiting the 72-hour follow-up.
    Returns the log_id.
    """
    log_id = str(uuid.uuid4())
    entry = {
        "log_id": log_id,
        "decision_id": decision_id,
        "recommendation": recommendation,
        "confidence_score": confidence_score,
        "campaign_name": campaign_name,
        "risk_profile": risk_profile,
        "created_at": datetime.utcnow().isoformat(),
        "followup_due_at": (datetime.utcnow() + timedelta(hours=72)).isoformat(),
        "status": "pending",           # pending → completed
        "was_followed": None,          # True / False
        "override_reason": None,       # free text if overridden
        "outcome_description": None,   # what actually happened
        "outcome_direction": None,     # improved / worsened / unchanged
        "outcome_logged_at": None,
    }
    _save_log(entry)
    return log_id


def record_outcome(
    log_id: str,
    was_followed: bool,
    override_reason: Optional[str],
    outcome_description: str,
    outcome_direction: str,   # "improved" | "worsened" | "unchanged"
):
    """
    Called when the operator submits the 72-hour follow-up form.
    Updates the log entry with outcome data.
    """
    logs = _load_all_logs()
    entry = next((l for l in logs if l["log_id"] == log_id), None)
    if not entry:
        return False

    entry["was_followed"] = was_followed
    entry["override_reason"] = override_reason
    entry["outcome_description"] = outcome_description
    entry["outcome_direction"] = outcome_direction
    entry["outcome_logged_at"] = datetime.utcnow().isoformat()
    entry["status"] = "completed"

    _save_log(entry)
    return True


def get_pending_followups() -> list:
    """
    Returns all log entries where:
    - status is 'pending'
    - followup_due_at has passed (i.e. 72 hours have elapsed)
    """
    logs = _load_all_logs()
    now = datetime.utcnow()
    pending = []
    for log in logs:
        if log["status"] == "pending":
            due = datetime.fromisoformat(log["followup_due_at"])
            if now >= due:
                pending.append(log)
    return pending


def get_override_summary() -> dict:
    """
    Returns aggregate stats for the override dashboard:
    - total decisions logged
    - follow rate (% recommendations followed)
    - override rate by action class
    - outcome correlation (followed vs overridden outcomes)
    """
    logs = _load_all_logs()
    completed = [l for l in logs if l["status"] == "completed"]

    if not completed:
        return {"total": 0, "completed": 0}

    total = len(logs)
    followed = [l for l in completed if l["was_followed"]]
    overridden = [l for l in completed if not l["was_followed"]]

    # Override rate by action class
    action_classes = ["SCALE", "HOLD", "REDUCE", "BLOCK"]
    override_by_class = {}
    for action in action_classes:
        action_logs = [l for l in completed if l["recommendation"] == action]
        action_overridden = [l for l in action_logs if not l["was_followed"]]
        override_by_class[action] = {
            "total": len(action_logs),
            "overridden": len(action_overridden),
            "override_rate": round(len(action_overridden) / len(action_logs) * 100, 1)
            if action_logs else 0,
        }

    # Outcome correlation
    followed_improved = len([l for l in followed if l["outcome_direction"] == "improved"])
    overridden_improved = len([l for l in overridden if l["outcome_direction"] == "improved"])

    return {
        "total": total,
        "completed": len(completed),
        "pending": len([l for l in logs if l["status"] == "pending"]),
        "follow_rate": round(len(followed) / len(completed) * 100, 1) if completed else 0,
        "override_rate": round(len(overridden) / len(completed) * 100, 1) if completed else 0,
        "override_by_class": override_by_class,
        "outcome_when_followed_improved": round(followed_improved / len(followed) * 100, 1)
        if followed else 0,
        "outcome_when_overridden_improved": round(overridden_improved / len(overridden) * 100, 1)
        if overridden else 0,
        "miscalibration_flags": [
            action for action, stats in override_by_class.items()
            if stats["override_rate"] > 50 and stats["total"] >= 3
        ],
    }