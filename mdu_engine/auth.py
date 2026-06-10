"""
MDU Engine — Authentication
Magic link / OTP login system.
Stores OTP records in GCS (same bucket as history + feedback).
No passwords. User enters email → receives 6-digit code → verified → session started.

Environment variables (reuses existing):
    MDU_FEEDBACK_BUCKET  — GCS bucket name
"""

from __future__ import annotations

import json
import os
import random
import string
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

# ── Configuration ──────────────────────────────────────────
AUTH_BUCKET  = os.environ.get("MDU_FEEDBACK_BUCKET", "")
OTP_BLOB     = "auth/otp_store.json"
USERS_BLOB   = "auth/users.json"
OTP_EXPIRY_MINUTES = 10


# ── GCS helpers ────────────────────────────────────────────

def _gcs_available() -> bool:
    if not AUTH_BUCKET:
        return False
    try:
        from google.cloud import storage  # noqa
        return True
    except ImportError:
        return False


def _gcs_read(blob_name: str) -> Any:
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(AUTH_BUCKET)
    blob = bucket.blob(blob_name)
    if not blob.exists():
        return {}
    try:
        return json.loads(blob.download_as_bytes().decode("utf-8"))
    except Exception:
        return {}


def _gcs_write(blob_name: str, data: Any) -> None:
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(AUTH_BUCKET)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        content_type="application/json",
    )


# ── Local fallback ─────────────────────────────────────────

LOCAL_OTP_FILE   = "auth_otp_local.json"
LOCAL_USERS_FILE = "auth_users_local.json"


def _local_read(filepath: str) -> Any:
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    return {}


def _local_write(filepath: str, data: Any) -> None:
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def _read(blob: str, local: str) -> Any:
    if _gcs_available():
        return _gcs_read(blob)
    return _local_read(local)


def _write(blob: str, local: str, data: Any) -> None:
    if _gcs_available():
        _gcs_write(blob, data)
    else:
        _local_write(local, data)


# ── OTP generation ─────────────────────────────────────────

def generate_otp() -> str:
    return "".join(random.choices(string.digits, k=6))


def store_otp(email: str, name: str, otp: str) -> None:
    """Store OTP with expiry. One active OTP per email at a time."""
    store = _read(OTP_BLOB, LOCAL_OTP_FILE)
    store[email.lower().strip()] = {
        "name": name.strip(),
        "otp": otp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)).isoformat(),
        "used": False,
    }
    _write(OTP_BLOB, LOCAL_OTP_FILE, store)


def verify_otp(email: str, otp_input: str) -> dict:
    """
    Verify OTP for email.
    Returns: {"success": bool, "name": str, "error": str}
    """
    email = email.lower().strip()
    store = _read(OTP_BLOB, LOCAL_OTP_FILE)

    if email not in store:
        return {"success": False, "name": "", "error": "No code found for this email. Please request a new one."}

    record = store[email]

    if record.get("used"):
        return {"success": False, "name": "", "error": "This code has already been used. Please request a new one."}

    expires_at = datetime.fromisoformat(record["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        return {"success": False, "name": "", "error": "Code expired. Please request a new one."}

    if record["otp"] != otp_input.strip():
        return {"success": False, "name": "", "error": "Incorrect code. Please try again."}

    # Mark as used
    store[email]["used"] = True
    _write(OTP_BLOB, LOCAL_OTP_FILE, store)

    # Register or update user
    _register_user(email, record["name"])

    return {"success": True, "name": record["name"], "error": ""}


# ── User registry ──────────────────────────────────────────

def _register_user(email: str, name: str) -> None:
    """Create or update user record on successful login."""
    users = _read(USERS_BLOB, LOCAL_USERS_FILE)
    email = email.lower().strip()

    if email not in users:
        users[email] = {
            "name": name,
            "email": email,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_login": datetime.now(timezone.utc).isoformat(),
            "login_count": 1,
        }
    else:
        users[email]["last_login"] = datetime.now(timezone.utc).isoformat()
        users[email]["login_count"] = users[email].get("login_count", 0) + 1
        if name:
            users[email]["name"] = name

    _write(USERS_BLOB, LOCAL_USERS_FILE, users)


def get_user(email: str) -> Optional[Dict[str, Any]]:
    """Return user record or None."""
    users = _read(USERS_BLOB, LOCAL_USERS_FILE)
    return users.get(email.lower().strip())


def get_all_users() -> list:
    """Return all registered users — for admin view."""
    users = _read(USERS_BLOB, LOCAL_USERS_FILE)
    return list(users.values())