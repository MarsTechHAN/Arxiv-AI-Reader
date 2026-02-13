"""
TOTP auth and invite code validation.
One-time TOTP login, long-lived session (cookie) until user clears.
"""

import pyotp
import secrets
from typing import Optional, Tuple

from .db import get_serving_db


def create_totp_secret() -> str:
    return pyotp.random_base32()


def verify_totp(secret: str, code: str) -> bool:
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)


def generate_invite_code() -> str:
    return secrets.token_urlsafe(12)


def validate_invite_code(code: str) -> bool:
    """Check if invite code exists and unused."""
    db = get_serving_db()
    conn = db._get_conn()
    try:
        row = conn.execute(
            "SELECT 1 FROM invite_codes WHERE code = ? AND used_by IS NULL",
            (code,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def create_session(user_id: int) -> str:
    return get_serving_db().create_session(user_id)


def verify_session(token: str) -> Optional[int]:
    return get_serving_db().get_session_user(token)


def register_prepare(username: str, invite_code: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Step 1: Validate invite, return (totp_secret, totp_uri, error).
    Store pending in cache for 5 min - caller should call register_confirm next.
    """
    db = get_serving_db()
    if db.get_user_by_username(username):
        return None, None, "Username already exists"
    if not validate_invite_code(invite_code):
        return None, None, "Invalid or used invite code"
    secret = create_totp_secret()
    uri = pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name="Arxiv-AI-Reader")
    _pending_registry[(username, invite_code)] = (secret, __import__("time").time())
    return secret, uri, None


# Pending registrations: (username, invite_code) -> (secret, timestamp)
_pending_registry: dict = {}


def _clean_pending():
    import time
    expired = [k for k, (_, ts) in _pending_registry.items() if time.time() - ts > 300]
    for k in expired:
        del _pending_registry[k]


def register_confirm(username: str, invite_code: str, totp_code: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Step 2: Verify TOTP, create user, return (session_token, error).
    """
    _clean_pending()
    db = get_serving_db()
    user = db.get_user_by_username(username)
    if user:
        if not verify_totp(user["totp_secret"], totp_code):
            return None, "Invalid TOTP code"
        return create_session(user["id"]), None

    key = (username, invite_code)
    if key not in _pending_registry:
        return None, "Registration expired or invalid. Please restart from step 1."
    secret, _ = _pending_registry[key]
    del _pending_registry[key]
    if not verify_totp(secret, totp_code):
        return None, "Invalid TOTP code"
    if not validate_invite_code(invite_code):
        return None, "Invite code was used meanwhile"
    try:
        user_id = db.create_user(username, secret, invite_code)
        return create_session(user_id), None
    except Exception as e:
        return None, str(e)


def login_user(username: str, totp_code: str) -> Tuple[Optional[str], Optional[str]]:
    """Login: verify TOTP, return session_token or error."""
    db = get_serving_db()
    user = db.get_user_by_username(username)
    if not user:
        return None, "User not found"
    if not verify_totp(user["totp_secret"], totp_code):
        return None, "Invalid TOTP code"
    token = create_session(user["id"])
    return token, None
