"""
Serving mode plugin - multi-user support with TOTP auth, per-user config, paper overlays.
Enable via env: ARXIV_SERVING_MODE=1
"""

from .db import ServingDB, get_serving_db
from .auth import verify_totp, create_totp_secret, validate_invite_code, create_session, verify_session
from .middleware import serving_auth_middleware

__all__ = [
    "ServingDB",
    "get_serving_db",
    "verify_totp",
    "create_totp_secret",
    "validate_invite_code",
    "create_session",
    "verify_session",
    "serving_auth_middleware",
]
