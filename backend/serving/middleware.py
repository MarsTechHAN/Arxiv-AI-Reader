"""
Auth middleware for serving mode. Injects current_user_id into request state.
"""

from starlette.requests import Request
from starlette.responses import Response, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Optional

from .db import get_serving_db

SESSION_COOKIE = "arxiv_session"
LOGIN_PATH = "/login"
PUBLIC_PATHS = {"/login", "/auth/", "/static", "/api/health"}


class ServingAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        return await serving_auth_middleware(request, call_next)


def _get_token_from_cookie(request: Request) -> Optional[str]:
    return request.cookies.get(SESSION_COOKIE)


def get_current_user_id(request: Request) -> Optional[int]:
    """Get user_id from request (set by middleware)."""
    return getattr(request.state, "user_id", None)


async def serving_auth_middleware(request: Request, call_next: Callable) -> Response:
    """Require auth for non-public paths when serving mode is on."""
    path = request.url.path
    for prefix in PUBLIC_PATHS:
        if path == prefix or path.startswith(prefix + "/"):
            response = await call_next(request)
            return response
    if path == "/" or path.startswith("/api/"):
        token = _get_token_from_cookie(request)
        user_id = get_serving_db().get_session_user(token) if token else None
        request.state.user_id = user_id
        from .integrate import set_serving_user_id
        set_serving_user_id(user_id)
        if user_id is None and path.startswith("/api/") and path != "/api/health":
            return Response(status_code=401, content='{"detail":"Login required"}', media_type="application/json")
        if user_id is None and (path == "/" or path.startswith("/static")):
            return RedirectResponse(url=LOGIN_PATH, status_code=302)
    response = await call_next(request)
    return response
