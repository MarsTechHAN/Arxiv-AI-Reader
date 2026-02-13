"""
Auth API routes and login page for serving mode.
"""

import json
import os
from fastapi import APIRouter, Request, Response, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .auth import register_prepare, register_confirm, login_user, generate_invite_code
from .db import get_serving_db
from .middleware import SESSION_COOKIE

router = APIRouter(prefix="/auth", tags=["auth"])
ADMIN_SECRET = os.environ.get("ARXIV_INVITE_SECRET", "")


class RegisterPrepareRequest(BaseModel):
    username: str
    invite_code: str


class RegisterConfirmRequest(BaseModel):
    username: str
    invite_code: str
    totp_code: str


class LoginRequest(BaseModel):
    username: str
    totp_code: str


@router.post("/register_prepare")
async def api_register_prepare(req: RegisterPrepareRequest):
    secret, uri, err = register_prepare(req.username.strip(), req.invite_code.strip())
    if err:
        return {"ok": False, "error": err}
    return {"ok": True, "totp_secret": secret, "totp_uri": uri}


@router.post("/register_confirm")
async def api_register_confirm(req: RegisterConfirmRequest):
    token, err = register_confirm(
        req.username.strip(), req.invite_code.strip(), req.totp_code.strip()
    )
    if err:
        return {"ok": False, "error": err}
    resp = Response(content=json.dumps({"ok": True}))
    resp.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 365)
    return resp


@router.post("/login")
async def api_login(req: LoginRequest):
    token, err = login_user(req.username.strip(), req.totp_code.strip())
    if err:
        return {"ok": False, "error": err}
    resp = Response(content=json.dumps({"ok": True}))
    resp.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 365)
    return resp


@router.post("/logout")
async def api_logout():
    resp = Response(content=json.dumps({"ok": True}))
    resp.delete_cookie(SESSION_COOKIE)
    return resp


class CreateInviteRequest(BaseModel):
    code: str = None  # If None, generate


@router.post("/create_invite")
async def api_create_invite(req: CreateInviteRequest, x_admin_secret: str = Header(None, alias="X-Admin-Secret")):
    """Create invite code. Requires X-Admin-Secret header matching ARXIV_INVITE_SECRET."""
    if not ADMIN_SECRET or x_admin_secret != ADMIN_SECRET:
        return {"ok": False, "error": "Unauthorized"}
    code = (req.code or "").strip() or generate_invite_code()
    get_serving_db().create_invite_code(code)
    return {"ok": True, "code": code}


LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Login - Arxiv AI Reader</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center; background: #0f0f12; color: #e0e0e0; }
    .card { background: #1a1a1f; padding: 2rem; border-radius: 12px; width: 100%; max-width: 380px; box-shadow: 0 4px 24px rgba(0,0,0,0.4); }
    h1 { margin: 0 0 1.5rem; font-size: 1.25rem; }
    input { width: 100%; padding: 0.75rem; margin-bottom: 1rem; border: 1px solid #333; border-radius: 8px; background: #25252b; color: #e0e0e0; font-size: 1rem; }
    button { width: 100%; padding: 0.75rem; background: #3b82f6; color: white; border: none; border-radius: 8px; font-size: 1rem; cursor: pointer; }
    button:hover { background: #2563eb; }
    button.secondary { background: #333; }
    .error { color: #ef4444; font-size: 0.875rem; margin-top: 0.5rem; }
    .step { display: none; }
    .step.active { display: block; }
    .hint { font-size: 0.8rem; color: #888; margin-top: 0.5rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Arxiv AI Reader</h1>
    <div id="error" class="error"></div>
    <form id="form">
      <div class="step active" id="stepCredentials">
        <input type="text" id="username" placeholder="Username" required autocomplete="username">
        <input type="text" id="inviteCode" placeholder="Invite code (new users)">
        <div class="hint">Existing users: leave invite code empty</div>
        <button type="submit">Next</button>
      </div>
      <div class="step" id="stepTotp">
        <div id="totpSecretBox" style="display:none;background:#25252b;padding:0.75rem;border-radius:8px;margin-bottom:1rem;font-family:monospace;word-break:break-all;"></div>
        <div id="totpHint" class="hint"></div>
        <input type="text" id="totpCode" placeholder="6-digit code" maxlength="6" pattern="[0-9]{6}" autocomplete="one-time-code">
        <button type="submit">Verify & Login</button>
        <button type="button" class="secondary" onclick="backToStep1()" style="margin-top:0.5rem">Back</button>
      </div>
    </form>
  </div>
  <script>
    const API = window.location.origin + '/auth';
    let step = 1;
    let username = '', inviteCode = '';

    function showError(msg) { document.getElementById('error').textContent = msg || ''; }
    function showStep(s) {
      document.querySelectorAll('.step').forEach(el => el.classList.remove('active'));
      document.getElementById(s).classList.add('active');
    }

    async function step1() {
      username = document.getElementById('username').value.trim();
      inviteCode = document.getElementById('inviteCode').value.trim();
      if (!username) { showError('Enter username'); return; }
      if (inviteCode) {
        const r = await fetch(API + '/register_prepare', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({username, invite_code: inviteCode})
        });
        const d = await r.json();
        if (!d.ok) { showError(d.error || 'Failed'); return; }
        document.getElementById('totpSecretBox').textContent = d.totp_secret;
        document.getElementById('totpSecretBox').style.display = 'block';
        document.getElementById('totpHint').textContent = 'Add this secret to your authenticator app (Google Authenticator, Authy, etc.)';
        step = 2;
        showStep('stepTotp');
      } else {
        document.getElementById('totpSecretBox').style.display = 'none';
        document.getElementById('totpHint').textContent = 'Enter 6-digit code from your authenticator app';
        step = 2;
        showStep('stepTotp');
      }
      showError('');
    }

    async function step2() {
      const totp = document.getElementById('totpCode').value.trim();
      if (!totp || totp.length !== 6) { showError('Enter 6-digit code'); return; }
      const body = inviteCode
        ? {username, invite_code: inviteCode, totp_code: totp}
        : {username, totp_code: totp};
      const endpoint = inviteCode ? API + '/register_confirm' : API + '/login';
      const r = await fetch(endpoint, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body), credentials: 'include'
      });
      const d = await r.json();
      if (!d.ok) { showError(d.error || 'Failed'); return; }
      window.location.href = '/';
    }

    function backToStep1() {
      step = 1;
      showStep('stepCredentials');
      showError('');
    }

    document.getElementById('form').addEventListener('submit', async e => {
      e.preventDefault();
      if (step === 1) await step1();
      else await step2();
    });
  </script>
</body>
</html>
"""


def get_login_router():
    """Return router that serves /login page."""
    r = APIRouter()
    @r.get("/login", response_class=HTMLResponse)
    async def login_page():
        return LOGIN_HTML
    return r
