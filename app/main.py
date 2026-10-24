"""NAK AI Detector - web service.

Contract:
- Feature extractor: src/features.py::extract_features(text) -> np.ndarray
  (1-D feature vector). Provided later by the ML pipeline.
- Model file: $MODEL_PATH or "models/nak_detector.joblib" relative to the
  project root. A joblib-dumped sklearn classifier exposing predict_proba.
- POST /api/detect -> {"ai_probability", "verdict", "confidence"} (+ credits_remaining).

If the model file is missing, /healthz reports model_loaded:false and
/api/detect returns 503 {"error": "model not trained yet"}.

Accounts & credits: clients log in (or use their X-API-Key); every scan
(/api/detect or /api/scan) costs 1 credit, deducted AFTER a successful
scan. Admin (ADMIN_PASSWORD env) manages accounts from /admin.
Reports and uploads auto-delete 30 minutes after the scan completes.
"""

from __future__ import annotations

import asyncio
import os
import re
import secrets
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)

from app import codecheck as codecheckmod, pages, plag as plagmod, store
from app.parse import ALLOWED_EXTS, parse_document
from app.scan import assemble_report, count_words, make_submission_id
from src.features import extract_features_unscaled

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / "models" / "nak_detector.joblib"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = (PROJECT_ROOT / MODEL_PATH).resolve()

DATA_DIR = Path(os.environ.get("DATA_DIR", "")) if os.environ.get("DATA_DIR") else None
if DATA_DIR is None:
    # Railway volume pattern (same as the Telegram bot): if /data exists
    # (volume mounted), keep the sqlite accounts DB there so accounts and
    # credits survive restarts/redeploys. Otherwise use local ./data.
    # Reports/uploads are always ephemeral (30-min auto-delete anyway).
    DATA_DIR = Path("/data") if Path("/data").is_dir() else PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "nak.db"
REPORTS_DIR = DATA_DIR / "reports"
UPLOADS_DIR = DATA_DIR / "uploads"
TMP_DIR = DATA_DIR / "tmp"

MIN_CHARS = 50
MAX_CHARS = 10_000
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100MB
SCAN_TTL = 30 * 60  # reports auto-delete 30 min after scan
SWEEP_INTERVAL = 5 * 60
COOKIE_NAME = "nak_session"
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
LOGO_PATH = (
    PROJECT_ROOT
    / "assets"
    / "media-generation-nak-ai-detector-logo-0-f792afd1-7bc0-47de-821f-2c86a5388273.webp"
)

model = None  # loaded once at startup (legacy v17 path; kept for reference)
v21_bundle = None  # NAK v21 AI detector (187-dim stylometric + frag rule) — LIVE
V21_BUNDLE_PATH = PROJECT_ROOT / "models" / "candidates" / "v21_187_bundle.joblib"
plag_oa: plagmod.OAIndex | None = None  # offline plagiarism reference index
PLAG_INDEX_PATH = PROJECT_ROOT / "plag_index" / "plag_index.db"
plag_web: plagmod.OAIndex | None = None  # web academic index (mdpi, frontiersin, etc.)
WEB_INDEX_PATH = PROJECT_ROOT / "plag_index" / "plag_web.db"


def _ensure_plag_index() -> None:
    """One-time background download of the mega plagiarism index.

    Railway's `railway up` upload caps out below the 1.4GB index size, so the
    full index lives on external free storage (Hugging Face Datasets) and is
    fetched on first boot when PLAG_INDEX_URL is set. Until it arrives,
    similarity gracefully falls back to doc-vs-doc only.
    """
    import threading
    import urllib.request

    url = os.environ.get("PLAG_INDEX_URL", "").strip()
    if not url:
        return

    def _dl() -> None:
        global plag_oa
        try:
            if (PLAG_INDEX_PATH.exists()
                    and PLAG_INDEX_PATH.stat().st_size > 1_000_000_000):
                return
            print("plag-index: downloading mega index (~1.4GB, one-time)...",
                  flush=True)
            PLAG_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = PLAG_INDEX_PATH.with_suffix(".db.download")
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < 1_000_000_000:
                print("plag-index: download too small, discarding",
                      flush=True)
                tmp.unlink(missing_ok=True)
                return
            tmp.rename(PLAG_INDEX_PATH)
            print("plag-index: download complete", flush=True)
        except Exception as e:  # never break boot on this
            print(f"plag-index: download failed ({e})", flush=True)

    threading.Thread(target=_dl, daemon=True).start()


def get_plag_oa() -> plagmod.OAIndex | None:
    """Lazily (re)bind the plagiarism index once the file exists."""
    global plag_oa
    if plag_oa is None and PLAG_INDEX_PATH.exists():
        try:
            idx = plagmod.OAIndex(PLAG_INDEX_PATH)
            if idx.available():
                plag_oa = idx
        except Exception:
            plag_oa = None
    return plag_oa


def _ensure_web_index() -> None:
    """One-time background download of the web academic plagiarism index.

    Covers mdpi.com, frontiersin.org and other web sources Turnitin checks.
    Fetched on first boot when WEB_INDEX_URL is set. 576MB, under Railway limits.
    """
    import threading
    import urllib.request

    url = os.environ.get("WEB_INDEX_URL", "").strip()
    if not url:
        return

    def _dl() -> None:
        global plag_web
        try:
            if (WEB_INDEX_PATH.exists()
                    and WEB_INDEX_PATH.stat().st_size > 400_000_000):
                return
            print("web-index: downloading academic index (~576MB, one-time)...",
                  flush=True)
            WEB_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = WEB_INDEX_PATH.with_suffix(".db.download")
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < 400_000_000:
                print("web-index: download too small, discarding",
                      flush=True)
                tmp.unlink(missing_ok=True)
                return
            tmp.rename(WEB_INDEX_PATH)
            print("web-index: download complete", flush=True)
        except Exception as e:  # never break boot on this
            print(f"web-index: download failed ({e})", flush=True)

    threading.Thread(target=_dl, daemon=True).start()


def get_plag_web() -> plagmod.OAIndex | None:
    """Lazily (re)bind the web academic index once the file exists."""
    global plag_web
    if plag_web is None and WEB_INDEX_PATH.exists():
        try:
            idx = plagmod.OAIndex(WEB_INDEX_PATH)
            if idx.available():
                plag_web = idx
        except Exception:
            plag_web = None
    return plag_web


# ---------------------------------------------------------------- startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, v21_bundle, plag_oa, plag_web
    for d in (DATA_DIR, REPORTS_DIR, UPLOADS_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
    store.init_db(DB_PATH)
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
        except Exception:
            model = None
    # NAK v21 AI detector — the live detection model (replaced v17 2026-10-23).
    if V21_BUNDLE_PATH.exists():
        try:
            v21_bundle = joblib.load(V21_BUNDLE_PATH)
        except Exception:
            v21_bundle = None
    # Offline plagiarism reference library. Ships with the deploy when small
    # enough; otherwise fetched on first boot via PLAG_INDEX_URL (see
    # _ensure_plag_index). Absent -> similarity falls back to doc-vs-doc.
    try:
        idx = plagmod.OAIndex(PLAG_INDEX_PATH)
        if idx.available():
            idx.open()
            plag_oa = idx
    except Exception:
        plag_oa = None
    _ensure_plag_index()
    # Web academic index (mdpi.com, frontiersin.org, etc.) - 576MB
    try:
        widx = plagmod.OAIndex(WEB_INDEX_PATH)
        if widx.available():
            widx.open()
            plag_web = widx
    except Exception:
        plag_web = None
    _ensure_web_index()
    asyncio.create_task(_sweeper())
    yield


app = FastAPI(title="NAK AI Detector", lifespan=lifespan)


async def _sweeper():
    while True:
        await asyncio.sleep(SWEEP_INTERVAL)
        try:
            await asyncio.to_thread(_prune_once)
        except Exception:
            pass


def _prune_once():
    now = time.time()
    for scan in store.expired_scans(DB_PATH, now):
        try:
            base = Path(scan["report_path"])
            base.unlink(missing_ok=True)
            base.parent.joinpath(f"{base.stem}.ai.html").unlink(missing_ok=True)
            base.parent.joinpath(f"{base.stem}.sim.html").unlink(missing_ok=True)
        except OSError:
            pass
        store.delete_scan(DB_PATH, scan["scan_id"])
    store.prune_sessions(DB_PATH)
    # safety net: orphan raw uploads / temp parts older than the TTL
    for d in (UPLOADS_DIR, TMP_DIR):
        if not d.exists():
            continue
        for p in d.iterdir():
            try:
                if now - p.stat().st_mtime > SCAN_TTL:
                    p.unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------- auth

def _session_token(request: Request) -> str | None:
    return request.cookies.get(COOKIE_NAME)


def get_client(request: Request) -> dict | None:
    """Client auth: session cookie OR X-API-Key header (per-user key or
    admin-created service key)."""
    token = _session_token(request)
    if token:
        user = store.get_session_user(DB_PATH, token)
        if user and not user.get("is_admin"):
            return user
    key = request.headers.get("x-api-key")
    if key:
        user = store.get_user_by_api_key(DB_PATH, key)
        if user:
            return user
        svc = store.get_service_api_key(DB_PATH, key)
        if svc:
            # service key acts as a client identity for quota purposes
            return {"username": f"svc:{svc['key_id']}", "is_service_key": True,
                    "service_key": svc, "credits": 10**9}
    return None


def get_requester(request: Request) -> tuple[dict | None, bool]:
    """Return (user_dict, is_admin). Admin sessions are accepted here so the
    admin can scan from the panel; admin scans never touch credits."""
    admin = get_admin(request)
    if admin:
        return admin, True
    return get_client(request), False


def get_admin(request: Request) -> dict | None:
    token = _session_token(request)
    if token:
        user = store.get_session_user(DB_PATH, token)
        if user and user.get("is_admin"):
            return user
    return None


def _require_client(request: Request):
    user = get_client(request)
    if not user:
        return None, JSONResponse(
            status_code=401,
            content={"error": "authentication required - log in or send your X-API-Key header"},
        )
    if user.get("suspended"):
        return None, JSONResponse(status_code=403, content={"error": "account suspended"})
    return user, None


def _require_client_or_admin(request: Request):
    """Auth for scan endpoints: accepts client sessions, API keys, AND admin
    sessions (so the admin can scan from the panel). Admin scans never touch
    credits. Returns (user, is_admin, error_response)."""
    user, is_admin = get_requester(request)
    if not user:
        return None, False, JSONResponse(
            status_code=401,
            content={"error": "authentication required - log in or send your X-API-Key header"},
        )
    if user.get("suspended"):
        return None, False, JSONResponse(status_code=403, content={"error": "account suspended"})
    # per-key daily quota (service keys only; 0 = unlimited)
    if user.get("is_service_key"):
        svc = user["service_key"]
        if not store.check_and_inc_quota(DB_PATH, svc["key_id"], svc["daily_quota"]):
            return None, False, JSONResponse(
                status_code=429, content={"error": "daily quota exceeded"})
        # per-key total credits (-1 = unlimited). Only peeked here so a failed
        # upload/scan never burns a credit; 1 credit is consumed per
        # *completed* scan in the scan handlers below.
        if not store.api_key_has_credit(DB_PATH, svc["key_id"]):
            return None, False, JSONResponse(
                status_code=402, content={"error": "insufficient credits - contact admin to top up"})
        store.touch_service_api_key(DB_PATH, svc["key_id"])
    return user, is_admin, None


def _set_session_cookie(response, username: str):
    token = store.new_session(DB_PATH, username)
    response.set_cookie(
        COOKIE_NAME, token, httponly=True, samesite="lax",
        max_age=store.SESSION_TTL, path="/",
    )
    return response


# ---------------------------------------------------------------- detection core
# NAK v21 (live since 2026-10-23): 187-dim stylometric window LR + frag rule.
# Per-window p from the v21 bundle; doc AI% = 100*max(p) if frag>0.6 else
# 100*share(p>=0.5). v17 (RoutedEnsemble, share>=0.9) retired to
# models/nak_detector_v17_backup.joblib.

def score_text(text: str) -> float:
    """Score one window of text with the v21 model -> P(AI) in [0,1]."""
    if v21_bundle is None:
        raise RuntimeError("v21 model bundle not loaded")
    vec = extract_features_unscaled(text)
    vec = np.asarray(vec, dtype=np.float64).reshape(1, -1)
    if vec.shape[1] != 187:
        fixed = np.zeros((1, 187))
        n = min(187, vec.shape[1])
        fixed[:, :n] = vec[:, :n]
        vec = fixed
    p = float(v21_bundle["clf"].predict_proba(
        v21_bundle["scaler"].transform(vec))[0][1])
    return min(max(p, 0.0), 1.0)


def verdict_for(p: float) -> str:
    if p >= 0.6:
        return "likely_ai"
    if p <= 0.4:
        return "likely_human"
    return "uncertain"


# ---------------------------------------------------------------- pages

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    user = get_client(request)
    if user:
        user = store.get_user(DB_PATH, user["username"]) or user  # fresh balance
        return pages.dashboard(user)
    return pages.landing_login()


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return RedirectResponse("/", status_code=303)





@app.post("/login")
async def login(request: Request):
    form = await request.form()
    username = str(form.get("username", "")).strip()
    password = str(form.get("password", ""))
    user = store.get_user(DB_PATH, username)
    ok = bool(
        user and not user.get("is_admin")
        and store.verify_password(password, user.get("pw_salt", ""), user.get("pw_hash", ""))
        and not user.get("suspended")
    )
    if not ok:
        return HTMLResponse(pages.landing_login("Invalid username or password."), status_code=401)
    resp = RedirectResponse("/", status_code=303)
    return _set_session_cookie(resp, username)


@app.get("/logout")
def logout(request: Request):
    token = _session_token(request)
    if token:
        store.delete_session(DB_PATH, token)
    resp = RedirectResponse("/", status_code=303)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


# ---------------------------------------------------------------- admin

@app.get("/admin", response_class=HTMLResponse)
def admin_home(request: Request):
    if not ADMIN_PASSWORD:
        return pages.admin_login(misconfigured=True)
    if get_admin(request):
        return pages.admin_panel(store.list_users(DB_PATH),
                                 repo_docs=store.list_repo_docs(DB_PATH),
                                 **_admin_panel_kwargs())
    return pages.admin_login()


@app.post("/admin/login")
async def admin_login(request: Request):
    form = await request.form()
    if not ADMIN_PASSWORD or str(form.get("password", "")) != ADMIN_PASSWORD:
        return HTMLResponse(pages.admin_login("Wrong password."), status_code=401)
    resp = RedirectResponse("/admin", status_code=303)
    return _set_session_cookie(resp, "admin")


@app.get("/admin/logout")
def admin_logout(request: Request):
    token = _session_token(request)
    if token:
        store.delete_session(DB_PATH, token)
    resp = RedirectResponse("/admin", status_code=303)
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


def _need_admin(request: Request):
    if not get_admin(request):
        return JSONResponse(status_code=403, content={"error": "admin only"})
    return None


@app.post("/admin/create", response_class=HTMLResponse)
async def admin_create(request: Request):
    denied = _need_admin(request)
    if denied:
        return denied
    form = await request.form()
    username = str(form.get("username", "")).strip()
    password = str(form.get("password", ""))
    try:
        credits = max(0, int(form.get("credits", 0) or 0))
    except ValueError:
        credits = 0
    if not username or not password:
        return HTMLResponse(pages.admin_panel(store.list_users(DB_PATH), "Username and password required."))
    try:
        user = store.create_user(DB_PATH, username, password, credits)
    except ValueError as e:
        return HTMLResponse(pages.admin_panel(store.list_users(DB_PATH), str(e)))
    # API keys are never shown to clients: the per-user key stays in the DB
    # for header auth only. Service-level keys are managed under API Access.
    return pages.admin_created(username)


@app.post("/admin/credits", response_class=HTMLResponse)
async def admin_credits(request: Request):
    denied = _need_admin(request)
    if denied:
        return denied
    form = await request.form()
    username = str(form.get("username", ""))
    try:
        delta = int(form.get("delta", 0) or 0)
    except ValueError:
        delta = 0
    try:
        new_balance = store.add_credits(DB_PATH, username, delta)
        msg = f"{username}: balance now {new_balance} credits."
    except ValueError as e:
        msg = str(e)
    return HTMLResponse(pages.admin_panel(store.list_users(DB_PATH), msg))


@app.post("/admin/suspend", response_class=HTMLResponse)
async def admin_suspend(request: Request):
    denied = _need_admin(request)
    if denied:
        return denied
    form = await request.form()
    username = str(form.get("username", ""))
    user = store.get_user(DB_PATH, username)
    if user and not user.get("is_admin"):
        store.set_suspended(DB_PATH, username, not user.get("suspended"))
        msg = f"{username} {'suspended' if not user.get('suspended') else 'unsuspended'}."
    else:
        msg = "Unknown user."
    return HTMLResponse(pages.admin_panel(store.list_users(DB_PATH), msg))


@app.post("/admin/repo/delete", response_class=HTMLResponse)
async def admin_repo_delete(request: Request):
    """Delete a document's fingerprints from the comparison repository
    (by doc_ref or filename substring)."""
    denied = _need_admin(request)
    if denied:
        return denied
    form = await request.form()
    doc_ref = str(form.get("doc_ref", "")).strip()
    filename = str(form.get("filename", "")).strip()
    if doc_ref:
        docs, fps = store.delete_doc_fps_by_ref(DB_PATH, doc_ref)
        msg = f"Removed {docs} document(s), {fps:,} fingerprints from the repository."
    elif filename:
        docs, fps = store.delete_doc_fps_by_filename(DB_PATH, filename)
        msg = f"Removed {docs} document(s), {fps:,} fingerprints from the repository."
    else:
        msg = "Enter a filename (or part of it)."
    return HTMLResponse(pages.admin_panel(
        store.list_users(DB_PATH), msg,
        repo_docs=store.list_repo_docs(DB_PATH),
                                 **_admin_panel_kwargs()))


# ------------------------------------------------- admin: API key management

def _admin_panel_kwargs() -> dict:
    """Common kwargs for admin_panel renders (API keys list + ai_min_words)."""
    try:
        mw = int(store.get_setting(DB_PATH, "ai_min_words", "300") or 300)
    except ValueError:
        mw = 300
    return {"api_keys": _api_key_display_list(), "ai_min_words": mw}


def _api_key_display_list() -> list[dict]:
    """Key rows for the admin API Access UI (never includes key material)."""
    import datetime as _dt
    out = []
    for k in store.list_service_api_keys(DB_PATH):
        created = _dt.datetime.fromtimestamp(
            k["created_at"], _dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        last = (None if not k.get("last_used_at") else
                _dt.datetime.fromtimestamp(
                    k["last_used_at"], _dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        out.append({
            "id": k["key_id"],
            "label": (k["label"] or "") + (" [revoked]" if k["revoked"] else ""),
            "owner": k["owner"],
            "created": created,
            "last_used": last,
            "daily_quota": int(k["daily_quota"] or 0),
            "credits": int(k.get("credits", 0)),
        })
    return out


@app.get("/admin/api/keys", response_class=HTMLResponse)
def admin_api_keys_page(request: Request):
    denied = _need_admin(request)
    if denied:
        return denied
    return HTMLResponse(pages.admin_api_keys(_api_key_display_list()))


@app.post("/admin/api/keys/create", response_class=HTMLResponse)
async def admin_api_keys_create(request: Request):
    denied = _need_admin(request)
    if denied:
        return denied
    form = await request.form()
    label = str(form.get("label", "")).strip()[:80] or "Untitled key"
    try:
        quota = max(0, int(str(form.get("daily_quota", "0") or 0)))
    except ValueError:
        quota = 0
    try:
        credits = max(0, int(str(form.get("credits", "0") or 0)))
    except ValueError:
        credits = 0
    unlimited = str(form.get("unlimited", "")).strip().lower() in ("1", "true", "yes", "on")
    key_id, raw_key = store.create_service_api_key(
        DB_PATH, label, daily_quota=quota, owner="admin",
        credits=-1 if unlimited else credits)
    # Shown ONCE to the admin only; never stored, never shown to clients.
    return HTMLResponse(pages.admin_key_created(label, raw_key))


@app.post("/admin/api/keys/revoke", response_class=HTMLResponse)
async def admin_api_keys_revoke(request: Request):
    denied = _need_admin(request)
    if denied:
        return denied
    form = await request.form()
    key_id = str(form.get("key_id", "")).strip()
    ok = store.revoke_service_api_key(DB_PATH, key_id) if key_id else False
    msg = f"Key {key_id} revoked." if ok else "Key not found."
    return HTMLResponse(pages.admin_api_keys(_api_key_display_list(), msg=msg))


@app.post("/admin/api/keys/delete", response_class=HTMLResponse)
async def admin_api_keys_delete(request: Request):
    denied = _need_admin(request)
    if denied:
        return denied
    form = await request.form()
    key_id = str(form.get("key_id", "")).strip()
    ok = store.delete_service_api_key(DB_PATH, key_id) if key_id else False
    msg = f"Key {key_id} permanently deleted." if ok else "Key not found or not revoked yet."
    return HTMLResponse(pages.admin_api_keys(_api_key_display_list(), msg=msg))


@app.post("/admin/api/keys/add_credits", response_class=HTMLResponse)
async def admin_api_keys_add_credits(request: Request):
    denied = _need_admin(request)
    if denied:
        return denied
    form = await request.form()
    key_id = str(form.get("key_id", "")).strip()
    try:
        delta = int(str(form.get("credits", "0") or 0))
    except ValueError:
        delta = 0
    if key_id and delta > 0:
        new_bal = store.add_api_key_credits(DB_PATH, key_id, delta)
        msg = f"Added {delta} credits. New balance: {new_bal}." if new_bal >= 0 else "Key not found."
    else:
        msg = "Enter a positive credit amount."
    return HTMLResponse(pages.admin_api_keys(_api_key_display_list(), msg=msg))


# ------------------------------------------------- admin: delete user

@app.post("/admin/delete_user", response_class=HTMLResponse)
async def admin_delete_user(request: Request):
    """Permanently delete a client account and ALL its data (scans, reports,
    repository fingerprints). The UI must ask for explicit confirmation
    before posting here."""
    denied = _need_admin(request)
    if denied:
        return denied
    form = await request.form()
    username = str(form.get("username", "")).strip()
    confirm = str(form.get("confirm", "")).strip().lower()
    if not username or username == "admin":
        msg = "Refusing to delete that account."
    elif confirm not in ("1", "true", "yes", "on", "delete"):
        msg = f"Deletion of '{username}' needs explicit confirmation."
    else:
        counts = store.delete_user(DB_PATH, username)
        msg = (f"Deleted '{username}': {counts['scans']} scan(s), "
               f"{counts['repo_docs']} repo document(s), "
               f"{counts['repo_fps']:,} fingerprints removed.")
    return HTMLResponse(pages.admin_panel(
        store.list_users(DB_PATH), msg,
        repo_docs=store.list_repo_docs(DB_PATH),
                                 **_admin_panel_kwargs()))


# ------------------------------------------------- admin: settings

@app.post("/admin/settings", response_class=HTMLResponse)
async def admin_settings(request: Request):
    """Admin-configurable settings (currently: ai_min_words)."""
    denied = _need_admin(request)
    if denied:
        return denied
    form = await request.form()
    msg = ""
    if "ai_min_words" in form:
        try:
            v = max(0, int(str(form.get("ai_min_words", "300") or 300)))
        except ValueError:
            v = 300
        store.set_setting(DB_PATH, "ai_min_words", str(v))
        msg = f"AI minimum words set to {v}."
    return HTMLResponse(pages.admin_panel(
        store.list_users(DB_PATH), msg,
        repo_docs=store.list_repo_docs(DB_PATH),
                                 **_admin_panel_kwargs()))


# ------------------------------------------------- client: scan history

@app.get("/api/history")
def api_history(request: Request):
    """The caller's own recent scans (within the 30-minute auto-delete
    window). History is inherently transient: entries disappear when the
    underlying scan auto-deletes. Newest first.

    Returns {"scans": [...]} for the dashboard; each entry carries both the
    documented API field names (s_no, file_name, ai_percentage,
    plag_percentage) and the dashboard's aliases (filename, ai_pct, sim_pct)
    plus per-section PDF links."""
    user, err = _require_client(request)
    if err:
        return err
    import datetime as _dt
    scans = store.list_user_scans(DB_PATH, user["username"], limit=50)
    out = []
    for i, s in enumerate(scans, start=1):
        ai_pct = float(s["ai_pct"] or 0.0)
        sim_pct = float(s.get("sim_pct") or 0.0)
        created = _dt.datetime.fromtimestamp(
            s["created_at"], _dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        out.append({
            "s_no": i,
            "scan_id": s["scan_id"],
            "file_name": s["filename"],
            "filename": s["filename"],
            "ai_percentage": ai_pct,
            "ai_pct": ai_pct,
            "plag_percentage": sim_pct,
            "sim_pct": sim_pct,
            "created_at": created,
            "ai_pdf_url": f"/report/{s['scan_id']}/pdf/ai",
            "plag_pdf_url": f"/report/{s['scan_id']}/pdf/similarity",
        })
    return {"scans": out}


# ---------------------------------------------------------------- health

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": v21_bundle is not None,
            "codecheck_available": shutil.which("dolos") is not None}


@app.get("/assets/logo.webp")
def logo():
    if not LOGO_PATH.exists():
        return JSONResponse(status_code=404, content={"error": "logo not found"})
    return FileResponse(LOGO_PATH, media_type="image/webp")


# ---------------------------------------------------------------- API: quick text check

@app.post("/api/detect")
async def detect(request: Request):
    global model
    user, is_admin, err = _require_client_or_admin(request)
    if err:
        return err
    if model is None:
        return JSONResponse(status_code=503, content={"error": "model not trained yet"})

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=422, content={"error": "invalid JSON body"})

    text = body.get("text") if isinstance(body, dict) else None
    if not isinstance(text, str) or not text.strip():
        return JSONResponse(status_code=422, content={"error": "text is required"})
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return JSONResponse(
            status_code=422,
            content={"error": f"text must be between {MIN_CHARS} and {MAX_CHARS} characters"},
        )
    if not is_admin and int(user.get("credits", 0)) < 1:
        return JSONResponse(
            status_code=402,
            content={"error": "Out of credits - contact admin to top up."},
        )

    try:
        p = await asyncio.to_thread(score_text, text)
    except Exception:
        return JSONResponse(status_code=500, content={"error": "detection failed"})

    credits_remaining = None
    if not is_admin and not user.get("is_service_key"):
        credits_remaining = store.add_credits(DB_PATH, user["username"], -1)
    elif user.get("is_service_key"):
        # 1 credit per completed scan (peeked at auth; failed scans above
        # never reach here, so they cost nothing).
        store.use_api_key_credit(DB_PATH, user["service_key"]["key_id"])
    return {
        "ai_probability": p,
        "verdict": verdict_for(p),
        "confidence": round(abs(p - 0.5) * 2, 3),
        "credits_remaining": credits_remaining,
    }


# ---------------------------------------------------------------- API: document scan

def _safe_filename(name: str) -> str:
    name = Path(name or "upload").name
    keep = "".join(c for c in name if c.isalnum() or c in ("-", "_", ".", " "))
    return (keep.strip() or "upload")[:120]


async def _save_upload_streaming(request: Request, dest: Path) -> tuple[int, str, dict]:
    """Stream the raw multipart body to disk without holding it in RAM.

    Reads the request body in chunks and carves out the file bytes between
    the multipart boundaries, so even a 100MB upload never sits fully in
    memory. Trailing small text fields (scan options) are captured.
    Returns (bytes_written, filename from the part headers, text_fields).
    """
    import re as _re

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type or "boundary=" not in content_type:
        raise ValueError("expected multipart/form-data with a file field named 'file'")
    boundary = content_type.split("boundary=")[1].split(";")[0].strip().strip('"').encode()
    delimiter = b"--" + boundary
    end_marker = b"\r\n" + delimiter

    size = 0
    in_file = False
    done = False
    filename = "upload"
    header_buf = b""
    prev_tail = b""
    trailing = b""  # small text fields after the file part (scan options)
    with open(dest, "wb") as f:
        async for chunk in request.stream():
            if done:
                if len(trailing) < 100_000:
                    trailing += chunk
                continue
            data = prev_tail + chunk
            if not in_file:
                header_buf += data
                prev_tail = b""
                idx = header_buf.find(b"\r\n\r\n")
                if idx == -1:
                    if len(header_buf) > 1_000_000:
                        raise ValueError("malformed upload")
                    continue
                header = header_buf[:idx]
                if b'name="file"' not in header and b"name='file'" not in header:
                    raise ValueError("no file field named 'file'")
                m = _re.search(rb'filename="([^"]*)"', header)
                if m:
                    filename = m.group(1).decode("utf-8", "replace") or "upload"
                in_file = True
                data = header_buf[idx + 4:]
                header_buf = b""
            end_idx = data.find(end_marker)
            if end_idx != -1:
                writable, prev_tail, done = data[:end_idx], b"", True
            else:
                keep = len(delimiter) + 4  # hold back a possible split boundary
                if len(data) > keep:
                    writable, prev_tail = data[:-keep], data[-keep:]
                else:
                    writable, prev_tail = b"", data
            size += len(writable)
            if size > MAX_UPLOAD_BYTES:
                raise _TooLarge()
            f.write(writable)
        if in_file and not done and prev_tail:
            # stream ended without a closing boundary; flush what we held back
            size += len(prev_tail)
            if size > MAX_UPLOAD_BYTES:
                raise _TooLarge()
            f.write(prev_tail)
        elif not in_file:
            raise ValueError("no file part found")
    fields: dict[str, str] = {}
    for m in _re.finditer(
            rb'name="([^"]+)"\r\n\r\n([^\r]*)\r\n', trailing):
        try:
            fields[m.group(1).decode("utf-8", "replace")] = \
                m.group(2).decode("utf-8", "replace")
        except Exception:
            pass
    return size, filename, fields


class _TooLarge(Exception):
    pass


_BIB_HEADING_RE = re.compile(
    r"(?m)^[ \t]*(?:\d+[.\)]\s*)?(references|bibliography|works?\s+cited|reference\s+list|literature\s+cited|list\s+of\s+references)[ \t]*:?[ \t]*$",
    re.IGNORECASE,
)


def _bib_heading_offset(unit: dict) -> int | None:
    """Char offset where the bibliography section starts within a unit's
    text (a standalone References/Bibliography/... line), or None."""
    m = _BIB_HEADING_RE.search(unit.get("text", ""))
    return m.start() if m else None


def _scan_file_sync(raw_path: Path, ext: str, scan_id: str, filename: str,
                    store_repo: bool = False,
                    exclude_bibliography: bool = False,
                    exclude_quotes: bool = False,
                    ai_min_words: int = 300,
                    owner: str = "",
                    exclude_cited: bool = False,
                    exclude_small: bool = False,
                    small_threshold: int = 8,
                    small_mode: str = "words") -> dict:
    """Parse -> sentence-split prose -> sliding-window score -> report parts.

    Runs in a worker thread. Fully streaming: the parser yields units one at
    a time, prose sentences flow through a bounded sliding-window scorer
    (8-sentence window, stride 4), and rendered blocks stream into the parts
    file. Only ~a dozen sentences are ever held in RAM. Non-prose units
    (headings, lists, tables) are shown unhighlighted and never scored.

    A plagiarism (similarity) pass runs on the same parse: raw block texts
    stream to a temp file, then winnowing fingerprints are matched against
    the offline reference library + previously scanned documents. One scan
    (AI + similarity) costs a single credit.
    """
    from collections import deque

    from app.scan import (
        AI_THRESHOLD,
        MIN_PROSE_WORDS,
        SentenceScorer,
        append_block_html,
        render_list_html,
        render_prose_paragraph,
        render_unit_html,
        reset_parts,
        split_sentences,
    )
    # NAK v21 doc aggregation (sealed-gate 2026-10-23): every scored window's
    # P(AI) is recorded; frag = share of scored sentences < 12 words.
    # frag > 0.6 -> AI% = 100*max(p); else AI% = 100*share(p >= 0.5).
    window_ps: list[float] = []

    def _recording_score(seg_text: str) -> float:
        p = score_text(seg_text)
        window_ps.append(p)
        return p

    reset_parts(TMP_DIR, scan_id)
    total_words = 0
    total_chars = 0
    prose_words = 0
    sent_count = 0
    ai_sent_count = 0
    pdf_pages: int | None = None
    if ext == ".pdf":
        # real page count for PDFs; docx/txt get a word-count estimate below
        from pypdf import PdfReader

        with open(raw_path, "rb") as _pf:
            pdf_pages = len(PdfReader(_pf).pages)

    scorer = SentenceScorer(_recording_score)
    para_queue: deque[dict] = deque()  # paragraphs awaiting sentence scores
    para_of: dict[int, dict] = {}  # sentence gidx -> paragraph record
    list_buf: list[dict] = []  # pending same-type list items
    # API enrichment: track every scored sentence for ai_spans char offsets.
    sent_texts: dict[int, str] = {}   # gidx -> sentence text (feed order)
    sent_scores: dict[int, float] = {}  # gidx -> AI score
    doc_text_parts: list[str] = []   # document text in parse order (for offsets)
    plag_blocks: list[str] = []     # plag-analyzed block texts (for plag_spans)

    def flush_lists() -> None:
        if list_buf:
            html = render_list_html(list_buf)
            append_block_html(html, sum(count_words(u["text"]) for u in list_buf),
                              TMP_DIR, scan_id)
            list_buf.clear()

    def flush_paragraphs() -> None:
        while para_queue and para_queue[0]["remaining"] == 0:
            p = para_queue.popleft()
            sents = [(p["texts"][g], p["scores"][g]) for g in p["gidxs"]]
            append_block_html(render_prose_paragraph(sents), p["words"], TMP_DIR, scan_id)

    def drain_finalized() -> None:
        nonlocal sent_count, ai_sent_count
        while scorer.finalized:
            g, s = scorer.finalized.popleft()
            sent_count += 1
            sent_scores[g] = s
            if s >= AI_THRESHOLD:
                ai_sent_count += 1
            p = para_of.pop(g)
            p["scores"][g] = s
            p["remaining"] -= 1
        flush_paragraphs()

    try:
        in_bibliography = False
        bib_excluded_words = 0
        for unit in parse_document(raw_path, ext):
            w = count_words(unit["text"])
            total_words += w
            total_chars += len(unit["text"])
            tag = unit["tag"]
            doc_text_parts.append(unit["text"])
            doc_text_parts.append("\n")
            # plagiarism: raw block texts stream to a temp file for the
            # similarity pass (runs after AI scoring, same parse).
            # Exclusion filters apply to plagiarism ONLY -- AI scoring below
            # always sees the full text.
            plag_text = unit["text"]
            if exclude_bibliography:
                if not in_bibliography:
                    off = _bib_heading_offset(unit)
                    if off is not None:
                        in_bibliography = True
                        bib_excluded_words += count_words(unit["text"][off:])
                        plag_text = unit["text"][:off]
                else:
                    bib_excluded_words += w
                    plag_text = ""
            if plag_text.strip():
                plagmod.write_plag_block(plag_text, TMP_DIR, scan_id,
                                         exclude_quotes=exclude_quotes,
                                         exclude_cited=exclude_cited)
                plag_blocks.append(plag_text)
            if tag in ("li-bullet", "li-number"):
                if list_buf and list_buf[0]["tag"] != tag:
                    flush_lists()
                list_buf.append(unit)
                continue
            flush_lists()
            if tag == "p" and unit["text"].strip():
                sentences = split_sentences(unit["text"]) or [unit["text"].strip()]
                rec: dict = {"gidxs": [], "texts": {}, "scores": {},
                             "remaining": 0, "words": w}
                for s in sentences:
                    g = scorer.feed(s)
                    rec["gidxs"].append(g)
                    rec["texts"][g] = s
                    sent_texts[g] = s
                    para_of[g] = rec
                rec["remaining"] = len(rec["gidxs"])
                para_queue.append(rec)
                prose_words += w
                drain_finalized()
            elif w:
                append_block_html(render_unit_html(unit), w, TMP_DIR, scan_id)
        scorer.finish()
        drain_finalized()
    except Exception:
        reset_parts(TMP_DIR, scan_id)
        raise

    # NAK v21 aggregation: frag = share of scored sentences < 12 words.
    # frag > 0.6 -> AI% = 100*max(window p); else 100*share(window p >= 0.5).
    n_scored = len(sent_texts)
    frag = (sum(1 for t in sent_texts.values() if len(t.split()) < 12)
            / n_scored) if n_scored else 0.0
    if window_ps:
        if frag > 0.6:
            ai_pct = round(100.0 * max(window_ps), 1)
        else:
            ai_pct = round(100.0 * sum(1 for p in window_ps if p >= 0.5)
                           / len(window_ps), 1)
    else:
        ai_pct = 0.0
    # AI minimum-words condition (Turnitin parity, default 300): below the
    # threshold the AI section is not scored at all -- but plagiarism still
    # runs fully. Never affects plagiarism.
    ai_too_short = total_words < ai_min_words
    ai_status = "not_scored_too_short" if ai_too_short else "scored"
    if ai_too_short:
        ai_pct = None

    # ai_spans: char offsets + per-sentence AI scores for API consumers.
    # Offsets refer to the document text in parse order (units joined by "\n").
    ai_spans: list[dict] = []
    if not ai_too_short and sent_scores:
        doc_text = "".join(doc_text_parts)
        cursor = 0
        for g in sorted(sent_scores):
            s = sent_texts.get(g, "")
            if not s:
                continue
            idx = doc_text.find(s, cursor)
            if idx < 0:
                idx = doc_text.find(s)  # fallback: search from start
            if idx < 0:
                continue
            ai_spans.append({"start": idx, "end": idx + len(s),
                             "score": round(float(sent_scores[g]), 4)})
            cursor = idx + len(s)

    if pdf_pages is not None:
        pages = max(1, pdf_pages)
    else:
        from app.scan import WORDS_PER_PAGE_ESTIMATE

        pages = max(1, -(-total_words // WORDS_PER_PAGE_ESTIMATE))  # ceil div

    # ---- plagiarism (similarity) pass: same scan, same single credit ----
    # "No repository" is the default: fingerprints are only stored when the
    # user explicitly opts in via store_repo.
    try:
        # Combine mega index + web academic index (if available)
        oa = get_plag_oa()
        web = get_plag_web()
        if oa is not None and web is not None:
            combined = plagmod.MultiOAIndex([oa, web])
        elif oa is not None:
            combined = oa
        elif web is not None:
            combined = web
        else:
            combined = None
        plag_result = plagmod.run_plagiarism(
            scan_id,
            filename,
            TMP_DIR,
            combined,
            doc_lookup_fn=lambda fps: store.query_doc_fps(DB_PATH, fps),
            record_doc_fn=(lambda ref, fn, words, fps: store.record_doc_fps(
                DB_PATH, ref, fn, words, fps, owner)) if store_repo else None,
            exclude_quotes=exclude_quotes,
            exclude_cited=exclude_cited,
            exclude_small=exclude_small,
            small_threshold=small_threshold,
            small_mode=small_mode,
        )
    except Exception:
        plag_result = plagmod._empty_result(0)
    finally:
        plagmod.cleanup_plag_tmp(TMP_DIR, scan_id)

    # plag_spans: char offsets per source for API consumers. Offsets refer to
    # the plagiarism-analyzed text (post exclusion filters), i.e. the
    # concatenation of plag_blocks joined by "\n".
    plag_spans: list[dict] = []
    try:
        from app.plag import split_keep as _split_keep
        # cumulative char offsets of each plag block
        block_offsets: list[int] = []
        _off = 0
        for _b in plag_blocks:
            block_offsets.append(_off)
            _off += len(_b) + 1  # +1 for the "\n" join
        for det in plag_result.get("span_details", []):
            src_id = ":".join(det.get("source_key", []))
            src_title = det.get("source_label", "")
            for bi, ws, we in det.get("blocks", []):
                if bi < 0 or bi >= len(plag_blocks):
                    continue
                orig, _norm = _split_keep(plag_blocks[bi])
                if ws < 0 or we > len(orig) or we <= ws:
                    continue
                # char offsets within the block: sum token lengths
                cs = sum(len(t) for t in orig[:ws])
                ce = sum(len(t) for t in orig[:we])
                base = block_offsets[bi]
                plag_spans.append({"start": base + cs, "end": base + ce,
                                   "source_id": src_id,
                                   "source_title": src_title})
    except Exception:
        plag_spans = []

    return {
        "words": total_words,
        "chars": total_chars,
        "ai_pct": ai_pct,
        "ai_status": ai_status,
        "ai_min_words": ai_min_words,
        "ai_spans": ai_spans,
        "plag_spans": plag_spans,
        "sentences_total": sent_count,
        "sentences_ai": ai_sent_count,
        "prose_words": prose_words,
        "short_notice": prose_words < MIN_PROSE_WORDS,
        "pages": pages,
        "sim_pct": plag_result["sim_pct"],
        "sim_lib_pct": plag_result["lib_pct"],
        "sim_doc_pct": plag_result["doc_pct"],
        "sim_sources": plag_result["sources"],
        "bib_excluded": bib_excluded_words,
        "quote_excluded": plag_result.get(
            "quote_excluded_words", plag_result.get("excluded_words", 0)),
        "cited_excluded": plag_result.get("cited_excluded_words", 0),
        "small_filtered": plag_result.get("small_filtered_words", 0),
        "store_repo": store_repo,
        "exclude_bibliography": exclude_bibliography,
        "exclude_quotes": exclude_quotes,
        "exclude_cited": exclude_cited,
        "exclude_small": exclude_small,
        "small_threshold": small_threshold,
        "small_mode": small_mode,
    }


@app.post("/api/scan")
async def scan_document(request: Request):
    global model
    user, is_admin, err = _require_client_or_admin(request)
    if err:
        return err
    if model is None:
        return JSONResponse(status_code=503, content={"error": "model not trained yet"})
    if not is_admin and int(user.get("credits", 0)) < 1:
        return JSONResponse(
            status_code=402,
            content={"error": "Out of credits - contact admin to top up."},
        )
    # Scan options (query params; also used by the dashboard form):
    #   store_repo=1          save fingerprints to the document repository
    #                         (default OFF -- no repository)
    #   exclude_bibliography=1  plagiarism only: skip References section
    #   exclude_quotes=1        plagiarism only: skip quoted text
    #   exclude_cited=1         plagiarism only: skip in-text citations
    #   exclude_small=1         plagiarism only: drop small matches
    #   small_threshold=N       words mode: 8-250 (default 8);
    #                           percentage mode: 1-100 (default 1)
    #   small_mode=words|percentage  small-match threshold unit
    #   ai_min_words=N         per-scan override of the AI minimum-words
    #                         threshold (default from settings, 300)
    qp = request.query_params
    def _flag(name: str, fields: dict | None = None) -> bool:
        raw = (fields or {}).get(name, "") or qp.get(name, "")
        return raw.strip().lower() in ("1", "true", "yes", "on")
    def _int_param(name: str, fields: dict | None = None) -> int | None:
        raw = (fields or {}).get(name, "") or qp.get(name, "")
        try:
            return int(str(raw).strip())
        except (ValueError, TypeError):
            return None
    def _str_param(name: str, fields: dict | None = None) -> str | None:
        raw = (fields or {}).get(name, "") or qp.get(name, "")
        raw = str(raw).strip().lower()
        return raw or None
    # query params read here; multipart text fields merged after upload parse
    store_repo = _flag("store_repo")
    exclude_bibliography = _flag("exclude_bibliography")
    exclude_quotes = _flag("exclude_quotes")
    exclude_cited = _flag("exclude_cited")
    exclude_small = _flag("exclude_small")
    small_mode_raw = _str_param("small_mode")
    small_threshold_raw = _int_param("small_threshold")
    ai_min_words_override = _int_param("ai_min_words")

    scan_id = secrets.token_urlsafe(16)
    raw_path = UPLOADS_DIR / f"{scan_id}.bin"
    try:
        size, filename, fields = await _save_upload_streaming(request, raw_path)
    except _TooLarge:
        raw_path.unlink(missing_ok=True)
        return JSONResponse(
            status_code=413, content={"error": "File too large - max 100MB per file."}
        )
    except ValueError as e:
        raw_path.unlink(missing_ok=True)
        return JSONResponse(status_code=422, content={"error": str(e)})
    # multipart text fields (dashboard checkboxes) override query params
    store_repo = _flag("store_repo", fields)
    exclude_bibliography = _flag("exclude_bibliography", fields)
    exclude_quotes = _flag("exclude_quotes", fields)
    exclude_cited = _flag("exclude_cited", fields)
    exclude_small = _flag("exclude_small", fields)
    _sm = _str_param("small_mode", fields)
    if _sm is not None:
        small_mode_raw = _sm
    _st = _int_param("small_threshold", fields)
    if _st is not None:
        small_threshold_raw = _st
    small_mode = small_mode_raw if small_mode_raw in ("words", "percentage") else "words"
    if small_mode == "words":
        small_threshold = min(250, max(8, small_threshold_raw if small_threshold_raw is not None else 8))
    else:
        small_threshold = min(100, max(1, small_threshold_raw if small_threshold_raw is not None else 1))
    _mw = _int_param("ai_min_words", fields)
    if _mw is not None:
        ai_min_words_override = _mw
    try:
        ai_min_words = int(store.get_setting(DB_PATH, "ai_min_words", "300") or 300)
    except ValueError:
        ai_min_words = 300
    if ai_min_words_override is not None and ai_min_words_override >= 0:
        ai_min_words = ai_min_words_override
    submission_ts = time.time()  # submission date = when the upload completed
    submission_id = make_submission_id()

    filename = _safe_filename(filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raw_path.unlink(missing_ok=True)
        return JSONResponse(
            status_code=422,
            content={"error": "Only .txt, .pdf, .doc and .docx files are accepted."},
        )

    try:
        stats = await asyncio.to_thread(
            _scan_file_sync, raw_path, ext, scan_id, filename,
            store_repo, exclude_bibliography, exclude_quotes,
            ai_min_words, user["username"],
            exclude_cited=exclude_cited, exclude_small=exclude_small,
            small_threshold=small_threshold, small_mode=small_mode,
        )
    except ValueError as e:
        raw_path.unlink(missing_ok=True)
        return JSONResponse(status_code=422, content={"error": str(e)})
    except Exception:
        raw_path.unlink(missing_ok=True)
        return JSONResponse(status_code=500, content={"error": "scan failed"})
    finally:
        raw_path.unlink(missing_ok=True)  # raw upload never kept

    if stats["words"] == 0:
        return JSONResponse(status_code=422, content={"error": "no readable text found in file"})

    deletes_at = store.record_scan(
        DB_PATH, scan_id, user["username"], filename,
        str(REPORTS_DIR / f"{scan_id}.html"),
        stats["ai_pct"] if stats["ai_pct"] is not None else 0.0,
        stats["words"], SCAN_TTL,
        submission_id=submission_id, file_size=size, pages=stats["pages"],
        sim_pct=stats["sim_pct"],
    )
    # Assemble the final report now that the cover stats are known
    # (chunk HTML was streamed into the parts file during scoring).
    assemble_report(
        scan_id=scan_id,
        filename=filename,
        words=stats["words"],
        chars=stats["chars"],
        ai_pct=stats["ai_pct"] if stats["ai_pct"] is not None else 0.0,
        ai_status=stats["ai_status"],
        ai_min_words=stats["ai_min_words"],
        pages=stats["pages"],
        file_size_bytes=size,
        submission_id=submission_id,
        submission_ts=submission_ts,
        short_notice=stats["short_notice"],
        tmp_dir=TMP_DIR,
        reports_dir=REPORTS_DIR,
        deletes_at_ms=int(deletes_at * 1000),
        sim_pct=stats["sim_pct"],
        sim_lib_pct=stats["sim_lib_pct"],
        sim_doc_pct=stats["sim_doc_pct"],
        sim_sources=stats["sim_sources"],
        store_repo=stats["store_repo"],
        bib_excluded=stats["bib_excluded"],
        quote_excluded=stats["quote_excluded"],
        cited_excluded=stats["cited_excluded"],
        small_filtered=stats["small_filtered"],
    )
    # Admin scans are unlimited: never decrement, never block.
    # Service API keys: 1 credit per completed scan (peeked at auth above;
    # failed uploads/scans returned early, so they cost nothing).
    credits_remaining = None
    if not is_admin and not user.get("is_service_key"):
        credits_remaining = store.add_credits(DB_PATH, user["username"], -1)
    elif user.get("is_service_key"):
        store.use_api_key_credit(DB_PATH, user["service_key"]["key_id"])
    return {
        "scan_id": scan_id,
        "report_url": f"/report/{scan_id}",
        "ai_text_percentage": stats["ai_pct"],
        "ai_status": stats["ai_status"],
        "ai_min_words": stats["ai_min_words"],
        "ai_spans": stats["ai_spans"],
        "plag_spans": stats["plag_spans"],
        "similarity_percentage": stats["sim_pct"],
        "stored_in_repository": stats["store_repo"],
        "bibliography_excluded_words": stats["bib_excluded"],
        "quotes_excluded_words": stats["quote_excluded"],
        "citations_excluded_words": stats["cited_excluded"],
        "small_matches_filtered_words": stats["small_filtered"],
        "similarity_sources": [
            {"label": s["label"], "kind": s["kind"], "percent": s["pct"],
             "matched_words": s["words"], "color": s["color"],
             "category": s.get("category", "")}
            for s in stats["sim_sources"][:10]
        ],
        "word_count": stats["words"],
        "character_count": stats["chars"],
        "sentences_total": stats["sentences_total"],
        "sentences_ai": stats["sentences_ai"],
        "credits_remaining": credits_remaining,
    }


# ---------------------------------------------------------------- report

@app.post("/api/codecheck")
async def code_check(request: Request):
    """Source-code similarity check via Dolos. 1 credit per completed
    check (admin free), mirroring /api/detect and /api/scan."""
    user, is_admin, err = _require_client_or_admin(request)
    if err:
        return err
    if not is_admin and int(user.get("credits", 0)) < 1:
        return JSONResponse(
            status_code=402,
            content={"error": "Out of credits - contact admin to top up."},
        )
    try:
        form = await request.form()
    except Exception:
        return JSONResponse(status_code=422, content={"error": "invalid upload"})
    uploads = form.getlist("files")
    language = str(form.get("language") or "auto")
    files = []
    total_raw = 0
    for up in uploads:
        name = getattr(up, "filename", "") or "file"
        try:
            raw = await up.read()
        except Exception:
            continue
        if len(raw) > codecheckmod.MAX_FILE_BYTES:
            return JSONResponse(
                status_code=422, content={"error": f"{name} exceeds 2MB"})
        total_raw += len(raw)
        if total_raw > codecheckmod.MAX_TOTAL_BYTES:
            return JSONResponse(
                status_code=422, content={"error": "files exceed 10MB total"})
        files.append((name, raw.decode("utf-8", errors="replace")))
    result = await asyncio.to_thread(codecheckmod.run_codecheck, files, language)
    if "error" in result:
        return JSONResponse(status_code=result.get("status", 500),
                            content={"error": result["error"]})
    credits_remaining = None
    if not is_admin and not user.get("is_service_key"):
        credits_remaining = store.add_credits(DB_PATH, user["username"], -1)
    elif user.get("is_service_key"):
        store.use_api_key_credit(DB_PATH, user["service_key"]["key_id"])
    result["credits_remaining"] = credits_remaining
    return result


@app.get("/report/{scan_id}")
async def get_report(scan_id: str, request: Request, download: int = 0):
    user, _is_admin = get_requester(request)
    if not user:
        return JSONResponse(
            status_code=401,
            content={"error": "authentication required - log in or send your X-API-Key header"},
        )
    scan = store.get_scan(DB_PATH, scan_id)
    if not scan:
        return JSONResponse(
            status_code=404,
            content={"error": "report not found or already auto-deleted"},
        )
    if scan["owner"] != user["username"] and not user.get("is_admin"):
        return JSONResponse(status_code=403, content={"error": "not your report"})
    path = Path(scan["report_path"])
    if not path.exists():
        return JSONResponse(
            status_code=404, content={"error": "report not found or already auto-deleted"}
        )
    headers = {}
    if download:
        import re as _re2
        from urllib.parse import quote as _quote2
        _stem = Path(scan.get("filename") or scan_id).stem or scan_id
        _safe = _re2.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", _stem).strip().strip(".") or scan_id
        _dl = f"{_safe}.html"
        headers["Content-Disposition"] = (
            f'attachment; filename="{_dl}"; filename*=UTF-8\'\'{_quote2(_dl)}')
    return FileResponse(path, media_type="text/html", headers=headers)


def _report_pdf_bytes(html_path: Path) -> bytes:
    """Render the stored report HTML to PDF via WeasyPrint (lazy import so a
    missing system lib can never break the rest of the app)."""
    from weasyprint import HTML

    return HTML(filename=str(html_path)).write_pdf()


@app.get("/report/{scan_id}/pdf/{variant}")
async def get_report_pdf_variant(scan_id: str, variant: str, request: Request):
    """Download one report section as PDF: variant = ai | similarity."""
    if variant not in ("ai", "similarity"):
        return JSONResponse(status_code=404, content={"error": "unknown variant"})
    return await _report_pdf_for(scan_id, request, variant)


async def _report_pdf_for(scan_id: str, request: Request, variant: str):
    user, _is_admin = get_requester(request)
    if not user:
        return JSONResponse(
            status_code=401,
            content={"error": "authentication required - log in or send your X-API-Key header"},
        )
    scan = store.get_scan(DB_PATH, scan_id)
    if not scan:
        return JSONResponse(
            status_code=404,
            content={"error": "report not found or already auto-deleted"},
        )
    if scan["owner"] != user["username"] and not user.get("is_admin"):
        return JSONResponse(status_code=403, content={"error": "not your report"})
    suffix = {"ai": ".ai", "similarity": ".sim"}[variant]
    label = {"ai": "AI", "similarity": "Similarity"}[variant]
    base = Path(scan["report_path"])
    path = base.parent / f"{base.stem}{suffix}.html"
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "report not found or already auto-deleted"},
        )
    try:
        pdf = await asyncio.to_thread(_report_pdf_bytes, path)
    except Exception:
        return JSONResponse(status_code=500, content={"error": "PDF generation failed"})
    # Download name follows the ORIGINAL uploaded file name, e.g.
    # "IBUS 3000 - AI Report.pdf" (Nisar's spec 2026-10-12).
    import re as _re
    from urllib.parse import quote as _quote
    stem = Path(scan.get("filename") or scan_id).stem or scan_id
    safe_stem = _re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", stem).strip().strip(".") or scan_id
    dl_name = f"{safe_stem}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition":
                f'attachment; filename="{dl_name}"; '
                f"filename*=UTF-8''{_quote(dl_name)}"
        },
    )
