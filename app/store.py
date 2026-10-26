"""SQLite account/session/scan store for NAK AI Detector.

Single uvicorn worker, tiny footprint. Passwords hashed with
hashlib.pbkdf2_hmac (stdlib). Sessions are server-side random tokens
stored here and handed to the browser as an HttpOnly cookie.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  username   TEXT PRIMARY KEY,
  pw_salt    TEXT NOT NULL DEFAULT '',
  pw_hash    TEXT NOT NULL DEFAULT '',
  credits    INTEGER NOT NULL DEFAULT 0,
  suspended  INTEGER NOT NULL DEFAULT 0,
  api_key    TEXT UNIQUE,
  is_admin   INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
  token      TEXT PRIMARY KEY,
  username   TEXT NOT NULL,
  created_at REAL NOT NULL,
  expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS scans (
  scan_id       TEXT PRIMARY KEY,
  owner         TEXT NOT NULL,
  filename      TEXT NOT NULL,
  created_at    REAL NOT NULL,
  deletes_at    REAL NOT NULL,
  report_path   TEXT NOT NULL,
  ai_pct        REAL NOT NULL,
  words         INTEGER NOT NULL,
  submission_id TEXT NOT NULL DEFAULT '',
  file_size     INTEGER NOT NULL DEFAULT 0,
  pages         INTEGER NOT NULL DEFAULT 0
);
-- Plagiarism: irreversible winnowing fingerprints of previously scanned
-- documents, for cross-document comparison. Only hashes are kept (the
-- documents themselves auto-delete after 30 min and can never be
-- reconstructed from these). Pruned to the most recent PLAG_DOC_KEEP docs.
CREATE TABLE IF NOT EXISTS plag_docs (
  doc_ref   TEXT PRIMARY KEY,
  filename  TEXT NOT NULL DEFAULT '',
  words     INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS plag_doc_fps (
  fp      INTEGER NOT NULL,
  doc_ref TEXT NOT NULL,
  pos     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_plag_doc_fps_fp ON plag_doc_fps(fp);
-- Service-level API keys (admin-managed; never shown to clients).
CREATE TABLE IF NOT EXISTS api_keys (
  key_id      TEXT PRIMARY KEY,
  label       TEXT NOT NULL DEFAULT '',
  key_hash    TEXT NOT NULL UNIQUE,
  owner       TEXT NOT NULL DEFAULT 'admin',
  created_at  REAL NOT NULL,
  last_used_at REAL,
  daily_quota INTEGER NOT NULL DEFAULT 0,
  credits     INTEGER NOT NULL DEFAULT 0,
  revoked     INTEGER NOT NULL DEFAULT 0
);
-- Per-key per-day usage counters for daily quotas.
CREATE TABLE IF NOT EXISTS api_key_usage (
  key_id TEXT NOT NULL,
  day    TEXT NOT NULL,
  used   INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (key_id, day)
);
-- Simple key/value settings (e.g. ai_min_words).
CREATE TABLE IF NOT EXISTS settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL DEFAULT ''
);
"""

SESSION_TTL = 30 * 24 * 3600  # 30 days
_PBKDF2_ITERS = 200_000


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        # migrate older DBs that predate the new scans columns
        for col, ddl in (
            ("submission_id", "TEXT NOT NULL DEFAULT ''"),
            ("file_size", "INTEGER NOT NULL DEFAULT 0"),
            ("pages", "INTEGER NOT NULL DEFAULT 0"),
        ):
            try:
                conn.execute(f"ALTER TABLE scans ADD COLUMN {col} {ddl}")
            except sqlite3.OperationalError:
                pass  # column already exists
        # migrate older DBs: owner on plag_docs (for per-user repo cleanup)
        try:
            conn.execute("ALTER TABLE plag_docs ADD COLUMN owner TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists
        # migrate older DBs: sim_pct on scans (for /api/history)
        try:
            conn.execute("ALTER TABLE scans ADD COLUMN sim_pct REAL NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.execute(
            "INSERT OR IGNORE INTO users (username, credits, is_admin, created_at)"
            " VALUES ('admin', 0, 1, ?)",
            (time.time(),),
        )


def hash_password(password: str) -> tuple[str, str]:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)
    return salt.hex(), dk.hex()


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERS)
        return secrets.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


def create_user(db_path: Path, username: str, password: str, credits: int = 0) -> dict:
    salt, pw_hash = hash_password(password)
    api_key = secrets.token_urlsafe(32)
    with _connect(db_path) as conn:
        try:
            conn.execute(
                "INSERT INTO users (username, pw_salt, pw_hash, credits, api_key, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (username, salt, pw_hash, credits, api_key, time.time()),
            )
        except sqlite3.IntegrityError:
            raise ValueError("username already exists")
    return get_user(db_path, username)


def get_user(db_path: Path, username: str) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return dict(row) if row else None


def get_user_by_api_key(db_path: Path, api_key: str) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE api_key = ?", (api_key,)).fetchone()
    return dict(row) if row else None


def list_users(db_path: Path) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT username, credits, suspended, is_admin, created_at FROM users ORDER BY created_at"
        ).fetchall()
    return [dict(r) for r in rows]


def add_credits(db_path: Path, username: str, delta: int) -> int:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT credits FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            raise ValueError("unknown user")
        new_balance = max(0, int(row["credits"]) + delta)
        conn.execute("UPDATE users SET credits = ? WHERE username = ?", (new_balance, username))
    return new_balance


def set_suspended(db_path: Path, username: str, suspended: bool) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE users SET suspended = ? WHERE username = ?", (1 if suspended else 0, username)
        )


def new_session(db_path: Path, username: str) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (token, username, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, username, now, now + SESSION_TTL),
        )
    return token


def get_session_user(db_path: Path, token: str) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM sessions WHERE token = ?", (token,)).fetchone()
        if not row:
            return None
        if float(row["expires_at"]) < time.time():
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return None
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (row["username"],)
        ).fetchone()
    return dict(user) if user else None


def delete_session(db_path: Path, token: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def prune_sessions(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))


def record_scan(
    db_path: Path,
    scan_id: str,
    owner: str,
    filename: str,
    report_path: str,
    ai_pct: float,
    words: int,
    ttl_seconds: int,
    submission_id: str = "",
    file_size: int = 0,
    pages: int = 0,
    sim_pct: float = 0.0,
) -> float:
    now = time.time()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO scans (scan_id, owner, filename, created_at, deletes_at,"
            " report_path, ai_pct, words, submission_id, file_size, pages, sim_pct)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (scan_id, owner, filename, now, now + ttl_seconds, report_path,
             ai_pct, words, submission_id, file_size, pages, sim_pct),
        )
    return now + ttl_seconds


def get_scan(db_path: Path, scan_id: str) -> dict | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM scans WHERE scan_id = ?", (scan_id,)).fetchone()
    return dict(row) if row else None


def delete_scan(db_path: Path, scan_id: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM scans WHERE scan_id = ?", (scan_id,))


def expired_scans(db_path: Path, now: float) -> list[dict]:
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT scan_id, report_path FROM scans WHERE deletes_at < ?", (now,)).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- plagiarism fingerprints

PLAG_DOC_KEEP = 5000  # retain fingerprints of the most recent N scanned docs


def record_doc_fps(db_path: Path, doc_ref: str, filename: str, words: int,
                   fps: list[tuple[int, int]], owner: str = "") -> None:
    """Store a scanned document's winnowing fingerprints for future
    cross-document comparison, then prune to the most recent PLAG_DOC_KEEP."""
    now = time.time()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO plag_docs (doc_ref, filename, words, created_at, owner)"
            " VALUES (?, ?, ?, ?, ?)",
            (doc_ref, filename, words, now, owner),
        )
        conn.executemany(
            "INSERT INTO plag_doc_fps (fp, doc_ref, pos) VALUES (?, ?, ?)",
            [(int(h), doc_ref, int(p)) for h, p in fps],
        )
        conn.execute(
            """DELETE FROM plag_doc_fps WHERE doc_ref IN (
                 SELECT doc_ref FROM plag_docs ORDER BY created_at DESC
                 LIMIT -1 OFFSET ?)""",
            (PLAG_DOC_KEEP,),
        )
        conn.execute(
            """DELETE FROM plag_docs WHERE doc_ref NOT IN (
                 SELECT doc_ref FROM plag_docs ORDER BY created_at DESC
                 LIMIT ?)""",
            (PLAG_DOC_KEEP,),
        )


def delete_doc_fps_by_ref(db_path: Path, doc_ref: str) -> tuple[int, int]:
    """Delete one document's fingerprints from the repository.
    Returns (docs_deleted, fps_deleted)."""
    with _connect(db_path) as conn:
        fps = conn.execute(
            "DELETE FROM plag_doc_fps WHERE doc_ref = ?", (doc_ref,)).rowcount
        docs = conn.execute(
            "DELETE FROM plag_docs WHERE doc_ref = ?", (doc_ref,)).rowcount
    return docs, fps


def delete_doc_fps_by_filename(db_path: Path, filename_substr: str) -> tuple[int, int]:
    """Delete repository fingerprints for docs whose filename contains the
    given substring (case-insensitive). Returns (docs_deleted, fps_deleted)."""
    with _connect(db_path) as conn:
        refs = [r[0] for r in conn.execute(
            "SELECT doc_ref FROM plag_docs WHERE filename LIKE ?",
            (f"%{filename_substr}%",))]
        fps = docs = 0
        for ref in refs:
            fps += conn.execute(
                "DELETE FROM plag_doc_fps WHERE doc_ref = ?", (ref,)).rowcount
            docs += conn.execute(
                "DELETE FROM plag_docs WHERE doc_ref = ?", (ref,)).rowcount
    return docs, fps


def list_repo_docs(db_path: Path, limit: int = 200) -> list[dict]:
    """Repository documents, newest first (for admin management)."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT doc_ref, filename, words, created_at FROM plag_docs "
            "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def query_doc_fps(db_path: Path, fps: list[int]) -> dict[int, list[tuple[str, str]]]:
    """Batched lookup: {fp -> [(doc_ref, filename)]} for doc-vs-doc matching."""
    out: dict[int, list[tuple[str, str]]] = {}
    if not fps:
        return out
    uniq = list(dict.fromkeys(int(h) for h in fps))
    with _connect(db_path) as conn:
        for i in range(0, len(uniq), 900):
            batch = uniq[i:i + 900]
            q = ("SELECT f.fp, f.doc_ref, d.filename FROM plag_doc_fps f "
                 "JOIN plag_docs d ON d.doc_ref = f.doc_ref "
                 f"WHERE f.fp IN ({','.join('?' * len(batch))})")
            for fp, doc_ref, filename in conn.execute(q, batch):
                out.setdefault(int(fp), []).append((str(doc_ref), str(filename or "")))
    return out


# ---------------------------------------------------------------- settings

def get_setting(db_path: Path, key: str, default: str = "") -> str:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else default


def set_setting(db_path: Path, key: str, value: str) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


# ------------------------------------------------------- service API keys

def _hash_api_key(raw: str) -> str:
    return hashlib.sha256(("nak-api:" + raw).encode()).hexdigest()


def create_service_api_key(db_path: Path, label: str, daily_quota: int = 0,
                           owner: str = "admin", credits: int = 0) -> tuple[str, str]:
    """Create a service-level API key. Returns (key_id, raw_key); the raw key
    is shown once and never stored (only its hash). daily_quota=0 = unlimited.
    credits: total scans allowed; -1 = unlimited, 0 = exhausted/empty,
    >0 = finite balance. 1 credit is consumed per completed scan."""
    key_id = "key_" + secrets.token_urlsafe(8)
    raw = "nak_" + secrets.token_urlsafe(32)
    _ensure_api_key_credits_col(db_path)
    credits_val = -1 if int(credits) < 0 else int(credits)
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO api_keys (key_id, label, key_hash, owner, created_at,"
            " last_used_at, daily_quota, credits, revoked)"
            " VALUES (?, ?, ?, ?, ?, NULL, ?, ?, 0)",
            (key_id, label, _hash_api_key(raw), owner, time.time(),
             int(daily_quota or 0), credits_val),
        )
    return key_id, raw


def _ensure_api_key_credits_col(db_path: Path) -> None:
    """Add credits column to api_keys if missing (migration).

    One-time backfill: keys created before the credits feature existed
    (2026-10-14) carried credits=0 under the old "0 = unlimited" convention.
    They are converted to -1 (the new explicit unlimited marker) so they
    keep working; keys created after always store an explicit value.
    """
    with _connect(db_path) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(api_keys)").fetchall()]
        if "credits" not in cols:
            conn.execute("ALTER TABLE api_keys ADD COLUMN credits INTEGER NOT NULL DEFAULT 0")
            conn.execute("UPDATE api_keys SET credits=-1 WHERE credits=0")
        conn.execute("CREATE TABLE IF NOT EXISTS schema_flags (name TEXT PRIMARY KEY)")
        done = conn.execute(
            "SELECT 1 FROM schema_flags WHERE name='credits_neg1_backfill'").fetchone()
        if not done:
            # 1760486400 = 2026-10-14 00:00:00 UTC, before the credits feature.
            conn.execute(
                "UPDATE api_keys SET credits=-1 WHERE credits=0 AND created_at < 1760486400")
            conn.execute(
                "INSERT OR IGNORE INTO schema_flags (name) VALUES ('credits_neg1_backfill')")


def add_api_key_credits(db_path: Path, key_id: str, delta: int) -> int:
    """Add (or remove) credits from an API key. Returns new balance.
    Unlimited keys (credits=-1) stay unlimited. Floor is 0 (exhausted)."""
    _ensure_api_key_credits_col(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT credits FROM api_keys WHERE key_id=?", (key_id,)).fetchone()
        if not row:
            return -1
        cur = int(row["credits"])
        if cur < 0:
            return -1  # unlimited stays unlimited
        new_bal = max(0, cur + int(delta))
        conn.execute("UPDATE api_keys SET credits=? WHERE key_id=?", (new_bal, key_id))
        return new_bal


def api_key_has_credit(db_path: Path, key_id: str) -> bool:
    """Peek: may this key start a scan? True when the key exists, is not
    revoked, and has credits left (credits=-1 means unlimited). Consumes
    nothing; call use_api_key_credit() after a completed scan."""
    _ensure_api_key_credits_col(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT credits, revoked FROM api_keys WHERE key_id=?",
                           (key_id,)).fetchone()
        if not row or row["revoked"]:
            return False
        return int(row["credits"]) != 0


def use_api_key_credit(db_path: Path, key_id: str) -> bool:
    """Atomically consume 1 credit from an API key. Returns True if consumed
    (or key is unlimited). credits=-1 → unlimited, always True, no decrement.
    credits=0 → exhausted, returns False. credits>0 → decremented by 1.
    The UPDATE ... WHERE credits>0 guard plus rowcount makes concurrent
    scans race-safe: the balance can never go below zero."""
    _ensure_api_key_credits_col(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT credits, revoked FROM api_keys WHERE key_id=?",
                           (key_id,)).fetchone()
        if not row or row["revoked"]:
            return False
        credits = int(row["credits"])
        if credits < 0:
            conn.execute("UPDATE api_keys SET last_used_at=? WHERE key_id=?",
                         (time.time(), key_id))
            return True
        if credits == 0:
            return False
        cur = conn.execute(
            "UPDATE api_keys SET credits=credits-1, last_used_at=? "
            "WHERE key_id=? AND credits>0",
            (time.time(), key_id))
        return cur.rowcount == 1


def list_service_api_keys(db_path: Path) -> list[dict]:
    _ensure_api_key_credits_col(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT key_id, label, owner, created_at, last_used_at,"
            " daily_quota, credits, revoked FROM api_keys ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def revoke_service_api_key(db_path: Path, key_id: str) -> bool:
    with _connect(db_path) as conn:
        cur = conn.execute("UPDATE api_keys SET revoked = 1 WHERE key_id = ?",
                           (key_id,))
    return cur.rowcount > 0


def delete_service_api_key(db_path: Path, key_id: str) -> bool:
    """Permanently delete an API key (only if already revoked)."""
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM api_keys WHERE key_id = ? AND revoked = 1",
                           (key_id,))
    return cur.rowcount > 0


def get_service_api_key(db_path: Path, raw: str) -> dict | None:
    """Look up a service key by its raw value. Returns None if unknown/revoked."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ? AND revoked = 0",
            (_hash_api_key(raw),)).fetchone()
    return dict(row) if row else None


def touch_service_api_key(db_path: Path, key_id: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("UPDATE api_keys SET last_used_at = ? WHERE key_id = ?",
                     (time.time(), key_id))


def check_and_inc_quota(db_path: Path, key_id: str, daily_quota: int) -> bool:
    """Increment today's usage for key_id. Returns False when the daily quota
    (0 = unlimited) is already exhausted; True when the call is allowed."""
    if int(daily_quota or 0) <= 0:
        return True
    import datetime as _dt
    day = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT used FROM api_key_usage WHERE key_id = ? AND day = ?",
            (key_id, day)).fetchone()
        used = int(row["used"]) if row else 0
        if used >= int(daily_quota):
            return False
        conn.execute(
            "INSERT INTO api_key_usage (key_id, day, used) VALUES (?, ?, 1)"
            " ON CONFLICT(key_id, day) DO UPDATE SET used = used + 1",
            (key_id, day),
        )
    return True


# ------------------------------------------------------------- user admin

def delete_user(db_path: Path, username: str) -> dict:
    """Permanently delete a client account and ALL of its data:
    user row, sessions, scan records, and any repository fingerprints from
    that user's scans (repo doc_refs are scan_ids; an owner column is also
    kept on plag_docs going forward). Returns counts for the audit message."""
    counts = {"scans": 0, "repo_docs": 0, "repo_fps": 0}
    with _connect(db_path) as conn:
        # scan_ids owned by this user (repo doc_refs are scan_ids)
        rows = conn.execute("SELECT scan_id FROM scans WHERE owner = ?",
                            (username,)).fetchall()
        scan_ids = [r["scan_id"] for r in rows]
        counts["scans"] = len(scan_ids)
        doc_rows = []
        if scan_ids:
            q = f"({','.join('?' * len(scan_ids))})"
            doc_rows = conn.execute(
                f"SELECT doc_ref FROM plag_docs WHERE doc_ref IN {q}"
                f" OR owner = ?", (*scan_ids, username)).fetchall()
        else:
            # no live scans left (auto-deleted) -- still drop repo docs by owner
            doc_rows = conn.execute(
                "SELECT doc_ref FROM plag_docs WHERE owner = ?",
                (username,)).fetchall()
        seen = set()
        for (doc_ref,) in doc_rows:
            if doc_ref in seen:
                continue
            seen.add(doc_ref)
            counts["repo_docs"] += 1
            cur = conn.execute("DELETE FROM plag_doc_fps WHERE doc_ref = ?",
                               (doc_ref,))
            counts["repo_fps"] += cur.rowcount
            conn.execute("DELETE FROM plag_docs WHERE doc_ref = ?", (doc_ref,))
        conn.execute("DELETE FROM scans WHERE owner = ?", (username,))
        conn.execute("DELETE FROM sessions WHERE username = ?", (username,))
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
    return counts
def list_user_scans(db_path: Path, username: str, limit: int = 50) -> list[dict]:
    """Recent scans for one user, newest first (within the auto-delete window)."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT scan_id, filename, ai_pct, sim_pct, created_at FROM scans"
            " WHERE owner = ? ORDER BY created_at DESC LIMIT ?",
            (username, limit)).fetchall()
    return [dict(r) for r in rows]


def list_all_scans(db_path: Path, limit: int = 100) -> list[dict]:
    """All recent scans, newest first (admin view only; read-only SELECT)."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT scan_id, owner, filename, ai_pct, sim_pct, created_at"
            " FROM scans ORDER BY created_at DESC LIMIT ?",
            (limit,)).fetchall()
    return [dict(r) for r in rows]
