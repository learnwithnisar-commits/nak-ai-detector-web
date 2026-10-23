"""One-off RECOVERY (2026-10-14): rebuild the docs table of the round-3
mega-index build DB after it was wiped by an unguarded relaunch
(build_mega_db.py line-150 fresh-start branch; guard since added).

The fps table (796M rows, phase-2 output) survived intact. This script
recreates docs with IDENTICAL id assignment by replicating phase-1's
exact logic: discover_jsonls() order, per-line skip rules, and
first-occurrence-wins dedup on doc_id. It does NOT touch fps.

Usage: cd ~/workspace/ai-detector && .venv/bin/python scripts/salvage_docs_r3.py
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_mega_db import (  # noqa: E402
    CORPUS, discover_jsonls, open_jsonl, parse_fps,
)

BUILD_DB = PROJECT_ROOT / "data" / "mega_index" / "plag_index.db.build"
CHECKPOINT = PROJECT_ROOT / "data" / "mega_index" / ".build_checkpoint.json"
BATCH_DOCS = 2500

DOCS_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs(
  id      INTEGER PRIMARY KEY,
  doc_id  TEXT UNIQUE NOT NULL,
  title   TEXT NOT NULL DEFAULT '',
  source  TEXT NOT NULL DEFAULT ''
);
"""


def main() -> None:
    cp = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    done_files = set(cp["done_files"])
    expect_docs = cp["n_docs"]
    files = [p for p in discover_jsonls() if p.name in done_files]
    if len(files) != len(done_files):
        missing = done_files - {p.name for p in files}
        raise SystemExit(f"file set mismatch: {len(files)} vs {len(done_files)}; "
                         f"e.g. missing {sorted(missing)[:5]}")
    print(f"salvage: rebuilding docs from {len(files)} files "
          f"(expect {expect_docs} docs)", flush=True)

    conn = sqlite3.connect(str(BUILD_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-262144")  # 256MB
    have = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "docs" in have:
        n0 = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        raise SystemExit(f"docs table already exists with {n0} rows; refusing")
    if "fps" not in have:
        raise SystemExit("fps table missing; salvage not applicable")
    conn.executescript(DOCS_SCHEMA)
    conn.commit()

    t0 = time.time()
    next_id = 1
    n_docs = 0
    pending: list[tuple[str, str, str]] = []

    def flush_pending() -> None:
        nonlocal next_id, n_docs
        if not pending:
            return
        ids = [p[0] for p in pending]
        existing: set[str] = set()
        for i in range(0, len(ids), 512):
            ch = ids[i:i + 512]
            existing.update(
                r[0] for r in conn.execute(
                    "SELECT doc_id FROM docs WHERE doc_id IN (%s)"
                    % ",".join("?" * len(ch)), ch))
        seen_now: set[str] = set()
        doc_rows: list[tuple[int, str, str, str]] = []
        for doc_id, title, source in pending:
            if doc_id in existing or doc_id in seen_now:
                continue
            seen_now.add(doc_id)
            doc_rows.append((next_id, doc_id, title, source))
            next_id += 1
            n_docs += 1
        conn.executemany(
            "INSERT INTO docs (id, doc_id, title, source) VALUES (?, ?, ?, ?)",
            doc_rows)
        conn.commit()
        pending.clear()

    for fi, jsonl in enumerate(files, 1):
        try:
            with open_jsonl(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    doc_id = d.get("doc_id", "")
                    if not doc_id:
                        continue
                    uniq = parse_fps(d.get("fps", ()))
                    if not uniq:
                        continue
                    pending.append(
                        (doc_id, (d.get("title") or doc_id)[:300],
                         d.get("source", "")))
                    if len(pending) >= BATCH_DOCS:
                        flush_pending()
        except Exception as e:
            raise SystemExit(f"ERROR reading {jsonl.name}: {e}")
        flush_pending()
        if fi % 10 == 0 or fi == len(files):
            dt = time.time() - t0
            print(f"  {fi}/{len(files)} files, {n_docs} docs "
                  f"({dt / 60:.1f} min)", flush=True)

    ndb = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    dt = time.time() - t0
    print(f"SALVAGED docs: {ndb} rows (expect {expect_docs}) "
          f"in {dt / 60:.1f} min", flush=True)
    conn.close()
    if ndb != expect_docs:
        raise SystemExit(f"COUNT MISMATCH: got {ndb}, expect {expect_docs}")


if __name__ == "__main__":
    main()
