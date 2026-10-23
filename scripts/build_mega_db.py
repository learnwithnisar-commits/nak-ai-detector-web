"""Build the shippable mega plagiarism fingerprint index (FAST two-phase).

Merges data/oa_corpus/fp_*.jsonl (round-1 fp_winnow, round-2 fp_mega*,
round-3 fp_r3*/fp_ccw_*/fp_pubmed2_*/fp_wiki_*) into
data/mega_index/plag_index.db — the file the app opens read-only at startup
(app/plag.py::OAIndex).

Two-phase bulk load (scales to 20M docs):
  Phase 1: parse jsonls (.jsonl and .jsonl.gz), dedupe by doc_id via
           indexed SELECT (no giant in-memory set), insert docs
           (sequential IDs) + insert (fp, doc) pairs into fps_raw heap
           table (NO index -> sequential appends, very fast).
           Checkpointed/resumable per file; accepts both legacy
           [[fp,pos],...] and compact [fp,...] fps formats.
  Phase 2: CREATE the WITHOUT ROWID fps table via
           INSERT INTO fps SELECT ... FROM fps_raw ORDER BY fp, doc
           (SQLite external sort + sequential B-tree build — far faster
           than random B-tree inserts). temp_store=FILE with
           SQLITE_TMPDIR on the big disk (never /tmp tmpfs).
  Phase 3: ANALYZE, drop fps_raw (already dropped in phase 2),
           convert to DELETE mode, swap into place.

Crash-safe (WAL + NORMAL): a SIGTERM/VM-restart never corrupts the .build
DB; re-running resumes from the checkpoint (done_files list).

Usage:
    python scripts/build_mega_db.py [out_dir]            # full build
    python scripts/build_mega_db.py --only FILE [out_dir]  # merge one
        shard/file into the build DB, then delete it (used by merge loop)
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT_ROOT / "data" / "oa_corpus"
SQLITE_TMP = CORPUS / "tmp_sqlite"
# Round-3: allow building to a different dir (e.g. data/mega_index) so the
# live seed at plag_index/ is never touched.


def discover_jsonls() -> list[Path]:
    """Round-1/2 files first (priority), then round-3 legs.

    Within the same doc_id scheme the FIRST occurrence wins, so round-2
    fp_mega_pubmed.jsonl sorts before fp_pubmed2_* shards, etc.
    """
    files: list[Path] = []
    winnow = CORPUS / "fp_winnow.jsonl"
    if winnow.exists():
        files.append(winnow)
    for pat in ("fp_mega*.jsonl", "fp_r3*.jsonl", "fp_ccw_*.jsonl",
                "fp_pubmed2_*.jsonl", "fp_wiki_*.jsonl"):
        files.extend(sorted(CORPUS.glob(pat)))
    return files


SCHEMA_PHASE1 = """
CREATE TABLE docs(
  id      INTEGER PRIMARY KEY,
  doc_id  TEXT UNIQUE NOT NULL,
  title   TEXT NOT NULL DEFAULT '',
  source  TEXT NOT NULL DEFAULT ''
);
CREATE TABLE fps_raw(
  fp  INTEGER NOT NULL,
  doc INTEGER NOT NULL
);
"""

BATCH_DOCS = 2500


def open_jsonl(p: Path):
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, encoding="utf-8")


def parse_fps(fps_in) -> list[int]:
    """Accept legacy [[fp,pos],...] and compact [fp,...] formats."""
    seen_fp: set[int] = set()
    uniq: list[int] = []
    for item in fps_in:
        h = item[0] if isinstance(item, (list, tuple)) else item
        try:
            h = int(h)
        except (TypeError, ValueError):
            continue
        if h not in seen_fp:
            seen_fp.add(h)
            uniq.append(h)
    return uniq


def main() -> None:
    args = sys.argv[1:]
    only: str | None = None
    if args and args[0] == "--only":
        only = args[1]
        args = args[2:]
    out_dir = Path(args[0]) if args else PROJECT_ROOT / "plag_index"
    OUT_DB = out_dir / "plag_index.db"
    BUILD_DB = out_dir / "plag_index.db.build"
    CHECKPOINT = out_dir / ".build_checkpoint.json"

    def _save_checkpoint(cp: dict) -> None:
        tmp = CHECKPOINT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cp), encoding="utf-8")
        tmp.replace(CHECKPOINT)

    out_dir.mkdir(parents=True, exist_ok=True)
    SQLITE_TMP.mkdir(parents=True, exist_ok=True)
    # temp files for the big phase-2 sort -> big disk, never /tmp tmpfs
    os.environ["SQLITE_TMPDIR"] = str(SQLITE_TMP)

    files = discover_jsonls()
    if only is not None:
        only_p = Path(only).resolve()
        if not only_p.exists():
            print(f"--only: {only} missing, nothing to do", flush=True)
            return
        files = [only_p]

    t0 = time.time()
    conn = sqlite3.connect(str(BUILD_DB))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")

    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    cp: dict = {}
    if CHECKPOINT.exists():
        try:
            cp = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"WARNING: checkpoint {CHECKPOINT.name} unreadable "
                  f"({e}); starting fresh", flush=True)
            cp = {}

    if cp.get("phase2_done"):
        # Phase 2 already built the final fps table (fps_raw dropped by design).
        # A relaunch must NEVER take the fresh-start branch below (it keys off
        # the missing fps_raw and would DROP docs+fps). Skip to phase 3.
        # 2026-10-14: an unguarded relaunch wiped docs this way; guard added.
        if "docs" not in tables or "fps" not in tables:
            raise SystemExit(
                "phase2_done checkpoint but docs/fps table missing "
                f"(tables={sorted(tables)}); refusing to finalize a broken "
                "build — rebuild docs or reset the checkpoint first")
        if only is not None:
            # --only merges must never trigger a finalize; phase 2 already
            # built fps, so there is nothing for a shard merge to do.
            print(f"--only {Path(only).name}: phase 2 already done, "
                  f"nothing to do", flush=True)
            conn.close()
            return
        n_docs = cp.get("n_docs", 0)
        n_fps = cp.get("n_fps", 0)
        n_dup = cp.get("n_dup", 0)
        next_id = cp.get("next_id", 1)
        done_files = set(cp.get("done_files", []))
        phase1_done = True
        ndb = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        if ndb != n_docs:
            raise SystemExit(
                f"phase2_done but docs count mismatch: db={ndb} ck={n_docs}; "
                "refusing to finalize a broken build")
        print(f"phase 1+2 already done (checkpoint): {n_docs} docs, "
              f"{n_fps:,} pairs; skipping to phase 3", flush=True)
    elif "docs" not in tables or "fps_raw" not in tables:
        for t in ("docs", "fps_raw", "fps"):
            conn.execute(f"DROP TABLE IF EXISTS {t}")
        conn.executescript(SCHEMA_PHASE1)
        conn.commit()
        n_docs = 0
        n_fps = 0
        n_dup = 0
        next_id = 1
        done_files: set[str] = set()
        phase1_done = False
    else:
        n_docs = cp.get("n_docs", 0)
        n_fps = cp.get("n_fps", 0)
        n_dup = cp.get("n_dup", 0)
        next_id = cp.get("next_id", 1)
        done_files = set(cp.get("done_files", []))
        phase1_done = cp.get("phase1_done", False)
        # sanity: if docs table is empty but checkpoint claims progress, reset
        if conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 0:
            n_docs = n_fps = n_dup = 0
            next_id = 1
            done_files = set()
            phase1_done = False
        # harden: checkpoint next_id can lag the DB after a mid-file kill;
        # the DB's MAX(id) is authoritative -- never reuse an id
        max_id = conn.execute("SELECT MAX(id) FROM docs").fetchone()[0] or 0
        if next_id <= max_id:
            print(f"  next_id {next_id} <= MAX(id) {max_id}: fast-forwarding",
                  flush=True)
            next_id = max_id + 1
        print(f"resuming: {n_docs} docs, {n_fps:,} pairs, "
              f"{len(done_files)} files done", flush=True)

    pending_files = [f for f in files if f.name not in done_files]

    # ---- PHASE 1: bulk load docs + fps_raw ----
    if not phase1_done or pending_files:
        pending: list[tuple[str, str, str, list[int]]] = []

        def flush_pending() -> None:
            nonlocal next_id, n_docs, n_fps, n_dup
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
            fp_rows: list[tuple[int, int]] = []
            for doc_id, title, source, uniq in pending:
                if doc_id in existing or doc_id in seen_now:
                    n_dup += 1
                    continue
                seen_now.add(doc_id)
                doc_rows.append((next_id, doc_id, title, source))
                fp_rows.extend((h, next_id) for h in uniq)
                next_id += 1
                n_docs += 1
                n_fps += len(uniq)
            if doc_rows:
                conn.executemany(
                    "INSERT INTO docs (id, doc_id, title, source) "
                    "VALUES (?, ?, ?, ?)", doc_rows)
            if fp_rows:
                conn.executemany(
                    "INSERT INTO fps_raw (fp, doc) VALUES (?, ?)", fp_rows)
            conn.commit()
            pending.clear()

        for jsonl in pending_files:
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
                             d.get("source", ""), uniq))
                        if len(pending) >= BATCH_DOCS:
                            flush_pending()
                        if n_docs and n_docs % 50000 == 0:
                            dt = time.time() - t0
                            print(f"  phase1: {n_docs} docs, {n_fps:,} pairs "
                                  f"({n_docs / dt:,.0f} docs/s)", flush=True)
            except Exception as e:
                print(f"  ERROR reading {jsonl.name}: {str(e)[:100]} "
                      f"(keeping for retry)", flush=True)
                continue
            flush_pending()
            done_files.add(jsonl.name)
            _save_checkpoint({"done_files": sorted(done_files),
                              "next_id": next_id, "n_docs": n_docs,
                              "n_fps": n_fps, "n_dup": n_dup,
                              "phase1_done": False})
            dt = time.time() - t0
            print(f"  merged {jsonl.name}: {n_docs} docs, {n_fps:,} pairs "
                  f"total ({dt / 60:.1f} min)", flush=True)
            # --only mode: this file is a leg shard, delete right after merge
            if only is not None and jsonl.parent == CORPUS and \
                    jsonl.name.startswith(("fp_ccw_", "fp_pubmed2_", "fp_wiki_")):
                jsonl.unlink(missing_ok=True)
                print(f"  deleted shard {jsonl.name}", flush=True)
        if only is None:
            _save_checkpoint({"done_files": sorted(done_files),
                              "next_id": next_id, "n_docs": n_docs,
                              "n_fps": n_fps, "n_dup": n_dup,
                              "phase1_done": True})
            dt = time.time() - t0
            print(f"PHASE 1 DONE: {n_docs} docs, {n_fps:,} pairs, "
                  f"{n_dup} dups in {dt / 60:.1f} min", flush=True)
        else:
            dt = time.time() - t0
            print(f"--only {Path(only).name}: build now {n_docs} docs, "
                  f"{n_fps:,} pairs ({dt / 60:.1f} min)", flush=True)
            conn.close()
            return
    else:
        print(f"phase 1 already done: {n_docs} docs, {n_fps:,} pairs",
              flush=True)
        if only is not None:
            # --only with nothing new to merge: do NOT fall through to
            # phase 2/3. Falling through would DROP a complete fps table
            # (then rebuild only the non-checkpointed buckets) and swap a
            # partial build into production. Just exit.
            print(f"--only {Path(only).name}: already merged, nothing to do",
                  flush=True)
            conn.close()
            return

    # ---- PHASE 2: sorted build of the WITHOUT ROWID fps table ----
    # Partitioned by (fp >> 58): _h64 is MD5-based (uniform 64-bit), so the
    # 64 buckets hold ~1/64th of rows each. Buckets are inserted in
    # increasing order, each with its own ORDER BY — globally sorted output
    # (sequential B-tree build) but no giant temp sort: each bucket's sort
    # is ~1/64th of the data, keeping peak disk low on the 90GB volume.
    # (fp >> 58) is order-preserving for signed 64-bit, so bucket order =
    # global fp order. Checkpointed per bucket; crash-safe.
    if not cp.get("phase2_done"):
        print("phase 2: building fps WITHOUT ROWID via partitioned "
              "sorted insert (64 buckets) ...", flush=True)
        t2 = time.time()
        buckets = list(range(-32, 32))
        done_buckets = set(cp.get("phase2_buckets", []))
        if not done_buckets:
            # Fresh phase-2 start: (re)create the fps table.
            conn.execute("DROP TABLE IF EXISTS fps")
            conn.execute("""
                CREATE TABLE fps(
                  fp  INTEGER NOT NULL,
                  doc INTEGER NOT NULL REFERENCES docs(id),
                  PRIMARY KEY (fp, doc)
                ) WITHOUT ROWID""")
            conn.commit()
        else:
            # Resume: KEEP the already-built buckets in the fps table.
            # (Before 2026-10-13 the code DROPped the table on every resume
            # while the checkpoint still claimed the old buckets done, so a
            # resumed run silently built a partial fps table. Never again:
            # resume is cumulative now.) A crash can only roll back the
            # in-flight bucket's single INSERT, so INSERT OR IGNORE below
            # makes re-running a bucket idempotent.
            print(f"  resuming phase 2: keeping {len(done_buckets)}/64 "
                  f"buckets already in fps table", flush=True)

        def _save_cp2() -> None:
            _save_checkpoint({"done_files": sorted(done_files),
                              "next_id": next_id, "n_docs": n_docs,
                              "n_fps": n_fps, "n_dup": n_dup,
                              "phase1_done": True,
                              "phase2_buckets": sorted(done_buckets)})

        for b in buckets:
            if b in done_buckets:
                continue
            tb = time.time()
            conn.execute("INSERT OR IGNORE INTO fps "
                         "SELECT fp, doc FROM fps_raw "
                         "WHERE (fp >> 58) = ? ORDER BY fp, doc", (b,))
            conn.commit()
            done_buckets.add(b)
            _save_cp2()
            print(f"  bucket {b:+d}: {len(done_buckets)}/64 "
                  f"({time.time() - tb:.0f}s)", flush=True)
        print(f"  partitioned sorted insert done in "
              f"{(time.time() - t2) / 60:.1f} min", flush=True)
        conn.execute("DROP TABLE fps_raw")
        conn.commit()
        _save_checkpoint({"done_files": sorted(done_files),
                          "next_id": next_id, "n_docs": n_docs,
                          "n_fps": n_fps, "n_dup": n_dup,
                          "phase1_done": True, "phase2_done": True})
    else:
        print("phase 2 already done (checkpoint)", flush=True)

    # ---- PHASE 3: finalize ----
    print("ANALYZE ...", flush=True)
    conn.execute("ANALYZE")
    conn.commit()
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()

    os.replace(BUILD_DB, OUT_DB)
    for suf in ("-wal", "-shm"):
        p = Path(str(OUT_DB) + suf)
        if p.exists():
            p.unlink()
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
    dt = time.time() - t0
    size_mb = OUT_DB.stat().st_size / 1e6
    print(f"BUILT {OUT_DB.name}: {n_docs} docs, {n_fps:,} fps, "
          f"{n_dup} dup doc_ids skipped, "
          f"{size_mb:.0f} MB in {dt / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
