"""Round-3 post-pass: cross-leg title-hash dedupe before the merge build.

The 5 r3 legs run in parallel, so fetch-time dedupe (vs r3_seen.db) can't
see other legs' output. This pass drops r3 docs whose title-hash already
exists in the round-1/2 index OR in a higher-priority r3 file.

Priority (first wins): theses > crossref > pubmed > wikipedia > web.
Theses win because they're the student-paper proxy.

Uses a SQLite working copy of r3_seen.db (memory-light). Rewrites each
fp_r3_*.jsonl in place (via .tmp + replace). Idempotent-ish: re-running
is safe (already-dropped docs stay dropped).

Usage: python scripts/r3_postpass.py
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_common import CORPUS, title_hash  # noqa: E402

ORDER = [
    "fp_r3_theses.jsonl",
    "fp_r3_crossref.jsonl",
    "fp_r3_pubmed.jsonl",
    "fp_r3_wikipedia.jsonl",
    "fp_r3_web.jsonl",
]
WORK_DB = CORPUS / "r3_postpass.db"


def main() -> None:
    t0 = time.time()
    # working copy of the round-1/2 seen snapshot
    if WORK_DB.exists():
        WORK_DB.unlink()
    shutil.copy(CORPUS / "r3_seen.db", WORK_DB)
    conn = sqlite3.connect(str(WORK_DB))
    # speed pragmas for bulk inserts
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")

    total_kept = total_dropped = 0
    for fname in ORDER:
        path = CORPUS / fname
        if not path.exists():
            print(f"  {fname}: missing, skipping", flush=True)
            continue
        tmp = path.with_suffix(".jsonl.tmp")
        kept = dropped = 0
        batch: list[tuple[str, str]] = []
        with open(path, encoding="utf-8") as fin, \
                open(tmp, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                doc_id = d.get("doc_id", "")
                th = title_hash(d.get("title") or "")
                dup = False
                if doc_id:
                    dup = conn.execute(
                        "SELECT 1 FROM seen WHERE doc_id=?",
                        (doc_id,)).fetchone() is not None
                if not dup:
                    dup = conn.execute(
                        "SELECT 1 FROM seen WHERE th=?",
                        (th,)).fetchone() is not None
                if dup:
                    dropped += 1
                    continue
                fout.write(line + "\n")
                batch.append((doc_id, th))
                kept += 1
                if len(batch) >= 5000:
                    conn.executemany(
                        "INSERT OR IGNORE INTO seen VALUES (?,?)", batch)
                    batch.clear()
            conn.executemany("INSERT OR IGNORE INTO seen VALUES (?,?)", batch)
            conn.commit()
        tmp.replace(path)
        total_kept += kept
        total_dropped += dropped
        print(f"  {fname}: kept={kept} dropped={dropped}", flush=True)
    conn.close()
    print(f"POSTPASS DONE: kept={total_kept} dropped={total_dropped} "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
