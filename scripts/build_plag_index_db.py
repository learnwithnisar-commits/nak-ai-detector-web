"""Build the shippable SQLite plagiarism fingerprint index.

Reads data/oa_corpus/fp_winnow.jsonl (built by scripts/build_plag_index.py)
and writes plag_index/plag_index.db — a compact (fp -> doc) lookup the app
opens read-only at startup.

Why SQLite and not an in-memory dict: 20k corpus docs already produce
~1.7M fingerprints; a Python dict costs ~400MB RSS, far too much for the
free tier. SQLite answers batched IN-queries in milliseconds for typical
documents and scales to 200k+ docs.

The plag_index/ directory (not data/) is used because .railwayignore
excludes data/ from deploys — this DB must ship to Railway.

Rebuild whenever the corpus grows:
    python scripts/build_plag_index.py        # fingerprints
    python scripts/build_plag_index_db.py     # this DB
then redeploy.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JSONL = PROJECT_ROOT / "data" / "oa_corpus" / "fp_winnow.jsonl"
OUT_DIR = PROJECT_ROOT / "plag_index"
OUT_DB = OUT_DIR / "plag_index.db"

SCHEMA = """
CREATE TABLE docs(
  id      INTEGER PRIMARY KEY,
  doc_id  TEXT UNIQUE NOT NULL,
  title   TEXT NOT NULL DEFAULT '',
  source  TEXT NOT NULL DEFAULT ''
);
CREATE TABLE fps(
  fp  INTEGER NOT NULL,
  doc INTEGER NOT NULL REFERENCES docs(id),
  PRIMARY KEY (fp, doc)
);
"""


def _signed(h: int) -> int:
    """Map uint64 hash into SQLite's signed-64 range (must match app/plag.py)."""
    h &= (1 << 64) - 1
    return h - (1 << 64) if h >= (1 << 63) else h


def main() -> None:
    if not JSONL.exists():
        sys.exit(f"missing {JSONL} - run scripts/build_plag_index.py first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_DB.exists():
        OUT_DB.unlink()
    conn = sqlite3.connect(str(OUT_DB))
    conn.executescript(SCHEMA)
    n_docs = 0
    n_fps = 0
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cur = conn.execute(
                "INSERT OR IGNORE INTO docs (doc_id, title, source) VALUES (?, ?, ?)",
                (d["doc_id"], d.get("title", "") or d["doc_id"], d.get("source", "")),
            )
            if cur.lastrowid:
                doc_rowid = cur.lastrowid
            else:
                doc_rowid = conn.execute(
                    "SELECT id FROM docs WHERE doc_id = ?", (d["doc_id"],)
                ).fetchone()[0]
            seen: set[int] = set()
            rows = []
            for h, _pos in d.get("fps", []):
                h = _signed(int(h))
                if h not in seen:
                    seen.add(h)
                    rows.append((h, doc_rowid))
            conn.executemany("INSERT OR IGNORE INTO fps (fp, doc) VALUES (?, ?)", rows)
            n_docs += 1
            n_fps += len(rows)
            if n_docs % 5000 == 0:
                conn.commit()
                print(f"  ...{n_docs} docs")
    conn.commit()
    conn.execute("ANALYZE")
    conn.commit()
    conn.close()
    size_mb = OUT_DB.stat().st_size / 1024 / 1024
    print(f"built {OUT_DB}: {n_docs} docs, {n_fps} (fp,doc) pairs, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
