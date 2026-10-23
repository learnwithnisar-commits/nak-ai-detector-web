"""Validate plag_web.db: schema, counts, self-match test via app.plag OAIndex."""
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location("bwi", str(PROJECT_ROOT / "scripts" / "build_web_index.py"))
bwi = importlib.util.module_from_spec(_spec)
sys.modules["bwi"] = bwi
_spec.loader.exec_module(bwi)

DB = PROJECT_ROOT / "data" / "web_academic_index" / "plag_web.db"

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
print("quick_check:", con.execute("PRAGMA quick_check").fetchone()[0])
ndocs = con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
nfps = con.execute("SELECT COUNT(*) FROM fps").fetchone()[0]
print(f"docs={ndocs} fps_pairs={nfps}")
print("domains:", con.execute("SELECT source, COUNT(*) FROM docs GROUP BY source").fetchall())
print("size MB:", DB.stat().st_size / 1024 / 1024)

# self-match: take an indexed doc's fingerprints, look them up, verify it matches itself
import json
from plag import OAIndex, StreamWinnower, split_keep
row = con.execute("SELECT id, doc_id, title FROM docs LIMIT 1 OFFSET 100").fetchone()
doc_rowid, doc_url, title = row
print(f"\nself-match test on doc {doc_rowid}: {title[:60]}")
# rebuild its fingerprints from stored fps (we stored per-doc deduped fps)
fps = [r[0] for r in con.execute("SELECT fp FROM fps WHERE doc=?", (doc_rowid,))]
print(f"  stored fps: {len(fps)}")
idx = OAIndex(DB)
hits = idx.lookup(fps[:5000])
# the doc itself should be the top hit
from collections import Counter
c = Counter()
for fp, docs in hits.items():
    for d in docs:
        c[d] += 1
top = c.most_common(3)
print("  top matched docs:", [(d, n, idx.meta(d)[1]) for d, n in top])
assert top[0][0] == doc_rowid, "SELF-MATCH FAILED"
print("  SELF-MATCH OK: doc matches itself with", top[0][1], "of", len(fps[:5000]), "fps")
con.close()
print("\nVALIDATION PASSED")
