"""Validate docs<->fps alignment after salvage: for sampled docs, check that
every (fp, my_id) pair exists in fps (PK lookup). A shifted id mapping
would fail (64-bit fps collisions negligible)."""
import json
import random
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_mega_db import discover_jsonls, open_jsonl, parse_fps  # noqa: E402

random.seed(42)
cp = json.loads(open(PROJECT_ROOT / "data/mega_index/.build_checkpoint.json").read())
done = set(cp["done_files"])
files = [p for p in discover_jsonls() if p.name in done]

# sample 25 docs from each of 8 files spread across the order
idxs = [0, 20, 40, 60, 80, 100, 120, len(files) - 1]
samples = []
for fi in idxs:
    fp = files[fi]
    ids = []
    with open_jsonl(fp) as f:
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
            ids.append((doc_id, uniq))
    step = max(1, len(ids) // 25)
    for i in range(0, len(ids), step)[:25]:
        samples.append((fp.name, ids[i][0], ids[i][1]))
print(f"sampled {len(samples)} docs from {len(idxs)} files", flush=True)

conn = sqlite3.connect(
    f"file:{PROJECT_ROOT}/data/mega_index/plag_index.db.build?mode=ro", uri=True)
ok = bad = missing = 0
for fname, doc_id, uniq in samples:
    r = conn.execute("SELECT id FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
    if not r:
        missing += 1
        print(f"  MISSING doc_id {doc_id} ({fname})", flush=True)
        continue
    did = r[0]
    # check a subset of fps via PK lookup (fast)
    check = uniq[:50] if len(uniq) > 50 else uniq
    good = True
    for h in check:
        if not conn.execute(
                "SELECT 1 FROM fps WHERE fp=? AND doc=?", (h, did)).fetchone():
            good = False
            break
    if good:
        ok += 1
    else:
        bad += 1
        if bad <= 3:
            print(f"  MISALIGNED {doc_id} -> id {did} ({fname})", flush=True)
print(f"RESULT ok={ok} bad={bad} missing={missing}", flush=True)
conn.close()
