"""Round-3 20M merge loop: incrementally merge finished leg shards into the
growing mega build DB (data/mega_index/plag_index.db.build) via
build_mega_db.py --only, then the shard is deleted by the build.

Runs until /tmp/r3_MERGE_FINALIZE appears (the coordinator then runs the
final full build itself). Also a disk watchdog: if free space < 15GB it
pauses the slowest (API-based) legs first, never the bulk legs.

A shard is merged only when its leg is no longer writing it:
  - the leg's .current marker exists and names a different shard, OR
  - the leg's progress file reports done=true.

Usage: python scripts/merge_loop.py   (runs in background, logs to
/tmp/r3_merge.log)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT_ROOT / "data" / "oa_corpus"
MEGA = PROJECT_ROOT / "data" / "mega_index"
LOG = Path("/tmp/r3_merge.log")
FINALIZE = Path("/tmp/r3_MERGE_FINALIZE")
SLEEP = 600
DISK_MIN_GB = 15


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def leg_state(prefix: str, key: str) -> tuple[str | None, bool]:
    """Return (active_shard_name_or_None, done_bool) for a leg."""
    marker = CORPUS / f"{prefix}{key}.current"
    prog = CORPUS / f"{prefix}{key}_progress.json"
    active = marker.read_text(encoding="utf-8").strip() if marker.exists() else None
    done = False
    if prog.exists():
        try:
            done = bool(json.loads(prog.read_text(encoding="utf-8")).get("done"))
        except Exception:
            pass
    return active, done


def shard_leg(shard: str) -> tuple[str, str] | None:
    # fp_ccw_{w}_p{n}.jsonl / fp_pubmed2_{tag}_p{n}.jsonl / fp_wiki_p{n}.jsonl
    parts = shard.split("_")
    if shard.startswith("fp_ccw_"):
        return ("r3_ccw_", parts[2])
    if shard.startswith("fp_pubmed2_"):
        return ("r3_pubmed2_", parts[2])
    if shard.startswith("fp_wiki_"):
        return ("r3_wiki", "")
    return None


def mergeable(shard: str) -> bool:
    """A shard is mergeable only when its leg has demonstrably moved on to a
    newer shard (its .current marker names a different file). The final
    shard of a finished leg is left for the final full build — never merge
    a file a worker might still be appending to."""
    leg = shard_leg(shard)
    if leg is None:
        return False
    prefix, key = leg
    active, _done = leg_state(prefix, key)
    return active is not None and active != shard


def disk_ok() -> bool:
    free_gb = shutil.disk_usage(CORPUS).free / 1e9
    if free_gb < DISK_MIN_GB:
        log(f"LOW DISK {free_gb:.1f}G < {DISK_MIN_GB}G: pausing API legs")
        for pat in ("scripts/r3_crossref.py", "scripts/r3_theses.py"):
            subprocess.run(["pkill", "-f", pat], capture_output=True)
        return False
    return True


def main() -> None:
    log("merge loop started")
    while True:
        if FINALIZE.exists():
            log("finalize trigger seen, merge loop exiting")
            return
        if not disk_ok():
            time.sleep(SLEEP)
            continue
        shards = sorted(p.name for p in CORPUS.glob("fp_ccw_*.jsonl"))
        shards += sorted(p.name for p in CORPUS.glob("fp_pubmed2_*.jsonl"))
        shards += sorted(p.name for p in CORPUS.glob("fp_wiki_*.jsonl"))
        merged = 0
        for shard in shards:
            if FINALIZE.exists():
                log("finalize trigger seen, merge loop exiting")
                return
            if not mergeable(shard):
                continue
            log(f"merging {shard} ...")
            r = subprocess.run(
                [sys.executable, "scripts/build_mega_db.py", "--only",
                 str(CORPUS / shard), str(MEGA)],
                cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=7200)
            tail = (r.stdout or "").strip().splitlines()[-2:]
            log(f"  rc={r.returncode} {' | '.join(tail)}")
            if r.returncode != 0:
                log(f"  stderr: {(r.stderr or '')[-300:]}")
            merged += 1
        # status line
        bdb = MEGA / "plag_index.db.build"
        if bdb.exists():
            log(f"  build DB: {bdb.stat().st_size / 1e9:.2f} GB "
                f"(+{merged} shards this round)")
        else:
            log(f"  no build DB yet (+{merged} shards this round)")
        time.sleep(SLEEP)


if __name__ == "__main__":
    main()
