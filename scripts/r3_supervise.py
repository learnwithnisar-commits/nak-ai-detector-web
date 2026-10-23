"""Supervisor for the round-3 20M index legs (run every ~30 min via cron).

For each leg: if its progress file reports done=true, leave it alone;
elif no matching process is alive, relaunch it with its original args
(all legs are resumable via progress files / build checkpoints).
The merge loop is restarted unless /tmp/r3_MERGE_FINALIZE exists.

Logs to /tmp/r3_supervise.log. Exits 0 when everything is done.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT_ROOT / "data" / "oa_corpus"
MEGA = PROJECT_ROOT / "data" / "mega_index"
LOG = Path("/tmp/r3_supervise.log")
FINALIZE = Path("/tmp/r3_MERGE_FINALIZE")
CAP_REACHED = Path("/tmp/r3_CAP_REACHED")
# Persistent cap marker (survives /tmp tmpfs wipes on reboot). Touched when
# the cap is enforced; the supervisor treats it as authoritative so a wiped
# /tmp can never resurrect fetch legs / merge_loop mid-finalize.
# 2026-10-14: added after a wiped /tmp caused a merge_loop resurrection.
CAP_PERSISTENT = MEGA / ".cap_reached"
PY = PROJECT_ROOT / ".venv" / "bin" / "python"

# Nisar's order 2026-10-12 ~12:11 PKT: ONE round only, stop at 50 lakh (5M)
# total docs, then finalize + halt. Do NOT go to 20M.
# UPDATE 2026-10-12 ~14:04 PKT: Nisar said "or documents add kr" — cap raised
# to 1 crore (10M), the upper end of his 50L-1cr band. Still one round, then halt.
DOC_CAP_TOTAL = 20_000_000
ROUND2_DOCS = 905_223  # re-included in final build from source jsonls

# (name, argv, progress_file, match_pattern)
LEGS = [
    ("ccw0", ["r3_cc_big.py", "0", "2", "255", "5200000"],
     "r3_ccw_0_progress.json", "r3_cc_big.py 0 "),
    ("ccw1", ["r3_cc_big.py", "1", "2", "255", "5200000"],
     "r3_ccw_1_progress.json", "r3_cc_big.py 1 "),
    ("wiki", ["r3_wiki_xml.py"], "r3_wiki_progress.json", "r3_wiki_xml.py"),
    ("pm20", ["r3_pubmed2.py", "0", "61", "70"],
     "r3_pubmed2_0_progress.json", "r3_pubmed2.py 0 "),
    ("pm21", ["r3_pubmed2.py", "1", "261", "70"],
     "r3_pubmed2_1_progress.json", "r3_pubmed2.py 1 "),
    ("crossref", ["r3_crossref.py", "300000"],
     "r3_crossref_progress.json", "r3_crossref.py"),
    ("theses", ["r3_theses.py", "277000"],
     "r3_theses_progress.json", "r3_theses.py"),
    ("pubmed", ["r3_pubmed.py", "44", "17"],
     "r3_pubmed_progress.json", "r3_pubmed.py 44"),
]


def log(msg: str) -> None:
    line = f"{time.strftime('%m-%d %H:%M')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_done(name: str, prog_file: str) -> bool:
    p = CORPUS / prog_file
    if p.exists():
        try:
            if bool(json.loads(p.read_text(encoding="utf-8")).get("done")):
                return True
        except Exception:
            pass
    # Self-healing: legs print "R3_* DONE:" to /tmp/r3_<name>.log but some
    # never write done=true to their progress file. Treat the log marker
    # as done and persist it so we stop resurrecting finished workers.
    logf = Path(f"/tmp/r3_{name}.log")
    try:
        if logf.exists():
            tail = logf.read_text(encoding="utf-8")[-2000:]
            if " DONE:" in tail:
                try:
                    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
                except Exception:
                    data = {}
                data["done"] = True
                p.write_text(json.dumps(data), encoding="utf-8")
                log(f"{name}: DONE marker found in log — marked done")
                return True
    except Exception:
        pass
    return False


def is_alive(pattern: str) -> bool:
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True)
    return r.returncode == 0


def count_total_docs() -> int:
    """Approximate total docs that the final build will contain."""
    total = ROUND2_DOCS
    # docs already merged into the build DB
    try:
        r = subprocess.run(
            ["timeout", "60", str(PY), "-c",
             "import sqlite3;print(sqlite3.connect('file:data/mega_index/"
             "plag_index.db.build?mode=ro',uri=True).execute("
             "'SELECT COUNT(*) FROM docs').fetchone()[0])"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=70)
        if r.returncode == 0:
            total += int(r.stdout.strip())
    except Exception:
        pass
    # unmerged fingerprint shards (all prefixes the legs write)
    # NOTE: wc over ~10M+ lines takes ~2min and grows; keep the timeout
    # generous and NEVER silently undercount — a swallowed timeout here
    # once let the fleet blow past the 10M cap (2026-10-12). On failure we
    # log loudly and fall back to the last good count.
    try:
        r = subprocess.run(
            "wc -l data/oa_corpus/fp_r3_*.jsonl data/oa_corpus/fp_ccw_*.jsonl "
            "data/oa_corpus/fp_pubmed2_*.jsonl data/oa_corpus/fp_wiki_*.jsonl "
            "2>/dev/null | tail -1",
            shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT,
            timeout=600)
        if r.returncode == 0 and r.stdout.strip():
            total += int(r.stdout.strip().split()[0])
            try:
                Path("/tmp/r3_doc_count.json").write_text(json.dumps(
                    {"total": total, "ts": time.time()}))
            except Exception:
                pass
        else:
            raise RuntimeError(f"wc failed rc={r.returncode}")
    except Exception as e:
        log(f"ERROR count_total_docs: shard wc failed ({e}); "
            f"falling back to last-good count")
        try:
            cached = json.loads(Path("/tmp/r3_doc_count.json").read_text())
            return int(cached["total"])
        except Exception:
            pass
    return total


FETCH_PATTERNS = [
    "r3_cc_big.py", "r3_pubmed2.py", "r3_pubmed.py", "r3_crossref.py",
    "r3_theses.py", "r3_wiki_xml.py", "r3_wikipedia.py", "r3_commoncrawl.py",
    "enwiki-latest", "warc.wet.gz",
]


def enforce_cap() -> None:
    """Nisar: one round, stop at 10M. Kill fetchers, mark legs done,
    signal merge finalize. A separate finalize job does the full build."""
    log(f"CAP REACHED ({DOC_CAP_TOTAL:,} docs) — halting all fetch legs")
    for pat in FETCH_PATTERNS:
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    time.sleep(3)
    for _, _, prog_file, _ in LEGS:
        p = CORPUS / prog_file
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            data["done"] = True
            data["cap_stopped"] = True
            p.write_text(json.dumps(data), encoding="utf-8")
        except Exception as e:
            log(f"warn: could not mark {prog_file} done: {e}")
    # also mark the retired legs' progress files so nothing relaunches
    for extra in ["r3_wikipedia_progress.json", "r3_commoncrawl_progress.json",
                  "r3_wiki_progress.json"]:
        p = CORPUS / extra
        try:
            data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            data["done"] = True
            data["cap_stopped"] = True
            p.write_text(json.dumps(data), encoding="utf-8")
        except Exception:
            pass
    FINALIZE.touch()
    CAP_REACHED.touch()
    log("cap enforced: fetchers killed, legs marked done, MERGE_FINALIZE set")


def launch(name: str, argv: list[str]) -> None:
    log(f"restarting {name}: {' '.join(argv)}")
    subprocess.Popen(
        [str(PY), f"scripts/{argv[0]}", *argv[1:]],
        cwd=PROJECT_ROOT,
        stdout=open(f"/tmp/r3_{name}.log", "a"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True)


def main() -> None:
    if CAP_PERSISTENT.exists():
        # Cap enforced in a previous boot; /tmp markers may have been wiped.
        # Re-touch them so merge_loop and the finalize job stay halted.
        CAP_REACHED.touch(exist_ok=True)
        FINALIZE.touch(exist_ok=True)
        log("cap already reached (persistent marker) — staying halted")
        return
    if CAP_REACHED.exists():
        log("cap already reached — staying halted")
        return
    total = count_total_docs()
    log(f"total docs (r2 + build + unmerged): {total:,}")
    if total >= DOC_CAP_TOTAL:
        enforce_cap()
        return
    restarted = []
    for name, argv, prog_file, pattern in LEGS:
        if is_done(name, prog_file):
            continue
        if is_alive(pattern):
            continue
        launch(name, argv)
        restarted.append(name)
    # merge loop
    if not FINALIZE.exists() and not is_alive("merge_loop.py"):
        launch("merge", ["merge_loop.py"])
        restarted.append("merge")
    if restarted:
        log(f"restarted: {', '.join(restarted)}")
    else:
        log("all legs alive or done")


if __name__ == "__main__":
    main()
