"""Round-3 big: Common Crawl web text at 10M+ scale (parallel workers).

Worker w of W processes WET paths paths[w::W] from
data/oa_corpus/r3_cc_wet.paths (CC-MAIN-2026-39, 100k paths), skipping the
few files the retired single-threaded leg already did. Each .warc.wet.gz is
stream-parsed (WARC conversion records), pages fingerprinted
(source='web', doc_id='web:'+sha1(url)[:24], compact int-list fps) and
appended to rotating shards data/oa_corpus/fp_ccw_{w}_p{n}.jsonl
(rotated every SHARD_DOCS). The active shard name is published in
data/oa_corpus/r3_ccw_{w}.current so the merge loop never touches it.
Each WET file is deleted right after parsing. Fingerprint-only.

Dedupe notes: NO r3_seen.db lookups ('web:' doc_ids never collide with
round-1/2 schemes) and NO in-memory own_ids (slices are disjoint; the
build's doc_id dedupe catches stragglers). This keeps RAM flat at 10M scale.

Usage: python scripts/r3_cc_big.py <worker_id> <n_workers> <files_per_worker> [target_pages]
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_common import CORPUS, fingerprint_text  # noqa: E402
from r3_commoncrawl import iter_wet_records  # noqa: E402

PATHS_FILE = CORPUS / "r3_cc_wet.paths"
OLD_PROGRESS = CORPUS / "r3_commoncrawl_progress.json"
TMP = CORPUS / "tmp_r3_web"
BASE = "https://data.commoncrawl.org/"
UA = "NAK-AI-Detector/1.0 (research fingerprint index; contact: scout@example.com)"
MIN_TOKENS = 50
MAX_TOKENS = 200
SHARD_DOCS = 50_000
# bound per-page work: fingerprinting only ever sees the head of a page
HEAD_CHARS = 60_000


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def main() -> None:
    worker = int(sys.argv[1])
    n_workers = int(sys.argv[2])
    files_per_worker = int(sys.argv[3])
    target_pages = int(sys.argv[4]) if len(sys.argv) > 4 else 5_200_000
    TMP.mkdir(parents=True, exist_ok=True)

    PROGRESS = CORPUS / f"r3_ccw_{worker}_progress.json"
    CURRENT = CORPUS / f"r3_ccw_{worker}.current"

    prog: dict = {}
    if PROGRESS.exists():
        try:
            prog = json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            prog = {}
    done_files: list[str] = prog.get("done_files", [])
    done_set = set(done_files)
    n_new = prog.get("n_new", 0)
    n_skip = prog.get("n_skip", 0)
    n_boiler = prog.get("n_boiler", 0)
    n_short = prog.get("n_short", 0)
    shard_no = prog.get("shard_no", 0)
    shard_docs = prog.get("shard_docs", 0)

    # paths already processed by the retired single-threaded leg
    old_done: set[str] = set()
    if OLD_PROGRESS.exists():
        try:
            old_done = set(json.loads(
                OLD_PROGRESS.read_text(encoding="utf-8")).get("done_files", []))
        except Exception:
            pass

    with open(PATHS_FILE, encoding="utf-8") as f:
        all_paths = [l.strip() for l in f if l.strip()]
    my_paths = [p for p in all_paths[worker::n_workers]
                if p not in old_done][:files_per_worker]
    print(f"r3_ccw{worker}: resume new={n_new}, slice={len(my_paths)} files "
          f"(of {len(all_paths)} total)", flush=True)

    def open_shard():
        p = CORPUS / f"fp_ccw_{worker}_p{shard_no}.jsonl"
        CURRENT.write_text(p.name, encoding="utf-8")
        return open(p, "a", encoding="utf-8")

    out = open_shard()
    try:
        for wpath in my_paths:
            if n_new >= target_pages:
                break
            if wpath in done_set:
                continue
            fname = wpath.rsplit("/", 1)[-1]
            dest = TMP / f"ccw{worker}_{fname}"
            try:
                download(BASE + wpath, dest)
            except Exception as e:
                print(f"  w{worker} download failed {fname}: {str(e)[:60]}",
                      flush=True)
                continue
            f_new = 0
            try:
                for url, text in iter_wet_records(dest):
                    if n_new >= target_pages:
                        break
                    if len(text) > HEAD_CHARS:
                        text = text[:HEAD_CHARS]
                    fps, n_tok = fingerprint_text(
                        text, min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS)
                    if not fps:
                        n_short += 1
                        continue
                    toks = text.split()
                    if len(set(toks[:2000])) / max(1, len(toks[:2000])) < 0.08:
                        n_boiler += 1
                        continue
                    doc_id = "web:" + hashlib.sha1(
                        url.encode()).hexdigest()[:24]
                    out.write(json.dumps({
                        "doc_id": doc_id,
                        "title": url[:300],
                        "source": "web",
                        "n_words": n_tok,
                        "fps": [int(h) for h, _pos in fps],
                    }) + "\n")
                    n_new += 1
                    f_new += 1
                    shard_docs += 1
                    if shard_docs >= SHARD_DOCS:
                        out.close()
                        shard_no += 1
                        shard_docs = 0
                        out = open_shard()
                        print(f"  w{worker} rotated to shard {shard_no} "
                              f"(total {n_new})", flush=True)
            except Exception as e:
                print(f"  w{worker} parse error {fname}: {str(e)[:60]}",
                      flush=True)
            dest.unlink(missing_ok=True)
            done_files.append(wpath)
            done_set.add(wpath)
            prog.update({"done_files": done_files, "n_new": n_new,
                         "n_skip": n_skip, "n_boiler": n_boiler,
                         "n_short": n_short, "shard_no": shard_no,
                         "shard_docs": shard_docs})
            PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
            print(f"  w{worker} {fname}: +{f_new} (total {n_new})", flush=True)
    finally:
        out.close()
    prog.update({"done_files": done_files, "n_new": n_new, "n_skip": n_skip,
                 "n_boiler": n_boiler, "n_short": n_short,
                 "shard_no": shard_no, "shard_docs": shard_docs,
                 "done": n_new >= target_pages or len(done_files) >= len(my_paths)})
    PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
    print(f"R3_CCW{worker} DONE: new={n_new} boiler={n_boiler} short={n_short}",
          flush=True)


if __name__ == "__main__":
    main()
