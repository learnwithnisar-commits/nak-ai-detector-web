"""Round-3: Wikipedia articles via CirrusSearch dump chunks (fingerprint-only).

Uses the sharded CirrusSearch content dump (1GB bz2 chunks, plain text per
article) — far more efficient than the MediaWiki API (which throttles
whole-article extracts to ~1 page/request).

Source: https://dumps.wikimedia.org/other/cirrus_search_index/20261011/
        index_name=enwiki_content/  (66 x ~633MB bz2 chunks)

Each chunk is downloaded, stream-parsed, fingerprinted (k=5/w=4), then
deleted. Filters: namespace==0, not a #REDIRECT. NO full text stored.

Dedupe: doc_id (wikipedia:{page_id}) + title-hash via r3_seen.db (round-1/2)
+ own in-memory set. Checkpoint: data/oa_corpus/r3_wikipedia_progress.json
(done_chunks). Resumable per chunk.

Usage: python scripts/r3_wikipedia.py [target] [n_chunks]  (default 700000, 18)
"""

from __future__ import annotations

import bz2
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_common import (  # noqa: E402
    CORPUS, title_hash, Seen, fingerprint_text)

OUT_JSONL = CORPUS / "fp_r3_wikipedia.jsonl"
PROGRESS = CORPUS / "r3_wikipedia_progress.json"
TMP = CORPUS / "tmp_r3_wiki"  # big disk, not /tmp (tmpfs is only 512MB)

DUMP = "20261011"
BASE = (f"https://dumps.wikimedia.org/other/cirrus_search_index/{DUMP}/"
        "index_name=enwiki_content/")
UA = "NAK-AI-Detector/1.0 (research fingerprint index; contact: scout@example.com)"
MIN_TOKENS = 50


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def main() -> None:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 700_000
    n_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 18
    TMP.mkdir(parents=True, exist_ok=True)
    prog: dict = {}
    if PROGRESS.exists():
        try:
            prog = json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            prog = {}
    done_chunks: list[int] = prog.get("done_chunks", [])
    n_new = prog.get("n_new", 0)
    n_skip = prog.get("n_skip", 0)
    n_short = prog.get("n_short", 0)
    n_redir = prog.get("n_redir", 0)

    seen = Seen()
    own_ids: set[str] = set()
    own_th: set[str] = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                own_ids.add(d.get("doc_id", ""))
                t = d.get("title") or ""
                if t:
                    own_th.add(title_hash(t))
    print(f"r3_wikipedia: resume new={n_new}, own={len(own_ids)}", flush=True)

    out = open(OUT_JSONL, "a", encoding="utf-8")
    try:
        for ci in range(n_chunks):
            if n_new >= target:
                break
            if ci in done_chunks:
                continue
            fname = f"enwiki_content-{DUMP}-{ci:05d}.json.bz2"
            dest = TMP / fname
            print(f"  downloading chunk {ci} ...", flush=True)
            t0 = time.time()
            try:
                download(BASE + fname, dest)
            except Exception as e:
                print(f"    download failed: {str(e)[:80]}", flush=True)
                continue
            print(f"    {(dest.stat().st_size / 1e6):.0f} MB in "
                  f"{time.time() - t0:.0f}s; parsing ...", flush=True)
            c_new = 0
            try:
                with bz2.open(dest, "rt", encoding="utf-8",
                              errors="ignore") as bz:
                    for line in bz:
                        if n_new >= target:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        if "page_id" not in d:
                            continue  # bulk header line
                        if d.get("namespace") != 0:
                            continue
                        st = (d.get("source_text") or "").lstrip()
                        if st.lower().startswith("#redirect"):
                            n_redir += 1
                            continue
                        page_id = d.get("page_id")
                        title = (d.get("title") or "").strip()
                        if not page_id or not title:
                            continue
                        doc_id = f"wikipedia:{page_id}"
                        if doc_id in own_ids or seen.has_doc(doc_id):
                            n_skip += 1
                            continue
                        th = title_hash(title)
                        if th in own_th or seen.has_th(th):
                            n_skip += 1
                            continue
                        fps, n_tok = fingerprint_text(
                            d.get("text") or "", min_tokens=MIN_TOKENS)
                        if not fps:
                            n_short += 1
                            continue
                        out.write(json.dumps({
                            "doc_id": doc_id,
                            "title": title[:300],
                            "source": "wikipedia",
                            "n_words": n_tok,
                            "fps": fps,
                        }) + "\n")
                        own_ids.add(doc_id)
                        own_th.add(th)
                        n_new += 1
                        c_new += 1
            except Exception as e:
                print(f"    parse error: {str(e)[:80]}", flush=True)
            dest.unlink(missing_ok=True)
            done_chunks.append(ci)
            prog.update({"done_chunks": done_chunks, "n_new": n_new,
                         "n_skip": n_skip, "n_short": n_short,
                         "n_redir": n_redir})
            PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
            print(f"    chunk {ci}: +{c_new} articles (total {n_new})",
                  flush=True)
    finally:
        out.close()
        seen.close()
    prog.update({"done_chunks": done_chunks, "n_new": n_new,
                 "n_skip": n_skip, "n_short": n_short, "n_redir": n_redir,
                 "done": n_new >= target})
    PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
    print(f"R3_WIKIPEDIA DONE: new={n_new} skip={n_skip} short={n_short} "
          f"redir={n_redir}", flush=True)


if __name__ == "__main__":
    main()
