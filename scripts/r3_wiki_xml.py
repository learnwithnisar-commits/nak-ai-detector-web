"""Round-3 big: ALL ~6M English Wikipedia articles via the XML dump.

Two resumable phases in one script:
  1. download enwiki-latest-pages-articles.xml.bz2 (~20GB) with curl -C -
     (resume loop) into data/oa_corpus/tmp_r3_wiki/; marker .dl_done.
  2. stream-parse with bz2 + ET.iterparse (namespace 0, non-redirect pages),
     light regex wikitext strip, winnowing fingerprints, compact shards
     data/oa_corpus/fp_wiki_p{n}.jsonl (rotated every SHARD_DOCS).
     Active shard published in data/oa_corpus/r3_wiki.current.
     The .bz2 is deleted after parsing. Fingerprint-only.

doc_id = wikipedia:{page_id} (same scheme as the retired cirrus leg, whose
output was empty, so no overlap). No seen-db checks ('wikipedia:' is a new
scheme); the build dedupes by doc_id.

Usage: python scripts/r3_wiki_xml.py
"""

from __future__ import annotations

import bz2
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_common import CORPUS, fingerprint_text  # noqa: E402

DUMP_URL = ("https://dumps.wikimedia.org/enwiki/latest/"
            "enwiki-latest-pages-articles.xml.bz2")
TMP = CORPUS / "tmp_r3_wiki"
DL_FILE = TMP / "enwiki-latest-pages-articles.xml.bz2"
DL_DONE = TMP / ".dl_done"
PROGRESS = CORPUS / "r3_wiki_progress.json"
CURRENT = CORPUS / "r3_wiki.current"
MIN_TOKENS = 50
MAX_TOKENS = 200
SHARD_DOCS = 250_000
HEAD_CHARS = 60_000

# ---- light wikitext strip (fast compiled regexes; fingerprints are statistical)
_RX_COMMENT = re.compile(r"<!--.*?-->", re.S)
_RX_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.S | re.I)
_RX_TPL = re.compile(r"\{\{[^{}]*\}\}")
_RX_TABLE = re.compile(r"\{\|.*?\|\}", re.S)
_RX_NSLINK = re.compile(r"\[\[(File|Image|Category|Wikipedia|Portal|Draft|Template|Help|MediaWiki):[^\]]*\]\]", re.I)
_RX_WLINK = re.compile(r"\[\[([^|\]]*\|)?([^\]]+)\]\]")
_RX_ELINK = re.compile(r"\[https?://[^\s\]]+\s+([^\]]+)\]")
_RX_URL = re.compile(r"https?://\S+")
_RX_HTML = re.compile(r"<[^>]+>")
_RX_QUOTES = re.compile(r"'{2,5}")
_RX_HEADING = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.M)
_RX_WS = re.compile(r"\s+")


def strip_wikitext(t: str) -> str:
    t = _RX_COMMENT.sub(" ", t)
    t = _RX_REF.sub(" ", t)
    for _ in range(3):  # nested templates, depth ~3
        t2 = _RX_TPL.sub(" ", t)
        if t2 == t:
            break
        t = t2
    t = _RX_TABLE.sub(" ", t)
    t = _RX_NSLINK.sub(" ", t)
    t = _RX_WLINK.sub(r"\2", t)
    t = _RX_ELINK.sub(r"\1", t)
    t = _RX_URL.sub(" ", t)
    t = _RX_HTML.sub(" ", t)
    t = _RX_QUOTES.sub("", t)
    t = _RX_HEADING.sub(r"\1", t)
    return _RX_WS.sub(" ", t).strip()


def download() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    if DL_DONE.exists() and DL_FILE.exists():
        print(f"  dump already downloaded "
              f"({DL_FILE.stat().st_size / 1e9:.1f} GB)", flush=True)
        return
    print(f"  downloading {DUMP_URL} ...", flush=True)
    while True:
        r = subprocess.run(
            ["curl", "-sS", "-C", "-", "--retry", "10", "--retry-delay", "10",
             "--max-time", "86400", "-o", str(DL_FILE), DUMP_URL],
            capture_output=True, text=True, timeout=87000)
        if r.returncode == 0 and DL_FILE.exists() and DL_FILE.stat().st_size > 0:
            break
        print(f"  curl rc={r.returncode}, retrying in 60s: "
              f"{r.stderr.strip()[-120:]}", flush=True)
        time.sleep(60)
    DL_DONE.write_text("ok", encoding="utf-8")
    print(f"  download complete ({DL_FILE.stat().st_size / 1e9:.1f} GB)",
          flush=True)


def parse() -> None:
    prog: dict = {}
    if PROGRESS.exists():
        try:
            prog = json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            prog = {}
    n_new = prog.get("n_new", 0)
    n_skip_ns = prog.get("n_skip_ns", 0)
    n_redir = prog.get("n_redir", 0)
    n_short = prog.get("n_short", 0)
    pages_seen = prog.get("pages_seen", 0)
    start_seen = pages_seen  # pages processed in previous runs (for skip)
    shard_no = prog.get("shard_no", 0)
    shard_docs = prog.get("shard_docs", 0)
    print(f"r3_wiki: resume new={n_new}, pages_seen={pages_seen}", flush=True)

    def open_shard():
        p = CORPUS / f"fp_wiki_p{shard_no}.jsonl"
        CURRENT.write_text(p.name, encoding="utf-8")
        return open(p, "a", encoding="utf-8")

    out = open_shard()
    t0 = time.time()
    try:
        with bz2.open(DL_FILE, "rb") as f:
            ctx = ET.iterparse(f, events=("start", "end"))
            _ev, root = next(ctx)  # <mediawiki>
            title = ns = page_id = text = None
            is_redir = False
            for event, elem in ctx:
                tag = elem.tag
                if event == "start" and tag == "page":
                    title = ns = page_id = text = None
                    is_redir = False
                elif event == "end":
                    if tag == "title":
                        title = elem.text or ""
                    elif tag == "ns":
                        ns = (elem.text or "").strip()
                    elif tag == "id" and page_id is None:
                        # first <id> under <page> is the page id
                        # (revision ids come later, inside <revision>)
                        page_id = (elem.text or "").strip()
                    elif tag == "redirect":
                        is_redir = True
                    elif tag == "text":
                        text = elem.text or ""
                    elif tag == "page":
                        pages_seen += 1
                        # restart-skip: the dump stream is deterministic, so
                        # pages already processed in a previous run are
                        # identical; skip re-fingerprinting them (build
                        # dedupes by doc_id anyway, this just saves CPU)
                        if pages_seen - start_seen <= start_seen and start_seen:
                            root.clear()
                            continue
                        if ns == "0" and not is_redir and page_id and text:
                            if len(text) > HEAD_CHARS:
                                text = text[:HEAD_CHARS]
                            clean = strip_wikitext(text)
                            fps, n_tok = fingerprint_text(
                                clean, min_tokens=MIN_TOKENS,
                                max_tokens=MAX_TOKENS)
                            if fps:
                                doc_id = f"wikipedia:{page_id}"
                                out.write(json.dumps({
                                    "doc_id": doc_id,
                                    "title": (title or "")[:300],
                                    "source": "wikipedia",
                                    "n_words": n_tok,
                                    "fps": [int(h) for h, _pos in fps],
                                }) + "\n")
                                n_new += 1
                                shard_docs += 1
                                if shard_docs >= SHARD_DOCS:
                                    out.close()
                                    shard_no += 1
                                    shard_docs = 0
                                    out = open_shard()
                                    print(f"  rotated to shard {shard_no} "
                                          f"(total {n_new})", flush=True)
                            else:
                                n_short += 1
                        else:
                            if ns != "0":
                                n_skip_ns += 1
                            elif is_redir:
                                n_redir += 1
                            else:
                                n_short += 1
                        root.clear()  # keep memory flat
                        if pages_seen % 200000 == 0:
                            dt = time.time() - t0
                            prog.update({"n_new": n_new, "n_skip_ns": n_skip_ns,
                                         "n_redir": n_redir, "n_short": n_short,
                                         "pages_seen": pages_seen,
                                         "shard_no": shard_no,
                                         "shard_docs": shard_docs})
                            PROGRESS.write_text(json.dumps(prog),
                                                encoding="utf-8")
                            print(f"  pages={pages_seen} new={n_new} "
                                  f"({pages_seen / dt:,.0f} pages/s)",
                                  flush=True)
    finally:
        out.close()
    prog.update({"n_new": n_new, "n_skip_ns": n_skip_ns, "n_redir": n_redir,
                 "n_short": n_short, "pages_seen": pages_seen,
                 "shard_no": shard_no, "shard_docs": shard_docs,
                 "done": True})
    PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
    DL_FILE.unlink(missing_ok=True)
    DL_DONE.unlink(missing_ok=True)
    print(f"R3_WIKI DONE: new={n_new} pages_seen={pages_seen} "
          f"redir={n_redir} short={n_short}", flush=True)


def main() -> None:
    download()
    parse()


if __name__ == "__main__":
    main()
