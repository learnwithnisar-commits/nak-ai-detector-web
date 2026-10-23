"""Round-3: Common Crawl web text (NEW source, 5th leg).

Samples WET files (extracted web text) from a recent crawl
(CC-MAIN-2026-39), stream-parses WARC conversion records, and appends
winnowing fingerprints (k=5/w=4) to data/oa_corpus/fp_r3_web.jsonl.
Fingerprint-only, NO full-text storage. Public crawl data.

Dedupe: doc_id = web:{sha1(url)} via r3_seen.db (round-1/2) + own
in-memory set. Light boilerplate filter (very low unique-token ratio
pages are skipped). Checkpoint: data/oa_corpus/r3_commoncrawl_progress.json
(done wet files). Each .warc.wet.gz is deleted after parsing.

Usage: python scripts/r3_commoncrawl.py [target_pages] [n_files]
       (default 1500000 pages, 80 files)
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_common import (  # noqa: E402
    CORPUS, title_hash, Seen, fingerprint_text)

OUT_JSONL = CORPUS / "fp_r3_web.jsonl"
PROGRESS = CORPUS / "r3_commoncrawl_progress.json"
PATHS_FILE = CORPUS / "r3_cc_wet.paths"
TMP = CORPUS / "tmp_r3_web"  # big disk, not /tmp (tmpfs is only 512MB)

CRAWL = "CC-MAIN-2026-39"
PATHS_URL = f"https://data.commoncrawl.org/crawl-data/{CRAWL}/wet.paths.gz"
BASE = "https://data.commoncrawl.org/"
UA = "NAK-AI-Detector/1.0 (research fingerprint index; contact: scout@example.com)"
MIN_TOKENS = 50
MAX_TOKENS = 4000


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def get_wet_paths(n_files: int) -> list[str]:
    if not PATHS_FILE.exists():
        print("  downloading wet.paths.gz ...", flush=True)
        tmp = PATHS_FILE.with_suffix(".paths.gz.tmp")
        download(PATHS_URL, tmp)
        with gzip.open(tmp, "rt", encoding="utf-8") as gz, \
                open(PATHS_FILE, "w", encoding="utf-8") as out:
            for line in gz:
                line = line.strip()
                if line:
                    out.write(line + "\n")
        tmp.unlink(missing_ok=True)
    with open(PATHS_FILE, encoding="utf-8") as f:
        all_paths = [l.strip() for l in f if l.strip()]
    print(f"  {len(all_paths)} wet paths; sampling {n_files}", flush=True)
    stride = max(1, len(all_paths) // n_files)
    return [all_paths[i] for i in range(0, len(all_paths), stride)][:n_files]


def iter_wet_records(path: Path):
    """Yield (url, text) for WARC conversion records in a .warc.wet.gz."""
    with gzip.open(path, "rb") as gz:
        while True:
            line = gz.readline()
            if not line:
                break
            if not line.startswith(b"WARC/1.0"):
                continue
            headers: dict[bytes, bytes] = {}
            while True:
                hline = gz.readline()
                if not hline or hline in (b"\r\n", b"\n"):
                    break
                if b":" in hline:
                    k, v = hline.split(b":", 1)
                    headers[k.strip()] = v.strip()
            try:
                length = int(headers.get(b"Content-Length", b"0"))
            except ValueError:
                length = 0
            is_conv = headers.get(b"WARC-Type") == b"conversion"
            url = headers.get(b"WARC-Target-URI", b"").decode(
                "utf-8", "ignore")
            if is_conv and length > 0 and url:
                block = gz.read(length)
                text = block.decode("utf-8", "ignore")
                yield url, text
            elif length > 0:
                remaining = length
                while remaining > 0:
                    chunk = gz.read(min(remaining, 1 << 16))
                    if not chunk:
                        break
                    remaining -= len(chunk)


def main() -> None:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 1_500_000
    n_files = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    TMP.mkdir(parents=True, exist_ok=True)
    prog: dict = {}
    if PROGRESS.exists():
        try:
            prog = json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            prog = {}
    done_files: list[str] = prog.get("done_files", [])
    n_new = prog.get("n_new", 0)
    n_skip = prog.get("n_skip", 0)
    n_boiler = prog.get("n_boiler", 0)
    n_short = prog.get("n_short", 0)

    seen = Seen()
    own_ids: set[str] = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                try:
                    own_ids.add(json.loads(line).get("doc_id", ""))
                except Exception:
                    continue
    print(f"r3_web: resume new={n_new}, own={len(own_ids)}", flush=True)

    wet_paths = get_wet_paths(n_files)
    out = open(OUT_JSONL, "a", encoding="utf-8")
    try:
        for wpath in wet_paths:
            if n_new >= target:
                break
            if wpath in done_files:
                continue
            fname = wpath.rsplit("/", 1)[-1]
            dest = TMP / fname
            print(f"  downloading {fname} ...", flush=True)
            t0 = time.time()
            try:
                download(BASE + wpath, dest)
            except Exception as e:
                print(f"    download failed: {str(e)[:80]}", flush=True)
                continue
            print(f"    {(dest.stat().st_size / 1e6):.0f} MB in "
                  f"{time.time() - t0:.0f}s; parsing ...", flush=True)
            f_new = 0
            try:
                for url, text in iter_wet_records(dest):
                    if n_new >= target:
                        break
                    doc_id = "web:" + hashlib.sha1(
                        url.encode()).hexdigest()[:24]
                    if doc_id in own_ids or seen.has_doc(doc_id):
                        n_skip += 1
                        continue
                    fps, n_tok = fingerprint_text(
                        text, min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS)
                    if not fps:
                        n_short += 1
                        continue
                    # light boilerplate filter: near-duplicate token soup
                    toks = text.split()
                    if len(set(toks[:2000])) / max(1, len(toks[:2000])) < 0.08:
                        n_boiler += 1
                        continue
                    out.write(json.dumps({
                        "doc_id": doc_id,
                        "title": url[:300],
                        "source": "web",
                        "n_words": n_tok,
                        "fps": fps,
                    }) + "\n")
                    own_ids.add(doc_id)
                    n_new += 1
                    f_new += 1
            except Exception as e:
                print(f"    parse error: {str(e)[:80]}", flush=True)
            dest.unlink(missing_ok=True)
            done_files.append(wpath)
            prog.update({"done_files": done_files, "n_new": n_new,
                         "n_skip": n_skip, "n_boiler": n_boiler,
                         "n_short": n_short})
            PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
            print(f"    {fname}: +{f_new} pages (total {n_new})", flush=True)
    finally:
        out.close()
        seen.close()
    prog.update({"done_files": done_files, "n_new": n_new, "n_skip": n_skip,
                 "n_boiler": n_boiler, "n_short": n_short,
                 "done": n_new >= target})
    PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
    print(f"R3_WEB DONE: new={n_new} skip={n_skip} boiler={n_boiler} "
          f"short={n_short}", flush=True)


if __name__ == "__main__":
    main()
