"""MEGA plagiarism index: Internet Archive texts pilot (fingerprint-only).

Fetches popular English text items from archive.org (keyless
advancedsearch API), downloads the OCR text ({id}_djvu.txt), and appends
winnowing fingerprints (k=5/w=4, same as app/plag.py) to
data/oa_corpus/fp_mega_ia.jsonl. NO full text stored.

Books are fingerprint-heavy (~0.4 fps/word); this pilot is SIZE-GUARDED:
it stops when the output jsonl approaches SIZE_CAP_MB (default 250MB)
to keep the shippable DB lean. Dedupe by ia:{identifier} + title-hash.

Checkpoint: data/oa_corpus/mega_ia_progress.json — rerunning resumes.

Usage:
    python scripts/mega_ia.py [max_items] [size_cap_mb]
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT_ROOT / "data" / "oa_corpus"
OUT_JSONL = CORPUS / "fp_mega_ia.jsonl"
PROGRESS = CORPUS / "mega_ia_progress.json"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_plag_index import winnow, norm_word  # noqa: E402  (same K/W/hash)

MIN_TOKENS = 30
MAX_WORDS = 150_000  # skip absurdly large OCR dumps


def fetch_json(url: str, timeout: int = 30):
    req = urllib.request.Request(
        url, headers={"User-Agent": "NAK-AI-Detector/1.0 (research index)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fetch_text(url: str, timeout: int = 120, max_bytes: int = 8_000_000) -> str | None:
    req = urllib.request.Request(
        url, headers={"User-Agent": "NAK-AI-Detector/1.0 (research index)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None
            data = r.read(max_bytes + 1)
    except Exception:
        return None
    if len(data) > max_bytes:
        return None
    return data.decode("utf-8", errors="replace")


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def title_hash(t: str) -> str:
    return hashlib.sha1(norm_title(t).encode()).hexdigest()[:16]


def load_seen() -> tuple[set[str], set[str]]:
    doc_ids: set[str] = set()
    thashes: set[str] = set()
    for path in [CORPUS / "fp_winnow.jsonl"] + sorted(CORPUS.glob("fp_mega*.jsonl")):
        if path == OUT_JSONL or not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                doc_ids.add(d.get("doc_id", ""))
                th = d.get("title") or ""
                if th:
                    thashes.add(title_hash(th))
    # also our own output (resume)
    if OUT_JSONL.exists():
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                doc_ids.add(d.get("doc_id", ""))
                th = d.get("title") or ""
                if th:
                    thashes.add(title_hash(th))
    return doc_ids, thashes


def search_items(page: int, rows: int = 200) -> list[dict]:
    q = "mediatype:texts AND language:eng"
    url = ("https://archive.org/advancedsearch.php?"
           + urllib.parse.urlencode({
               "q": q, "fl": "identifier,title,downloads",
               "rows": rows, "page": page,
               "sort": "downloads desc", "output": "json"}))
    data = fetch_json(url)
    return data.get("response", {}).get("docs", [])


def main() -> None:
    max_items = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    size_cap_mb = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
    prog: dict = {}
    if PROGRESS.exists():
        try:
            prog = json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            prog = {}
    page = prog.get("page", 1)
    n_new = prog.get("n_new", 0)
    n_skip = prog.get("n_skip", 0)

    print("loading seen doc_ids + title hashes ...", flush=True)
    seen_ids, seen_th = load_seen()
    print(f"  {len(seen_ids)} doc_ids, {len(seen_th)} title hashes", flush=True)

    out = open(OUT_JSONL, "a", encoding="utf-8")
    try:
        while n_new < max_items:
            if OUT_JSONL.exists() and OUT_JSONL.stat().st_size / 1e6 >= size_cap_mb:
                print(f"  size cap {size_cap_mb}MB reached, stopping", flush=True)
                break
            try:
                items = search_items(page)
            except Exception as e:
                print(f"  search err: {str(e)[:70]}, retry", flush=True)
                time.sleep(5)
                continue
            if not items:
                print("  no more items", flush=True)
                break
            for it in items:
                ident = it.get("identifier") or ""
                if not ident:
                    continue
                doc_id = f"ia:{ident}"
                if doc_id in seen_ids:
                    n_skip += 1
                    continue
                title = (it.get("title") or "").strip() or ident
                th = title_hash(title)
                if th in seen_th:
                    n_skip += 1
                    continue
                txt = fetch_text(
                    f"https://archive.org/download/{ident}/{ident}_djvu.txt")
                if not txt:
                    n_skip += 1
                    continue
                # strip Project Gutenberg-style headers/footers crudely: keep
                # the middle 90% to avoid boilerplate fingerprints
                tokens = [t for t in (norm_word(w) for w in txt.split()) if t]
                if len(tokens) < MIN_TOKENS:
                    continue
                if len(tokens) > MAX_WORDS:
                    tokens = tokens[:MAX_WORDS]
                fps = winnow(tokens)
                if not fps:
                    continue
                out.write(json.dumps({
                    "doc_id": doc_id,
                    "title": title[:300],
                    "source": "ia",
                    "n_words": len(tokens),
                    "fps": fps,
                }) + "\n")
                out.flush()
                seen_ids.add(doc_id)
                seen_th.add(th)
                n_new += 1
                if n_new % 50 == 0:
                    sz = OUT_JSONL.stat().st_size / 1e6
                    print(f"  ia: new={n_new} skip={n_skip} size={sz:.0f}MB",
                          flush=True)
                if n_new >= max_items:
                    break
                if OUT_JSONL.stat().st_size / 1e6 >= size_cap_mb:
                    break
            page += 1
            prog.update({"page": page, "n_new": n_new, "n_skip": n_skip})
            PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
            time.sleep(0.5)
    finally:
        out.close()
    prog.update({"page": page, "n_new": n_new, "n_skip": n_skip, "done": True})
    PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
    print(f"IA DONE: new={n_new} skip={n_skip}", flush=True)


if __name__ == "__main__":
    main()
