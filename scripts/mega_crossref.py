"""MEGA plagiarism index: Crossref works abstracts (fingerprint-only).

Fetches Crossref works with abstracts (keyless API, polite pool via mailto),
strips JATS XML, and appends winnowing fingerprints (k=5/w=4, same as
app/plag.py) to data/oa_corpus/fp_mega_crossref.jsonl. NO full text stored.

Dedupe: doc_id (crossref:{doi}) + title-hash against the existing index
(fp_winnow.jsonl + fp_mega*.jsonl snapshot at start).

Checkpoint: data/oa_corpus/mega_crossref_progress.json (cursor + counts).
Re-running resumes automatically.

Usage:
    python scripts/mega_crossref.py [target_docs]   (default 200000)
"""

from __future__ import annotations

import hashlib
import html as _html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT_ROOT / "data" / "oa_corpus"
OUT_JSONL = CORPUS / "fp_mega_crossref.jsonl"
PROGRESS = CORPUS / "mega_crossref_progress.json"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_plag_index import winnow, norm_word  # noqa: E402  (same K/W/hash)

MAILTO = "scout@example.com"
ROWS = 1000
MIN_TOKENS = 30
_JATS_TAG = re.compile(r"<[^>]+>")


def fetch_json(url: str, retries: int = 5):
    req = urllib.request.Request(
        url, headers={"User-Agent": f"NAK-AI-Detector/1.0 (mailto:{MAILTO})"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"  fetch err ({str(e)[:70]}), retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("crossref fetch failed")


def clean_abstract(jats: str | None) -> str | None:
    if not jats:
        return None
    t = _JATS_TAG.sub(" ", jats)
    t = _html.unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def title_hash(t: str) -> str:
    return hashlib.sha1(norm_title(t).encode()).hexdigest()[:16]


def load_seen() -> tuple[set[str], set[str]]:
    doc_ids: set[str] = set()
    thashes: set[str] = set()
    for path in [CORPUS / "fp_winnow.jsonl"] + sorted(CORPUS.glob("fp_mega*.jsonl")):
        if not path.exists():
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
    return doc_ids, thashes


def main() -> None:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    prog: dict = {}
    if PROGRESS.exists():
        try:
            prog = json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            prog = {}
    cursor = prog.get("cursor", "*")
    n_new = prog.get("n_new", 0)
    n_skip = prog.get("n_skip", 0)
    n_noabs = prog.get("n_noabs", 0)

    print("loading seen doc_ids + title hashes ...", flush=True)
    seen_ids, seen_th = load_seen()
    print(f"  {len(seen_ids)} doc_ids, {len(seen_th)} title hashes", flush=True)

    filt = "has-abstract:true"
    sel = "DOI,title,abstract,published"
    out = open(OUT_JSONL, "a", encoding="utf-8")
    try:
        while n_new < target:
            url = ("https://api.crossref.org/works?filter="
                   + urllib.parse.quote(filt)
                   + f"&rows={ROWS}&cursor={urllib.parse.quote(cursor)}"
                   + f"&select={sel}&mailto={MAILTO}")
            data = fetch_json(url)
            msg = data.get("message", {})
            items = msg.get("items", [])
            if not items:
                print("  no more items", flush=True)
                break
            for it in items:
                doi = (it.get("DOI") or "").strip().lower()
                if not doi:
                    continue
                doc_id = f"crossref:{doi}"
                if doc_id in seen_ids:
                    n_skip += 1
                    continue
                title = " ".join(it.get("title") or []).strip()
                if not title:
                    continue
                th = title_hash(title)
                if th in seen_th:
                    n_skip += 1
                    continue
                abs_txt = clean_abstract(it.get("abstract"))
                if not abs_txt:
                    n_noabs += 1
                    continue
                tokens = [t for t in (norm_word(w) for w in abs_txt.split()) if t]
                if len(tokens) < MIN_TOKENS:
                    continue
                fps = winnow(tokens)
                if not fps:
                    continue
                out.write(json.dumps({
                    "doc_id": doc_id,
                    "title": title[:300],
                    "source": "crossref",
                    "n_words": len(tokens),
                    "fps": fps,
                }) + "\n")
                seen_ids.add(doc_id)
                seen_th.add(th)
                n_new += 1
                if n_new >= target:
                    break
            cursor = msg.get("next-cursor")
            if not cursor:
                print("  cursor exhausted", flush=True)
                break
            if n_new % 5000 < ROWS:
                print(f"  crossref: new={n_new} skip={n_skip} noabs={n_noabs}",
                      flush=True)
            prog.update({"cursor": cursor, "n_new": n_new, "n_skip": n_skip,
                         "n_noabs": n_noabs})
            PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
            time.sleep(1.0)  # Crossref polite pool courtesy
    finally:
        out.close()
    prog.update({"cursor": cursor, "n_new": n_new, "n_skip": n_skip,
                 "n_noabs": n_noabs, "done": n_new >= target})
    PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
    print(f"CROSSREF DONE: new={n_new} skip={n_skip} noabs={n_noabs}", flush=True)


if __name__ == "__main__":
    main()
