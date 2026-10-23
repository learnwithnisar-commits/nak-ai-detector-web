"""Round-3: Crossref type:dissertation, 150k -> 427k (fingerprint-only).

Closest PUBLIC proxy for Turnitin's private "student papers" repository.
Resumes from the round-2 cursor (data/oa_corpus/mega_theses_progress.json)
and appends to data/oa_corpus/fp_r3_theses.jsonl.

Dedupe: r3_seen.db (round-1/2) + own in-memory set. Note the general
r3_crossref leg skips dissertations, so the 'thesis' source label wins.
Checkpoint: data/oa_corpus/r3_theses_progress.json.

Usage: python scripts/r3_theses.py [new_target]  (default 277000)
"""

from __future__ import annotations

import html as _html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_common import (  # noqa: E402
    CORPUS, title_hash, Seen, fingerprint_text)

OUT_JSONL = CORPUS / "fp_r3_theses.jsonl"
PROGRESS = CORPUS / "r3_theses_progress.json"
OLD_PROGRESS = CORPUS / "mega_theses_progress.json"

MAILTO = "scout@example.com"
ROWS = 1000
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
    raise RuntimeError("crossref theses r3 fetch failed")


def clean_abstract(jats: str | None) -> str | None:
    if not jats:
        return None
    t = _JATS_TAG.sub(" ", jats)
    t = _html.unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


def main() -> None:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 277_000
    prog: dict = {}
    if PROGRESS.exists():
        try:
            prog = json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            prog = {}
    cursor = prog.get("cursor")
    if not cursor and OLD_PROGRESS.exists():
        try:
            cursor = json.loads(OLD_PROGRESS.read_text()).get("cursor", "*")
        except Exception:
            cursor = "*"
    cursor = cursor or "*"
    n_new = prog.get("n_new", 0)
    n_skip = prog.get("n_skip", 0)
    n_noabs = prog.get("n_noabs", 0)

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
    print(f"r3_theses: resume new={n_new}, own={len(own_ids)}", flush=True)

    filt = "type:dissertation,has-abstract:true"
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
                if doc_id in own_ids or seen.has_doc(doc_id):
                    n_skip += 1
                    continue
                title = " ".join(it.get("title") or []).strip()
                if not title:
                    continue
                th = title_hash(title)
                if th in own_th or seen.has_th(th):
                    n_skip += 1
                    continue
                abs_txt = clean_abstract(it.get("abstract"))
                if not abs_txt:
                    n_noabs += 1
                    continue
                fps, n_tok = fingerprint_text(abs_txt)
                if not fps:
                    continue
                out.write(json.dumps({
                    "doc_id": doc_id,
                    "title": title[:300],
                    "source": "thesis",
                    "n_words": n_tok,
                    "fps": fps,
                }) + "\n")
                own_ids.add(doc_id)
                own_th.add(th)
                n_new += 1
                if n_new >= target:
                    break
            cursor = msg.get("next-cursor")
            if not cursor:
                print("  cursor exhausted", flush=True)
                break
            if n_new % 5000 < ROWS:
                print(f"  r3_theses: new={n_new} skip={n_skip} noabs={n_noabs}",
                      flush=True)
            prog.update({"cursor": cursor, "n_new": n_new, "n_skip": n_skip,
                         "n_noabs": n_noabs})
            PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
            time.sleep(1.0)
    finally:
        out.close()
        seen.close()
    prog.update({"cursor": cursor, "n_new": n_new, "n_skip": n_skip,
                 "n_noabs": n_noabs, "done": n_new >= target})
    PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
    print(f"R3_THESES DONE: new={n_new} skip={n_skip} noabs={n_noabs}",
          flush=True)


if __name__ == "__main__":
    main()
