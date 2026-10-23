"""MEGA plagiarism index expansion — fingerprint-only, resumable, keyless sources.

Appends winnowing fingerprints (k=5, w=4, md5 -> signed int64 — IDENTICAL
to scripts/build_plag_index.py and app/plag.py) to
data/oa_corpus/fp_mega.jsonl, one line per doc:
    {"doc_id": ..., "title": ..., "source": ..., "n_words": ..., "fps": [[h, pos], ...]}

NO full text is stored (index stays lean). Dedupe: doc_ids already present
in fp_winnow.jsonl or fp_mega.jsonl are skipped.

Sources (all keyless, no account, no card):
  - openalex_expansion: OpenAlex API (polite pool, mailto), date slices NOT
    covered by the original sweep (which did is_oa from 2015-01-01, 30k docs).

Checkpoint: data/oa_corpus/mega_progress.json (per-source cursor + counts).
Re-running resumes automatically.

Usage:
    python scripts/mega_plag_index.py [target_docs]
Then build the shippable DB:
    python scripts/build_mega_db.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT_ROOT / "data" / "oa_corpus"
WINNOW_JSONL = CORPUS / "fp_winnow.jsonl"
MEGA_JSONL = CORPUS / "fp_mega.jsonl"
PROGRESS = CORPUS / "mega_progress.json"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_plag_index import winnow, norm_word  # noqa: E402  (same K/W/hash)

MAILTO = "scout@example.com"
PER_PAGE = 200
MIN_TOKENS = 30


def fetch_json(url: str, retries: int = 5):
    req = urllib.request.Request(
        url, headers={"User-Agent": f"NAK-AI-Detector/1.0 (mailto:{MAILTO})"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except Exception as e:
            wait = min(2 ** attempt, 30)
            print(f"  fetch err ({str(e)[:70]}), retry in {wait}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"fetch failed after {retries} retries: {url[:120]}")


def recon_abstract(inv: dict | None) -> str | None:
    if not inv:
        return None
    pos: dict[int, str] = {}
    for w, idxs in inv.items():
        for i in idxs:
            pos[i] = w
    if not pos:
        return None
    return " ".join(pos[i] for i in sorted(pos))


def load_seen() -> set[str]:
    seen: set[str] = set()
    for path in (WINNOW_JSONL, MEGA_JSONL):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    seen.add(json.loads(line)["doc_id"])
                except Exception:
                    pass
    return seen


def load_progress() -> dict:
    if PROGRESS.exists():
        try:
            return json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_progress(p: dict) -> None:
    tmp = PROGRESS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(p), encoding="utf-8")
    tmp.replace(PROGRESS)


def run_openalex(target: int, seen: set[str], prog: dict) -> int:
    """Fetch pre-2015 OA works (slice the original sweep did not cover)."""
    key = "openalex_expansion"
    st = prog.get(key, {})
    cursor = st.get("cursor", "*")
    n_new = st.get("n_new", 0)
    n_skip = st.get("n_skip", 0)
    n_noabs = st.get("n_noabs", 0)

    # Slices not covered by the original sweep (is_oa, from 2015-01-01).
    slices = st.get("slices") or [
        "is_oa:true,to_publication_date:2014-12-31",
        "is_oa:true,from_publication_date:2024-01-01",
    ]
    slice_idx = st.get("slice_idx", 0)
    if slice_idx >= len(slices):
        print("openalex_expansion: all slices done", flush=True)
        return n_new

    out = open(MEGA_JSONL, "a", encoding="utf-8")
    try:
        while n_new < target and slice_idx < len(slices):
            filt = slices[slice_idx]
            url = (
                "https://api.openalex.org/works?filter="
                + urllib.parse.quote(filt)
                + f"&per-page={PER_PAGE}&cursor={urllib.parse.quote(cursor)}"
                + "&select=id,doi,title,abstract_inverted_index"
                + f"&mailto={MAILTO}"
            )
            data = fetch_json(url)
            results = data.get("results", [])
            if not results:
                print(f"  slice {slice_idx} exhausted", flush=True)
                slice_idx += 1
                cursor = "*"
                continue
            for r in results:
                wid = (r.get("id") or "").split("/")[-1]
                if not wid:
                    continue
                doc_id = f"openalex:{wid}"
                if doc_id in seen:
                    n_skip += 1
                    continue
                title = r.get("title") or ""
                if not title or title.lower() == "null":
                    continue
                abs_txt = recon_abstract(r.get("abstract_inverted_index"))
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
                    "source": "openalex",
                    "n_words": len(tokens),
                    "fps": fps,
                }) + "\n")
                seen.add(doc_id)
                n_new += 1
                if n_new >= target:
                    break
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                print(f"  slice {slice_idx} exhausted (no cursor)",
                      flush=True)
                slice_idx += 1
                cursor = "*"
                continue
            if n_new % 2000 < PER_PAGE:
                print(f"  openalex_expansion: new={n_new} skip={n_skip} "
                      f"noabs={n_noabs}", flush=True)
            prog[key] = {"cursor": cursor, "n_new": n_new, "n_skip": n_skip,
                         "n_noabs": n_noabs, "slices": slices,
                         "slice_idx": slice_idx}
            save_progress(prog)
            time.sleep(0.15)  # polite pool courtesy
    finally:
        out.close()
    prog[key] = {"cursor": cursor, "n_new": n_new, "n_skip": n_skip,
                 "n_noabs": n_noabs, "slices": slices,
                 "slice_idx": slice_idx, "done": n_new >= target}
    save_progress(prog)
    print(f"openalex_expansion finished: new={n_new} skip={n_skip} "
          f"noabs={n_noabs}", flush=True)
    return n_new


def main() -> None:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    print(f"loading seen doc_ids ...", flush=True)
    seen = load_seen()
    print(f"  {len(seen)} already indexed", flush=True)
    prog = load_progress()
    total = run_openalex(target, seen, prog)
    print(f"MEGA DONE: {total} new docs -> {MEGA_JSONL}", flush=True)


if __name__ == "__main__":
    main()
