"""Build the plagiarism winnowing fingerprint index from the OA corpus.

Reads every *.txt under data/oa_corpus/<source>/, computes position-aware
winnowing fingerprints (k=5 word shingles, window w=4, md5 -> SIGNED int64),
and writes data/oa_corpus/fp_winnow.jsonl, one line per doc:
    {"doc_id": ..., "title": ..., "source": ..., "n_words": ..., "fps": [[h, pos], ...]}

This is SEPARATE from the MinHash sigs (fp_<source>.jsonl) built by the OA
sweep for corpus dedup: MinHash estimates whole-doc Jaccard but carries no
positions; plagiarism highlighting needs (hash -> word offset), hence
winnowing.

Coordination: if you change K/W, rebuild this file AND keep app/plag.py's
Winnower in sync (same K, W, normalization, hash function).

Re-run any time new corpus docs land: it skips doc_ids already indexed
(unless --rebuild).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT_ROOT / "data" / "oa_corpus"
OUT = CORPUS / "fp_winnow.jsonl"

K = 5  # shingle size in words
W = 4  # winnowing window (detects matches >= K+W-1 = 8 words)

_PUNCT = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def norm_word(w: str) -> str:
    w = w.lower()
    w = _PUNCT.sub("", w)
    return w


def _h64(s: str) -> int:
    # SIGNED 64-bit (SQLite-compatible). The min() tie-break in winnow()
    # MUST use these signed values — keep in sync with app/plag.py.
    h = int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")
    h &= (1 << 64) - 1
    return h - (1 << 64) if h >= (1 << 63) else h


def winnow(tokens: list[str], k: int = K, w: int = W) -> list[tuple[int, int]]:
    """Return [(hash, kgram_start_pos)] winnowing fingerprints.

    Standard winnowing: in every window of w consecutive k-gram hashes,
    select the minimum (rightmost on ties). Guarantees detection of any
    shared substring of length >= k + w - 1 words.
    """
    n = len(tokens)
    if n < k:
        return []
    hashes = []
    for i in range(n - k + 1):
        hashes.append((_h64(" ".join(tokens[i:i + k])), i))
    out: list[tuple[int, int]] = []
    last = None
    # windows of w hashes: window j covers hashes[j .. j+w-1]
    for j in range(len(hashes) - w + 1):
        window = hashes[j:j + w]
        m = min(window, key=lambda t: (t[0], -t[1]))  # min hash, rightmost tie
        if m != last:
            out.append(m)
            last = m
    # tail: k-grams never covered by a full window (last w-1 of them)
    # (standard winnowing ignores them; matches >= k+w-1 are still guaranteed)
    return out


def title_for(txt_path: Path, docs_jsonl: Path | None) -> str:
    return txt_path.stem


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    seen: set[str] = set()
    if OUT.exists() and not rebuild:
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        seen.add(json.loads(line)["doc_id"])
                    except Exception:
                        pass
    # doc_id -> title lookup from <source>_docs.jsonl manifests
    titles: dict[str, str] = {}
    for manifest in CORPUS.glob("*_docs.jsonl"):
        try:
            with open(manifest, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    titles[d.get("doc_id", "")] = d.get("title", "") or d.get("doc_id", "")
        except Exception:
            pass

    n_new = 0
    n_skip = 0
    mode = "w" if rebuild else "a"
    with open(OUT, mode, encoding="utf-8") as out:
        for txt in sorted(CORPUS.glob("*/*.txt")):
            if txt.name == "":
                continue
            # doc_id: prefer the manifest mapping via stem lookup
            # doc_id mirrors the manifest format, e.g. "openalex:W3038568908"
            stem = txt.stem
            source = txt.parent.name
            doc_id = f"{source}:{stem}"
            if doc_id in seen:
                n_skip += 1
                continue
            try:
                text = txt.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            tokens = [t for t in (norm_word(w) for w in text.split()) if t]
            if len(tokens) < 30:
                continue
            fps = winnow(tokens)
            title = titles.get(doc_id, "") or next(
                (t for did, t in titles.items() if did.endswith(stem)), stem
            )
            out.write(json.dumps({
                "doc_id": doc_id,
                "title": title,
                "source": source,
                "n_words": len(tokens),
                "fps": fps,
            }) + "\n")
            n_new += 1
    print(f"indexed {n_new} new docs ({n_skip} already indexed) -> {OUT}")


if __name__ == "__main__":
    main()
