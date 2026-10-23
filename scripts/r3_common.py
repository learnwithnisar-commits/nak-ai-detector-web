"""Round-3 mega-index shared helpers.

Dedupe for round 3 is memory-light: all legs check new candidates against
data/oa_corpus/r3_seen.db (doc_id + title-hash snapshot of the 905k-doc
round-1/2 mega DB, built once by the coordinator). Each leg ALSO keeps an
in-memory set of its own newly added (doc_id, title-hash) pairs for
intra-leg dedupe. Cross-leg title collisions are resolved in a post-pass
before the merge build.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT_ROOT / "data" / "oa_corpus"
SEEN_DB = CORPUS / "r3_seen.db"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_plag_index import winnow, norm_word  # noqa: E402  (same K/W/hash)

__all__ = ["PROJECT_ROOT", "CORPUS", "winnow", "norm_word", "title_hash",
           "Seen", "fingerprint_text"]


def title_hash(t: str) -> str:
    return hashlib.sha1(
        re.sub(r"[^a-z0-9]", "", (t or "").lower()).encode()).hexdigest()[:16]


def fingerprint_text(text: str, min_tokens: int = 30,
                     max_tokens: int = 4000) -> tuple[list[tuple[int, int]], int]:
    """Tokenize + winnow. Returns (fps, n_tokens). Empty fps if too short."""
    toks = [t for t in (norm_word(w) for w in text.split()) if t]
    if len(toks) < min_tokens:
        return [], len(toks)
    if len(toks) > max_tokens:
        toks = toks[:max_tokens]
    fps = winnow(toks)
    return (fps if fps else []), len(toks)


class Seen:
    """Read-only dedupe handle over r3_seen.db (round-1/2 mega DB snapshot)."""

    def __init__(self, path: Path | None = None):
        self.conn = sqlite3.connect(
            f"file:{path or SEEN_DB}?mode=ro", uri=True, check_same_thread=False)

    def has_doc(self, doc_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM seen WHERE doc_id=?", (doc_id,)).fetchone() is not None

    def has_th(self, th: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM seen WHERE th=?", (th,)).fetchone() is not None

    def close(self) -> None:
        self.conn.close()
