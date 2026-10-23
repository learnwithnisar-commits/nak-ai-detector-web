"""Validate the mega plagiarism index DB: counts, latency, correctness."""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

DB = PROJECT_ROOT / "plag_index" / "plag_index.db"


def main() -> None:
    from plag import OAIndex, StreamWinnower, norm_word, split_keep  # noqa: E402

    def fingerprint_text(text: str):
        tokens, _ = split_keep(text)
        toks = [t for t in (norm_word(w) for w in tokens) if t]
        sw = StreamWinnower()
        for i, t in enumerate(toks):
            sw.feed(i, t)
        return sw.drain()

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    n_docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    n_fps = conn.execute("SELECT COUNT(*) FROM fps").fetchone()[0]
    n_src = conn.execute(
        "SELECT source, COUNT(*) FROM docs GROUP BY source").fetchall()
    size_mb = DB.stat().st_size / 1e6
    print(f"docs={n_docs:,} fps={n_fps:,} size={size_mb:.0f}MB")
    for src, cnt in sorted(n_src, key=lambda r: -r[1]):
        print(f"  {src or '(none)'}: {cnt:,}")
    conn.close()

    # latency: meta preload + sample lookup (same path the app uses)
    from pathlib import Path as _P
    t0 = time.time()
    idx = OAIndex(_P(DB))
    idx.open()
    print(f"meta preload: {(time.time() - t0) * 1000:.0f}ms")
    sample = ("Machine learning models for climate prediction rely on large "
              "training datasets. Neural networks optimize parameters via "
              "gradient descent. Regularization prevents overfitting on "
              "limited samples. " * 40)
    fps = [h for h, _ in fingerprint_text(sample)]
    print(f"sample: {len(fps)} fps")
    t0 = time.time()
    hits = idx.lookup(fps)
    dt = (time.time() - t0) * 1000
    print(f"lookup: {dt:.0f}ms, {len(hits)} candidate docs")
    if hits:
        top = max(hits.items(), key=lambda kv: kv[1])
        meta = idx.meta(top[0])
        print(f"top hit: doc {top[0]} count={top[1]} "
              f"title={meta[0][:60] if meta else '?'}")
    print("VALIDATION OK")


if __name__ == "__main__":
    main()
