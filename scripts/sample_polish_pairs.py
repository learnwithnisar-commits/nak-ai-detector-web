#!/usr/bin/env python3
"""Sample ~800 diverse English abstracts from oa_corpus for AI-polish pairs.

Stratified by source, 100-500 words, deduped against all eval sets.
Output: data/polish_pairs.jsonl  [{doc_id, source, title, original}]
"""
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
CORP = os.path.join(DATA, "oa_corpus")

SEED = 20261012
N_TARGET = 800
SOURCES = ["openalex", "arxiv", "europepmc", "semanticscholar", "doaj"]
MIN_W, MAX_W = 100, 500

EVAL_FILES = [
    "wiki_human_eval.jsonl", "pubmed_eval.jsonl", "real_human_eval.jsonl",
    "arxiv_eval.jsonl", "europepmc_eval.jsonl", "proposal_chunks.jsonl",
    "wiki_full_eval.jsonl", "test.jsonl",
]


def norm_hash(text):
    t = "".join(text.lower().split())
    return hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest()


def load_eval_hashes():
    hs = set()
    for f in EVAL_FILES:
        p = os.path.join(DATA, f)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                hs.add(norm_hash(o.get("text", "")))
            except Exception:
                pass
    return hs


def is_english(text):
    if not text:
        return False
    ascii_n = sum(1 for c in text if ord(c) < 128)
    return ascii_n / max(1, len(text)) > 0.92


def main():
    rng = random.Random(SEED)
    eval_hashes = load_eval_hashes()
    print(f"eval hashes: {len(eval_hashes)}", flush=True)

    by_src = {s: [] for s in SOURCES}
    seen = set()
    for line in open(os.path.join(CORP, "manifest.jsonl"), encoding="utf-8"):
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("source") not in SOURCES:
            continue
        if not o.get("train_eligible"):
            continue
        nw = o.get("n_words") or 0
        if not (MIN_W <= nw <= MAX_W):
            continue
        by_src[o["source"]].append(o)

    for s in SOURCES:
        print(f"  candidate {s}: {len(by_src[s])}", flush=True)

    # stratified: equal per source, fallback to proportional if short
    per = N_TARGET // len(SOURCES)
    picked = []
    for s in SOURCES:
        cands = by_src[s][:]
        rng.shuffle(cands)
        picked.extend(cands[:per])
    # top up from largest pools if any source was short
    if len(picked) < N_TARGET:
        rest = [o for s in SOURCES for o in by_src[s]
                if o not in picked]
        rng.shuffle(rest)
        picked.extend(rest[:N_TARGET - len(picked)])

    out_path = os.path.join(DATA, "polish_pairs.jsonl")
    n_out, n_skip = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for o in picked:
            tp = os.path.join(CORP, o["text_path"])
            try:
                with open(tp, encoding="utf-8", errors="ignore") as fh:
                    text = fh.read().strip()
            except Exception:
                n_skip += 1
                continue
            nw = len(text.split())
            if not (MIN_W <= nw <= MAX_W + 100):
                n_skip += 1
                continue
            if not is_english(text):
                n_skip += 1
                continue
            h = norm_hash(text)
            if h in eval_hashes or h in seen:
                n_skip += 1
                continue
            seen.add(h)
            f.write(json.dumps({
                "doc_id": o["doc_id"], "source": o["source"],
                "title": o.get("title", ""), "original": text,
            }, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"wrote {out_path}: {n_out} pairs, skipped {n_skip}", flush=True)


if __name__ == "__main__":
    main()
