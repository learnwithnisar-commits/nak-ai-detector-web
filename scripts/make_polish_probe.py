#!/usr/bin/env python3
"""Build a HELD-OUT polish probe: 100 fresh abstracts (doc_ids not in
training pairs), polished with the same simulator. Used to verify the
polish signal generalizes (mini-LOGO for the new genre).
Output: data/ai_polished_probe.jsonl
"""
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from make_ai_polished import polish

DATA = os.path.join(ROOT, "data")
CORP = os.path.join(DATA, "oa_corpus")
SEED = 777
N = 100
SOURCES = ["openalex", "arxiv", "europepmc", "semanticscholar", "doaj"]


def norm_hash(t):
    return hashlib.sha1("".join(t.lower().split()).encode()).hexdigest()


def main():
    rng = random.Random(SEED)
    train_ids = set(json.loads(l)["doc_id"] for l in
                    open(os.path.join(DATA, "polish_pairs.jsonl")))
    cands = []
    for line in open(os.path.join(CORP, "manifest.jsonl"), encoding="utf-8"):
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("source") not in SOURCES or not o.get("train_eligible"):
            continue
        nw = o.get("n_words") or 0
        if not (100 <= nw <= 500):
            continue
        if o["doc_id"] in train_ids:
            continue
        cands.append(o)
    rng.shuffle(cands)
    out = os.path.join(DATA, "ai_polished_probe.jsonl")
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for o in cands:
            if n >= N:
                break
            tp = os.path.join(CORP, o["text_path"])
            try:
                text = open(tp, encoding="utf-8", errors="ignore").read().strip()
            except Exception:
                continue
            if not (100 <= len(text.split()) <= 600):
                continue
            if sum(1 for c in text if ord(c) < 128) / max(1, len(text)) < 0.92:
                continue
            seed = int(hashlib.md5(o["doc_id"].encode()).hexdigest(), 16) % (2 ** 31)
            pol = polish(text, seed)
            if pol == text:
                continue
            wr = len(pol.split()) / max(1, len(text.split()))
            if not (0.80 <= wr <= 1.25):
                continue
            f.write(json.dumps({"text": text, "label": 0,
                                "doc_id": o["doc_id"]}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"text": pol, "label": 1,
                                "doc_id": o["doc_id"]}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {out}: {n} probe pairs", flush=True)


if __name__ == "__main__":
    main()
