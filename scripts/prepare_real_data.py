#!/usr/bin/env python3
"""Prepare new real data for training.

- Reads data/real_human/arxiv_raw.jsonl + europepmc_raw.jsonl
- Dedupes (by id), shuffles with seed, splits 85% train / 15% eval
- Writes: arxiv_train.jsonl / arxiv_eval.jsonl,
           europepmc_train.jsonl / europepmc_eval.jsonl
  (eval files are NEVER trained on — held-out discipline)

AI samples are generated separately by gen_ai_on_papers.py, which reads
these split files and generates train-AI on train papers and eval-AI on
eval papers (disjoint topics).
"""
import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUMAN_DIR = os.path.join(ROOT, "data", "real_human")
DATA = os.path.join(ROOT, "data")
SEED = 20261012
EVAL_FRAC = 0.15

def load_raw(fname):
    p = os.path.join(HUMAN_DIR, fname)
    recs = []
    if not os.path.exists(p):
        print(f"missing {fname}, skipping")
        return recs
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs

def slim(r):
    return {
        "text": r["text"], "label": 0, "genre": "academic_abstract",
        "source": r.get("source"), "title": r.get("title", "")[:150],
        "words": r.get("words"),
    }

def main():
    rng = random.Random(SEED)
    for raw_name, stem in (("arxiv_raw.jsonl", "arxiv"),
                           ("europepmc_raw.jsonl", "europepmc")):
        recs = load_raw(raw_name)
        # dedupe by id
        seen, uniq = set(), []
        for r in recs:
            key = r.get("arxiv_id") or r.get("pmid")
            if key and key not in seen:
                seen.add(key)
                uniq.append(r)
        rng.shuffle(uniq)
        n_eval = max(25, int(len(uniq) * EVAL_FRAC))
        eval_recs, train_recs = uniq[:n_eval], uniq[n_eval:]
        for split_recs, suffix in ((train_recs, "train"), (eval_recs, "eval")):
            out = os.path.join(DATA, f"{stem}_{suffix}.jsonl")
            with open(out, "w", encoding="utf-8") as f:
                for r in split_recs:
                    f.write(json.dumps(slim(r), ensure_ascii=False) + "\n")
            print(f"{stem}_{suffix}: {len(split_recs)} -> {out}")

if __name__ == "__main__":
    main()
