#!/usr/bin/env python3
"""Validate score-mapping safety on real-human gate sets.

Loads models/score_mapping.json, applies it to raw document-level scores
of known-human gate docs, and checks they stay <20% (displayed as *%/0%).

Gate sets (never trained on):
- data/wiki_human_eval.jsonl (150 Wikipedia intros)
- 100 man pages (src/fetch_man.py outputs — resolved at runtime)
- data/pubmed_eval.jsonl (50 PubMed abstracts)
- data/proposal_chunks.jsonl (17 chunks of Nisar's proposal)

Each gate doc is scored at DOCUMENT level (mean of sentence scores,
same as the app), then mapped. FAIL if any mapped score >= 20.

Usage: .venv/bin/python scripts/validate_mapping.py [model_joblib]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import joblib
import numpy as np

from src.features import extract_features

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "models", "nak_detector.joblib")
MAP = os.path.join(ROOT, "models", "score_mapping.json")


def load_mapping():
    m = json.load(open(MAP))
    return np.array(m["xs"]), np.array(m["ys"])


def map_score(pct, xs, ys):
    return float(np.interp(pct, xs, ys))


def doc_score(texts, model):
    """Document-level score = mean of per-sample scores (app pipeline)."""
    scores = []
    for t in texts:
        f = np.asarray(extract_features(t), dtype=np.float64).reshape(1, -1)
        scores.append(float(model.predict_proba(f)[0][1]))
    return sum(scores) / len(scores) * 100 if scores else 0.0


def load_texts(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                out.append(json.loads(line)["text"])
    return out


def main():
    model = joblib.load(MODEL)
    xs, ys = load_mapping()
    gates = {
        "wiki_human_eval": load_texts(os.path.join(ROOT, "data", "wiki_human_eval.jsonl")),
        "pubmed_eval": load_texts(os.path.join(ROOT, "data", "pubmed_eval.jsonl")),
        "proposal_chunks": load_texts(os.path.join(ROOT, "data", "proposal_chunks.jsonl")),
    }
    # man pages if present
    man_path = os.path.join(ROOT, "data", "man_eval.jsonl")
    if os.path.exists(man_path):
        gates["man_eval"] = load_texts(man_path)

    worst = 0.0
    failed = []
    for name, texts in gates.items():
        raws, mappeds = [], []
        for t in texts:
            r = doc_score([t], model)
            m = map_score(r, xs, ys)
            raws.append(r)
            mappeds.append(m)
        mx = max(mappeds) if mappeds else 0.0
        n_over = sum(1 for m in mappeds if m >= 20)
        print(f"{name:18s} n={len(texts):4d} raw_max={max(raws):6.2f} "
              f"mapped_max={mx:6.2f} mapped>=20: {n_over}")
        worst = max(worst, mx)
        if n_over:
            failed.append(name)
    print(f"WORST mapped gate score: {worst:.2f}")
    if failed:
        print(f"RESULT: UNSAFE — gates breaching 20%: {failed}")
        sys.exit(1)
    print("RESULT: SAFE — all real-human gates stay <20% after mapping")


if __name__ == "__main__":
    main()
