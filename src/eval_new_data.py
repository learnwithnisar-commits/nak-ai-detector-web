#!/usr/bin/env python3
"""Evaluate a model bundle on the NEW held-out sets (never trained on):
  - data/arxiv_eval.jsonl + data/europepmc_eval.jsonl  (real human -> FPR)
  - data/real_ai/real_ai_eval.jsonl                    (synthetic AI -> recall)

Usage: .venv/bin/python src/eval_new_data.py [--models-dir DIR]
Compares old (backup) vs new model by pointing at different dirs.
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from src.features import extract_features, set_lm  # noqa: E402

DATA = os.path.join(ROOT, "data")

def load_set(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.path.join(ROOT, "models"))
    a = ap.parse_args()

    bundle = joblib.load(os.path.join(a.models_dir, "feature_artifacts.joblib"))
    if bundle.get("lm") is not None:
        set_lm(bundle["lm"])
    model = joblib.load(os.path.join(a.models_dir, "nak_detector.joblib"))
    print(f"model from {a.models_dir} (dim {bundle.get('dim', '?')})", flush=True)

    sets = [
        ("arxiv_eval (human)", os.path.join(DATA, "arxiv_eval.jsonl"), 0),
        ("europepmc_eval (human)", os.path.join(DATA, "europepmc_eval.jsonl"), 0),
        ("real_ai_eval (AI)", os.path.join(DATA, "real_ai", "real_ai_eval.jsonl"), 1),
    ]
    for name, path, label in sets:
        recs = load_set(path)
        texts = [r["text"] for r in recs]
        X = np.stack([extract_features(t) for t in texts])
        p = model.predict_proba(X)[:, 1]
        for thr in (0.5, 0.7):
            if label == 0:
                m = float((p >= thr).mean())
                print(f"  {name:26s} n={len(recs):4d} FPR@{thr} = {m:.4f} "
                      f"(mean_p={p.mean():.4f} max_p={p.max():.4f})", flush=True)
            else:
                m = float((p >= thr).mean())
                print(f"  {name:26s} n={len(recs):4d} recall@{thr} = {m:.4f} "
                      f"(mean_p={p.mean():.4f} min_p={p.min():.4f})", flush=True)

if __name__ == "__main__":
    main()
