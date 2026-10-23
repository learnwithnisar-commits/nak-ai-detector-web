#!/usr/bin/env python3
"""FPR gate evaluation: real human texts must stay <20% AI.

Scores each doc with the model (full-text proba * 100) and reports
per-set mean AI% + fraction >= 20% (breach fraction).
Usage: .venv/bin/python scripts/eval_gates.py [model_joblib] [artifacts_joblib]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import joblib
import numpy as np

from src.features import extract_features, set_lm

DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")

GATES = [
    ("wiki_intro", "wiki_human_eval.jsonl"),
    ("pubmed", "pubmed_eval.jsonl"),
    ("proposal", "proposal_chunks.jsonl"),
    ("real_human", "real_human_eval.jsonl"),
    ("arxiv", "arxiv_eval.jsonl"),
    ("europepmc", "europepmc_eval.jsonl"),
]


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        MODELS, "nak_detector.joblib")
    art_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        MODELS, "feature_artifacts.joblib")
    bundle = joblib.load(art_path)
    if bundle.get("lm") is not None:
        set_lm(bundle["lm"])
    model = joblib.load(model_path)

    report = {"model": model_path, "gates": {}}
    worst_mean, worst_breach = 0.0, 0.0
    for name, fn in GATES:
        p = os.path.join(DATA, fn)
        if not os.path.exists(p):
            continue
        texts = [json.loads(l)["text"] for l in open(p, encoding="utf-8")
                 if l.strip()]
        X = np.stack([extract_features(t) for t in texts])
        pcts = model.predict_proba(X)[:, 1] * 100.0
        mean_ai = float(np.mean(pcts))
        breach = float(np.mean(pcts >= 20.0))
        mx = float(np.max(pcts))
        report["gates"][name] = {
            "n": len(texts), "mean_ai_pct": round(mean_ai, 2),
            "breach_frac_ge20": round(breach, 4),
            "max_ai_pct": round(mx, 2),
            "PASS": bool(mean_ai < 20.0),
        }
        worst_mean = max(worst_mean, mean_ai)
        worst_breach = max(worst_breach, breach)
        print(f"{name:12s} n={len(texts):4d} mean={mean_ai:6.2f}% "
              f"breach(>=20%)={breach:.3f} max={mx:6.2f}% "
              f"{'PASS' if mean_ai < 20 else 'FAIL'}", flush=True)
    report["worst_mean_ai_pct"] = round(worst_mean, 2)
    report["worst_breach_frac"] = round(worst_breach, 4)
    report["verdict"] = "PASS" if worst_mean < 20.0 else "FAIL"
    print("verdict:", report["verdict"], flush=True)
    out = os.path.join(DATA, "calibration", "gates_scored.json")
    json.dump(report, open(out, "w"), indent=1)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
