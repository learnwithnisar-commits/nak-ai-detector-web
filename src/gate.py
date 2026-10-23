#!/usr/bin/env python3
"""SHIP GATE (morning demo): decides whether the current model may ship.

Measures on REAL human text (never trained on):
  - data/wiki_human_eval.jsonl      (150 Wikipedia intros)
  - data/wiki_full_eval.jsonl       (50 Wikipedia article extracts, if present)
  - data/real_human_eval.jsonl      (100 man-page samples)
  - data/proposal_chunks.jsonl      (17 real academic proposal chunks)
And on synthetic test.jsonl (AI recall must not collapse).

PASS: real-human FPR <= 5% on every set (low single digits) AND
      test.jsonl AI recall >= 0.95 AND no proposal chunk scores >= 0.9.
Otherwise: FAIL -> run the --no-lm fallback.
"""
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
MODELS = os.path.join(ROOT, "models")

REAL_SETS = [
    ("wiki_intro", "wiki_human_eval.jsonl"),
    ("wiki_full", "wiki_full_eval.jsonl"),
    ("man_page", "real_human_eval.jsonl"),
    ("proposal", "proposal_chunks.jsonl"),
]


def load_set(fname):
    path = os.path.join(DATA, fname)
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    bundle = joblib.load(os.path.join(MODELS, "feature_artifacts.joblib"))
    if bundle.get("lm") is not None:
        set_lm(bundle["lm"])
    model = joblib.load(os.path.join(MODELS, "nak_detector.joblib"))
    dim = model.n_features_in_
    print(f"model dim: {dim} (LM {'ON' if bundle.get('lm') else 'OFF'})",
          flush=True)

    print("\n== REAL-HUMAN FPR gate ==")
    all_fprs, worst05, worst07 = {}, 0.0, 0.0
    for name, fname in REAL_SETS:
        recs = load_set(fname)
        if not recs:
            print(f"  {name:10s} n=0 (missing, skipped)")
            continue
        probs = np.array([model.predict_proba([extract_features(r["text"])])[0, 1]
                          for r in recs])
        fpr05 = float((probs >= 0.5).mean())
        fpr07 = float((probs >= 0.7).mean())
        all_fprs[name] = {"n": len(recs), "fpr05": fpr05, "fpr07": fpr07,
                          "mean_p": float(probs.mean()),
                          "max_p": float(probs.max())}
        worst05 = max(worst05, fpr05)
        worst07 = max(worst07, fpr07)
        print(f"  {name:10s} n={len(recs):3d} FPR@0.5={fpr05:.4f} "
              f"FPR@0.7={fpr07:.4f} mean_p={probs.mean():.4f} "
              f"max_p={probs.max():.4f}")

    print("\n== synthetic test.jsonl (AI recall must not collapse) ==")
    recs = [json.loads(l) for l in open(os.path.join(DATA, "test.jsonl"))
            if l.strip()]
    probs = np.array([model.predict_proba([extract_features(r["text"])])[0, 1]
                      for r in recs])
    y = np.array([r["label"] for r in recs])
    pred = (probs >= 0.5).astype(int)
    acc = float((pred == y).mean())
    ai = y == 1
    recall = float((pred[ai] == 1).mean())
    fpr_syn = float((pred[~ai] == 1).mean())
    print(f"  acc={acc:.4f} AI_recall={recall:.4f} synth_FPR={fpr_syn:.4f}")

    prop = all_fprs.get("proposal", {})
    # Verdict at the APP's operating threshold (app/scan.py AI_THRESHOLD=0.7):
    # real-human FPR must be low single digits on every set, AI recall high,
    # and no formal academic chunk near 1.0.
    verdict = (worst07 <= 0.05 and recall >= 0.95
               and prop.get("max_p", 1.0) < 0.9)
    print(f"\n== GATE @0.7 (app threshold): "
          f"{'PASS ✔ SHIP IT' if verdict else 'FAIL ✘ DO NOT SHIP'} "
          f"(worst real FPR@0.7={worst07:.4f} [0.5: {worst05:.4f}], "
          f"AI recall={recall:.4f}, "
          f"proposal max_p={prop.get('max_p', float('nan')):.4f}) ==")
    json.dump({"real_fpr": all_fprs, "worst_real_fpr07": worst07,
               "worst_real_fpr05": worst05,
               "test_acc": acc, "ai_recall": recall,
               "verdict": "PASS" if verdict else "FAIL",
               "dim": dim},
              open(os.path.join(MODELS, "gate.json"), "w"), indent=2)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
