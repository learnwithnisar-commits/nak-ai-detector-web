#!/usr/bin/env python3
"""Diagnose WHY real proposal chunks are flagged: per-feature contributions
(coef * scaled_value) averaged over flagged vs unflagged proposal chunks."""
import json
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.features import extract_features_unscaled, feature_names  # noqa: E402

DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")


def main():
    bundle = joblib.load(os.path.join(MODELS, "feature_artifacts.joblib"))
    scaler = bundle["scaler"]
    model = joblib.load(os.path.join(MODELS, "nak_detector.joblib"))
    names = feature_names()
    coef = model.coef_[0]
    print(f"dim={len(names)} intercept={model.intercept_[0]:.3f}")

    recs = [json.loads(l) for l in open(os.path.join(DATA, "proposal_chunks.jsonl"))
            if l.strip()]
    flagged, unflagged = [], []
    for r in recs:
        v = scaler.transform(
            extract_features_unscaled(r["text"]).reshape(1, -1))[0]
        p = float(1 / (1 + np.exp(-(v @ coef + model.intercept_[0]))))
        (flagged if p >= 0.5 else unflagged).append((r["text"][:60], p, v))

    print(f"flagged={len(flagged)} unflagged={len(unflagged)}")
    for tag, group in (("FLAGGED", flagged), ("OK", unflagged)):
        if not group:
            continue
        mean_contrib = np.mean([coef * v for _, _, v in group], axis=0)
        order = np.argsort(-mean_contrib)
        print(f"\n== {tag} chunks: top AI-driving features (mean coef*x) ==")
        for j in order[:12]:
            print(f"  {mean_contrib[j]:+7.3f}  {names[j]}")
        print(f"   ... intercept contributes {model.intercept_[0]:+.3f}")


if __name__ == "__main__":
    main()
