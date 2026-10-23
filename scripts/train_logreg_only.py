#!/usr/bin/env python3
"""Train ONLY logistic regression on the polish-augmented data (fast path).

Uses the already-built LM from models/feature_artifacts.joblib.
Saves candidate to /tmp/candidate_logreg/ (does NOT touch models/).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import joblib
import numpy as np

from src.features import (extract_features_unscaled, feature_names, set_lm,
                          set_lm_enabled)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATA = os.path.join(ROOT, "data")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/candidate_logreg"
CHUNK = 2500


def load_jsonl(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            texts.append(o["text"])
            labels.append(int(o["label"]))
    return texts, np.asarray(labels, dtype=int)


def main():
    set_lm_enabled(True)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    bundle = joblib.load(os.path.join(ROOT, "models",
                                      "feature_artifacts.joblib"))
    set_lm(bundle["lm"])
    print(f"LM loaded [{time.time()-t0:.1f}s]", flush=True)

    tr_texts, y_tr = load_jsonl(os.path.join(DATA, "train.jsonl"))
    for aug in ("real_human_train.jsonl", "wiki_full_train.jsonl",
                "pubmed_train.jsonl", "wiki_short_train.jsonl",
                "arxiv_train.jsonl", "europepmc_train.jsonl",
                "oa_sub_train.jsonl", "ai_polished_train.jsonl"):
        p = os.path.join(DATA, aug)
        if os.path.exists(p):
            t, y = load_jsonl(p)
            tr_texts += t
            y_tr = np.concatenate([y_tr, y])
    t, y = load_jsonl(os.path.join(DATA, "real_ai", "real_ai_train.jsonl"))
    tr_texts += t
    y_tr = np.concatenate([y_tr, y])
    print(f"train: {len(tr_texts)} positives={y_tr.sum()}", flush=True)

    n = len(tr_texts)
    Xs = []
    for i in range(0, n, CHUNK):
        s = slice(i, min(i + CHUNK, n))
        t0 = time.time()
        X = np.stack([extract_features_unscaled(t) for t in tr_texts[s]])
        Xs.append(X)
        print(f"  chunk {i//CHUNK+1} featurized {X.shape} "
              f"[{time.time()-t0:.1f}s]", flush=True)
    X_raw = np.vstack(Xs)
    scaler = StandardScaler().fit(X_raw)
    X_tr = scaler.transform(X_raw)

    t0 = time.time()
    clf = LogisticRegression(max_iter=3000, C=1.0,
                             class_weight="balanced")
    clf.fit(X_tr, y_tr)
    print(f"logreg fit [{time.time()-t0:.1f}s]", flush=True)

    joblib.dump(clf, os.path.join(OUT, "nak_detector.joblib"))
    joblib.dump({"scaler": scaler, "feature_names": feature_names(),
                 "lm": bundle["lm"], "winner": "logreg"},
                os.path.join(OUT, "feature_artifacts.joblib"))
    print("saved to", OUT, flush=True)


if __name__ == "__main__":
    main()
