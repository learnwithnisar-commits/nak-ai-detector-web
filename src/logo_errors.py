#!/usr/bin/env python3
"""Inspect leave-one-genre-out errors: WHY do how_to / story_opening fail?"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.features import _raw_features, feature_names  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

DATA = os.path.join(ROOT, "data")


def load(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return recs


def main():
    names = feature_names()
    tr = load(os.path.join(DATA, "train.jsonl"))
    te = load(os.path.join(DATA, "test.jsonl"))
    X_tr = np.stack([_raw_features(r["text"]) for r in tr])
    y_tr = np.array([r["label"] for r in tr])
    g_tr = np.array([r["genre"] for r in tr])
    X_te = np.stack([_raw_features(r["text"]) for r in te])
    y_te = np.array([r["label"] for r in te])
    g_te = np.array([r["genre"] for r in te])

    for g in ["how_to", "story_opening"]:
        m = LogisticRegression(max_iter=3000, n_jobs=-1).fit(
            X_tr[g_tr != g], y_tr[g_tr != g])
        idx = np.where(g_te == g)[0]
        p = m.predict(X_te[idx])
        proba = m.predict_proba(X_te[idx])[:, 1]
        err = idx[p != y_te[idx]]
        print(f"\n===== held-out genre: {g}  n={len(idx)} "
              f"errors={len(err)} =====")
        # which direction are the errors?
        for e in err[:6]:
            r = te[e]
            print(f"--- true={'AI' if r['label']==1 else 'HUMAN'} "
                  f"p(AI)={proba[np.where(idx==e)[0][0]]:.3f} "
                  f"genre={r['genre']} style/persona={r.get('style', r.get('persona'))}")
            print("    " + r["text"][:280].replace("\n", " | ") + "...")
        # feature profile: mean of misclassified AI vs correctly classified AI
        ai_idx = idx[y_te[idx] == 1]
        ai_ok = ai_idx[p[y_te[idx] == 1] == 1]
        ai_bad = ai_idx[p[y_te[idx] == 1] == 0]
        hu_idx = idx[y_te[idx] == 0]
        hu_ok = hu_idx[p[y_te[idx] == 0] == 0]
        hu_bad = hu_idx[p[y_te[idx] == 0] == 1]
        print(f"  AI: ok={len(ai_ok)} bad={len(ai_bad)} | "
              f"HUMAN: ok={len(hu_ok)} bad={len(hu_bad)}")
        if len(ai_bad):
            print("  misclassified-AI feature means vs correctly-classified-AI:")
            for feat in ["sent_mean_w", "sent_burstiness", "pron_2nd",
                         "contractions_1k", "pct_short_sent8", "n_sents",
                         "tell_cliche", "bullet_line_ratio"]:
                j = names.index(feat)
                print(f"    {feat:20s} badAI={X_te[ai_bad][:, j].mean():8.3f} "
                      f"okAI={X_te[ai_ok][:, j].mean():8.3f} "
                      f"okHUMAN-test-dist={X_te[(g_te!=g)&(y_te==0)][:, j].mean():8.3f}")


if __name__ == "__main__":
    main()
