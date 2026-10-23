#!/usr/bin/env python3
"""Artifact diagnostics: figure out WHAT the model is keying on.

- single-feature AUC ranking on train
- class-conditional prevalence of tell markers
- ablation: retrain logistic regression without feature groups
- leave-one-genre-out generalization
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.features import _raw_features, feature_names  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score  # noqa: E402

DATA = os.path.join(ROOT, "data")


def load(path):
    texts, labels, genres = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            texts.append(o["text"])
            labels.append(int(o["label"]))
            genres.append(o["genre"])
    return texts, np.asarray(labels), np.asarray(genres)


def main():
    names = feature_names()
    tr_texts, y_tr, g_tr = load(os.path.join(DATA, "train.jsonl"))
    te_texts, y_te, g_te = load(os.path.join(DATA, "test.jsonl"))
    X_tr = np.stack([_raw_features(t) for t in tr_texts])
    X_te = np.stack([_raw_features(t) for t in te_texts])
    print(f"train {X_tr.shape} test {X_te.shape}")

    # ---- 1. single-feature AUC -------------------------------------------
    aucs = []
    for j, nm in enumerate(names):
        try:
            a = roc_auc_score(y_tr, X_tr[:, j])
        except ValueError:
            a = 0.5
        aucs.append((max(a, 1 - a), nm, j, a))
    aucs.sort(reverse=True)
    print("\n== top 25 single-feature AUCs (train) ==")
    for auc, nm, j, raw in aucs[:25]:
        mu1 = X_tr[y_tr == 1, j].mean()
        mu0 = X_tr[y_tr == 0, j].mean()
        print(f"  {auc:.4f}  {nm:28s} meanAI={mu1:9.3f} meanHU={mu0:9.3f}")

    # ---- 2. tell-marker prevalence ----------------------------------------
    print("\n== tell-group prevalence: % of samples with value > 0 ==")
    for grp in ["tell_formal_trans", "tell_contrast", "tell_hedge_academic",
                "tell_closer", "tell_cliche"]:
        j = names.index(grp)
        p1 = (X_te[y_te == 1, j] > 0).mean() * 100
        p0 = (X_te[y_te == 0, j] > 0).mean() * 100
        print(f"  {grp:22s} AI(test)={p1:6.2f}%  human(test)={p0:6.2f}%")

    # also: any-tell
    tell_js = [names.index(g) for g in
               ["tell_formal_trans", "tell_contrast", "tell_hedge_academic",
                "tell_closer", "tell_cliche"]]
    anytell_te = (X_te[:, tell_js] > 0).any(axis=1)
    print(f"  ANY tell: AI={(anytell_te[y_te==1]).mean()*100:.2f}% "
          f"human={(anytell_te[y_te==0]).mean()*100:.2f}%")

    # ---- 3. ablation --------------------------------------------------------
    groups = {
        "no_tells": [j for j, n in enumerate(names) if n.startswith("tell_")],
        "no_pronouns": [j for j, n in enumerate(names)
                        if n.startswith("pron_") or n in ("fw_i", "fw_we", "fw_you")],
        "no_punct": [j for j, n in enumerate(names) if n.startswith("punct_")],
        "no_funcwords": [j for j, n in enumerate(names) if n.startswith("fw_")],
        "no_structure": [j for j, n in enumerate(names)
                         if n.startswith(("bullet_", "heading_", "para_",
                                           "newline_", "colon_", "dialogue_",
                                           "quote_"))],
        "no_rhythm": [j for j, n in enumerate(names)
                      if n.startswith(("sent_", "ttr", "guiraud", "hapax",
                                       "dis_ratio", "yule", "word_entropy",
                                       "bigram_", "trigram_",
                                       "pct_words_repeated"))],
    }
    print("\n== ablation: logreg test F1/acc after dropping a feature group ==")
    keep_all = np.ones(X_tr.shape[1], dtype=bool)
    base = LogisticRegression(max_iter=3000, n_jobs=-1).fit(X_tr, y_tr)
    pb = base.predict(X_te)
    print(f"  {'ALL features':22s} F1={f1_score(y_te, pb):.4f} "
          f"acc={accuracy_score(y_te, pb):.4f}")
    for gname, js in groups.items():
        keep = keep_all.copy()
        keep[js] = False
        m = LogisticRegression(max_iter=3000, n_jobs=-1).fit(X_tr[:, keep], y_tr)
        p = m.predict(X_te[:, keep])
        print(f"  {gname:22s} dropped={len(js):3d} "
              f"F1={f1_score(y_te, p):.4f} acc={accuracy_score(y_te, p):.4f}")

    # ---- 4. leave-one-genre-out ----------------------------------------------
    print("\n== leave-one-genre-out (logreg, all features): test on unseen genre ==")
    genres = sorted(set(g_tr))
    for g in genres:
        tri = g_tr != g
        tei = g_te == g
        m = LogisticRegression(max_iter=3000, n_jobs=-1).fit(X_tr[tri], y_tr[tri])
        p = m.predict(X_te[tei])
        print(f"  held-out {g:18s} n={tei.sum():3d} "
              f"acc={accuracy_score(y_te[tei], p):.4f} "
              f"F1={f1_score(y_te[tei], p):.4f}")


if __name__ == "__main__":
    main()
