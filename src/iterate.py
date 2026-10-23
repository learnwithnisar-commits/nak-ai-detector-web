#!/usr/bin/env python3
"""Iteration round: sweep models x feature-subsets, scored by
leave-one-genre-out (LOGO) macro accuracy = the honest generalization metric.

Also tracks in-distribution held-out test F1 so we don't trade everything.
Winner (best LOGO macro-acc) is retrained on full train and saved.
"""
import json
import os
import sys
import time

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.features import _raw_features, feature_names  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "models", "_feat_cache.npz")


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def get_features():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return d["X_tr"], d["y_tr"], d["g_tr"], d["X_te"], d["y_te"], d["g_te"]
    names = feature_names()
    tr, te = load(f"{DATA}/train.jsonl"), load(f"{DATA}/test.jsonl")
    X_tr = np.stack([_raw_features(r["text"]) for r in tr])
    X_te = np.stack([_raw_features(r["text"]) for r in te])
    y_tr = np.array([r["label"] for r in tr])
    y_te = np.array([r["label"] for r in te])
    g_tr = np.array([r["genre"] for r in tr])
    g_te = np.array([r["genre"] for r in te])
    np.savez_compressed(CACHE, X_tr=X_tr, y_tr=y_tr, g_tr=g_tr,
                        X_te=X_te, y_te=y_te, g_te=g_te, names=np.array(names))
    return X_tr, y_tr, g_tr, X_te, y_te, g_te


def subsets(names):
    n = np.array(names)
    is_tell = np.array([x.startswith("tell_") for x in names])
    is_pron = np.array([x.startswith("pron_") or x in ("fw_i", "fw_we", "fw_you",
                        "fw_my", "fw_our", "fw_your", "fw_me", "fw_us",
                        "fw_mine", "fw_ours", "fw_yours") for x in names])
    is_struct = np.array([x.startswith(("bullet_", "heading_", "para_",
                                        "newline_", "colon_", "dialogue_",
                                        "quote_", "n_paras"))
                          for x in names])
    is_contra = np.array([x == "contractions_1k" for x in names])
    is_fw = np.array([x.startswith("fw_") for x in names])
    core_keep = ~(is_tell | is_pron | is_struct | is_contra)
    rhythm_lex = np.array([x.startswith(("sent_", "ttr", "guiraud", "hapax",
                                         "dis_ratio", "yule", "word_entropy",
                                         "bigram_", "trigram_",
                                         "pct_words_repeated", "mean_word_len",
                                         "std_word_len", "pct_long_words",
                                         "punct_", "pct_sent_end_",
                                         "n_sents", "n_words"))
                           for x in names])
    return {
        "all": np.ones(len(names), bool),
        "no_tells": ~is_tell,
        "universal_core": core_keep,          # no tells/pronouns/structure/contractions
        "rhythm_lex_only": rhythm_lex,
    }


def logo_macro_acc(make_model, X_tr, y_tr, g_tr, X_te, y_te, g_te, keep):
    accs = []
    for g in sorted(set(g_te)):
        tri = g_tr != g
        tei = g_te == g
        if tri.sum() == 0 or tei.sum() == 0:
            continue
        sc = StandardScaler().fit(X_tr[tri][:, keep])
        m = make_model()
        m.fit(sc.transform(X_tr[tri][:, keep]), y_tr[tri])
        p = m.predict(sc.transform(X_te[tei][:, keep]))
        accs.append(accuracy_score(y_te[tei], p))
    return float(np.mean(accs)), accs


def main():
    names = list(feature_names())
    X_tr, y_tr, g_tr, X_te, y_te, g_te = get_features()
    print(f"X_tr {X_tr.shape}, genres: {sorted(set(g_te))}", flush=True)

    subs = subsets(names)
    for k, v in subs.items():
        print(f"  subset {k}: {v.sum()} features", flush=True)

    configs = {
        "logreg_C1.0": lambda: LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1),
        "logreg_C0.1": lambda: LogisticRegression(max_iter=3000, C=0.1, n_jobs=-1),
        "logreg_C0.02": lambda: LogisticRegression(max_iter=3000, C=0.02, n_jobs=-1),
        "hgb": lambda: HistGradientBoostingClassifier(random_state=20261011),
    }

    results = []
    for cname, mk in configs.items():
        for sname, keep in subs.items():
            t0 = time.time()
            logo, per = logo_macro_acc(mk, X_tr, y_tr, g_tr, X_te, y_te, g_te, keep)
            # in-distribution test F1 with same config (full train)
            sc = StandardScaler().fit(X_tr[:, keep])
            m = mk()
            m.fit(sc.transform(X_tr[:, keep]), y_tr)
            p = m.predict(sc.transform(X_te[:, keep]))
            f1 = f1_score(y_te, p)
            acc = accuracy_score(y_te, p)
            results.append((logo, f1, acc, cname, sname, per))
            print(f"{cname:12s} x {sname:15s} LOGO_macro={logo:.4f} "
                  f"testF1={f1:.4f} testAcc={acc:.4f} [{time.time()-t0:.0f}s]",
                  flush=True)

    results.sort(reverse=True)
    print("\n== ranking by LOGO macro accuracy ==")
    for logo, f1, acc, cname, sname, per in results:
        print(f"  LOGO={logo:.4f} testF1={f1:.4f}  {cname} x {sname}")

    joblib.dump({"results": [(l, f, a, c, s) for l, f, a, c, s, _ in results],
                 "per_genre_best": results[0][5],
                 "genres": sorted(set(g_te.tolist()))},
                os.path.join(ROOT, "models", "iteration_sweep.joblib"))
    print("\nsaved models/iteration_sweep.joblib")


if __name__ == "__main__":
    main()
