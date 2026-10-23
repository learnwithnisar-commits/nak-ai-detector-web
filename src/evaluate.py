#!/usr/bin/env python3
"""Final evaluation of the shipped model (models/nak_detector.joblib).

- Held-out test.jsonl metrics @0.5: acc/prec/rec/F1/FPR/confusion, per-genre.
- LOGO per-genre for the final config (honest generalization estimate).
- Threshold tuning on train out-of-fold probabilities (F1 vs FPR tradeoff).
- Writes models/final_metrics.json
"""
import json
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.features import extract_features, feature_names, set_lm  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)  # noqa: E402
from sklearn.model_selection import cross_val_predict, StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
CACHE = os.path.join(MODELS, "_feat_cache.npz")


def load_raw():
    import joblib as _joblib
    bundle = _joblib.load(os.path.join(MODELS, "feature_artifacts.joblib"))
    if bundle.get("lm") is not None:
        set_lm(bundle["lm"])
    from src.features import extract_features_unscaled  # noqa: E402

    def _load(path):
        recs = [json.loads(l) for l in open(path, encoding="utf-8")
                if l.strip()]
        X = np.stack([extract_features_unscaled(r["text"]) for r in recs])
        y = np.array([r["label"] for r in recs])
        g = np.array([r["genre"] for r in recs])
        return X, y, g

    X_tr, y_tr, g_tr = _load(os.path.join(DATA, "train.jsonl"))
    X_te, y_te, g_te = _load(os.path.join(DATA, "test.jsonl"))
    print(f"feature matrices: train {X_tr.shape}, test {X_te.shape}", flush=True)
    return (X_tr, y_tr, g_tr, X_te, y_te, g_te,
            bundle["feature_names"])


def prf(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
    }


def main():
    X_tr, y_tr, g_tr, X_te, y_te, g_te, names = load_raw()

    # ---- final model, via the public contract (fresh-feel: scaler+model files)
    bundle = joblib.load(os.path.join(MODELS, "feature_artifacts.joblib"))
    scaler = bundle["scaler"]
    model = joblib.load(os.path.join(MODELS, "nak_detector.joblib"))
    assert bundle["feature_names"] == names, "feature schema drift!"

    # sanity: extract_features applies the same scaler the model was trained with
    from src.features import extract_features_unscaled  # noqa: E402
    sample = json.loads(open(os.path.join(DATA, "test.jsonl")).readline())["text"]
    v_api = extract_features(sample)
    v_manual = scaler.transform(
        extract_features_unscaled(sample).reshape(1, -1)).ravel()
    print("api-vs-manual max abs diff:",
          float(np.max(np.abs(v_api - v_manual))))
    assert np.allclose(v_api, v_manual), "scaler mismatch between API and training!"

    proba = model.predict_proba(scaler.transform(X_te))[:, 1]
    pred05 = (proba >= 0.5).astype(int)
    m = prf(y_te, pred05)
    tn, fp, fn, tp = confusion_matrix(y_te, pred05).ravel()
    m["fpr"] = float(fp / (fp + tn))
    m["fnr"] = float(fn / (fn + tp))
    m["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    print("\n== held-out test @0.5 ==")
    for k, v in m.items():
        print(f"  {k}: {v}")

    # per-genre in-distribution
    per_genre = {}
    print("\n== per-genre test accuracy (in-distribution) ==")
    for g in sorted(set(g_te)):
        i = g_te == g
        a = float(accuracy_score(y_te[i], pred05[i]))
        per_genre[g] = {"n": int(i.sum()), "acc": a}
        print(f"  {g:20s} n={i.sum():3d} acc={a:.4f}")

    # ---- LOGO for the final config (logreg C=1.0, all features)
    print("\n== LOGO (final config) ==")
    logo_rows, logo_accs = {}, []
    for g in sorted(set(g_te)):
        tri, tei = g_tr != g, g_te == g
        sc = StandardScaler().fit(X_tr[tri])
        mdl = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1)
        mdl.fit(sc.transform(X_tr[tri]), y_tr[tri])
        p = mdl.predict(sc.transform(X_te[tei]))
        a = float(accuracy_score(y_te[tei], p))
        logo_rows[g] = {"n": int(tei.sum()), "acc": a}
        logo_accs.append(a)
        print(f"  held-out {g:20s} n={tei.sum():3d} acc={a:.4f}")
    logo_macro = float(np.mean(logo_accs))
    print(f"  LOGO macro-avg accuracy: {logo_macro:.4f}")

    # ---- threshold tuning on train OOF probabilities
    print("\n== threshold tuning (train 3-fold OOF) ==")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=20261011)
    oof = cross_val_predict(LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1),
                            scaler.transform(X_tr), y_tr, cv=cv,
                            method="predict_proba")[:, 1]
    rows = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        p = (oof >= t).astype(int)
        d = prf(y_tr, p)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_tr, p).ravel()
        d.update(threshold=t, fpr=float(fp_ / (fp_ + tn_)))
        rows.append(d)
        print(f"  thr={t:.1f} F1={d['f1']:.4f} FPR={d['fpr']:.4f} "
              f"acc={d['accuracy']:.4f}")
    feas = [r for r in rows if r["fpr"] <= 0.02]
    best = max(feas, key=lambda r: r["f1"]) if feas else max(rows, key=lambda r: r["f1"])
    print(f"  recommended (max F1 s.t. OOF FPR<=2%): thr={best['threshold']} "
          f"F1={best['f1']:.4f} FPR={best['fpr']:.4f}")
    # apply recommended + app default 0.7 to test for reference
    for t, tag in [(best["threshold"], "recommended"), (0.7, "app default")]:
        p = (proba >= t).astype(int)
        d = prf(y_te, p)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_te, p).ravel()
        print(f"  test @{tag}({t}): F1={d['f1']:.4f} "
              f"FPR={fp_/(fp_+tn_):.4f} acc={d['accuracy']:.4f}")

    out = {
        "test_at_0.5": m,
        "per_genre_test": per_genre,
        "logo_per_genre": logo_rows,
        "logo_macro_acc": logo_macro,
        "threshold_sweep_oof": rows,
        "recommended_threshold": best["threshold"],
        "n_features": len(names),
        "winner": bundle.get("winner"),
    }
    with open(os.path.join(MODELS, "final_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved models/final_metrics.json")


if __name__ == "__main__":
    main()
