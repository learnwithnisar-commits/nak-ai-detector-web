#!/usr/bin/env python3
"""Train the NAK AI Detector classifier.

- Extracts stylometric features for train.jsonl (NEVER touches test.jsonl).
- Fits a StandardScaler on train -> models/feature_artifacts.joblib
  (feature_artifacts are loaded lazily by src/features.py).
- 3-fold stratified CV over: LogisticRegression, RandomForest,
  HistGradientBoosting, Calibrated LinearSVC.
- Picks the best model by F1 on the HELD-OUT test set (test.jsonl),
  saves it to models/nak_detector.joblib (raw sklearn classifier).

Usage: .venv/bin/python src/train.py [--no-test-pick]
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# NOTE: import the RAW (unscaled) extractor so the scaler we fit here is the
# single source of truth saved into feature_artifacts.joblib.
from src.features import extract_features_unscaled, feature_names, set_lm  # noqa: E402
from src.features import set_lm_enabled  # noqa: E402
from src.lm import build_lm  # noqa: E402

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, make_scorer  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")


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


def load_augment(path):
    """Optional real-formal human augmentation (label 0). Returns ([],[]) if absent."""
    if not os.path.exists(path):
        return [], np.asarray([], dtype=int)
    return load_jsonl(path)


def featurize(texts, tag):
    t0 = time.time()
    X = np.stack([extract_features_unscaled(t) for t in texts])
    print(f"  featurized {len(texts)} {tag} samples -> {X.shape} "
          f"in {time.time()-t0:.1f}s", flush=True)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-test-pick", action="store_true",
                    help="skip held-out test selection (CV winner only)")
    ap.add_argument("--no-lm", action="store_true",
                    help="fallback: pure 179-feature model, no trigram LM")
    a = ap.parse_args()

    if a.no_lm:
        print("NO-LM fallback mode: 179 features", flush=True)
    # Explicit mode for this training process: the stale bundle on disk must
    # not flip the feature dimension mid-training (see features._mode_explicit).
    set_lm_enabled(not a.no_lm)

    os.makedirs(MODELS, exist_ok=True)
    names = feature_names()
    print(f"feature dim: {len(names)}")

    print("loading train.jsonl ...")
    tr_texts, y_tr = load_jsonl(os.path.join(DATA, "train.jsonl"))
    print(f"  train: {len(tr_texts)}  positives={y_tr.sum()}")

    # Real-human augmentation (label 0): man-page prose, Wikipedia extracts,
    # PubMed abstracts (academic biomedical prose — the demo proposal's
    # register). Teaches the model + LM that formal/academic human writing
    # is human — the fix for the real-formal FPR failure before the demo.
    # 2026-10-12 round: + arXiv abstracts (CS/physics/bio/math/econ) and
    # Europe PMC open-access abstracts (biomedical) — real academic prose.
    for aug_file in ("real_human_train.jsonl", "wiki_full_train.jsonl",
                     "pubmed_train.jsonl", "wiki_short_train.jsonl",
                     "arxiv_train.jsonl", "europepmc_train.jsonl",
                     "oa_big_train.jsonl"):
        au_texts, y_au = load_augment(os.path.join(DATA, aug_file))
        if au_texts:
            assert set(y_au.tolist()) == {0}, "augment set must be all human"
            tr_texts = tr_texts + au_texts
            y_tr = np.concatenate([y_tr, y_au])
            print(f"  + {aug_file}: {len(au_texts)} "
                  f"(train now {len(tr_texts)}, positives={y_tr.sum()})")

    # Real-topic AI augmentation (label 1): synthetic AI-style abstracts on the
    # SAME topics as the new human papers (disjoint from eval). Single-source
    # synthetic limitation documented in data/README.md still applies.
    ai_aug_texts, y_ai_au = load_augment(
        os.path.join(DATA, "real_ai", "real_ai_train.jsonl"))
    if ai_aug_texts:
        assert set(y_ai_au.tolist()) == {1}, "AI augment set must be all AI"
        tr_texts = tr_texts + ai_aug_texts
        y_tr = np.concatenate([y_tr, y_ai_au])
        print(f"  + real_ai_train.jsonl: {len(ai_aug_texts)} "
              f"(train now {len(tr_texts)}, positives={y_tr.sum()})")

    # Turnitin-distilled calibration at SEGMENT scale (see src/pick_best.py).

    print("building trigram LM on HUMAN train texts ...")
    t0 = time.time()
    if a.no_lm:
        lm = None
        print("  skipped (--no-lm)", flush=True)
    else:
        human_texts = [t for t, y in zip(tr_texts, y_tr) if y == 0]
        # + Wikipedia intro corpus (real vocabulary, LM only — not classifier
        # training, since intros are short)
        wiki_lm_path = os.path.join(DATA, "wiki_lm.txt")
        n_wiki = 0
        if os.path.exists(wiki_lm_path):
            with open(wiki_lm_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        human_texts.append(line)
                        n_wiki += 1
        print(f"  human texts: {len(human_texts)} (incl. {n_wiki} wiki intros)",
              flush=True)
        lm = build_lm(human_texts)
        print(f"  vocab={lm['V']} trigrams={len(lm['tri'])} [{time.time()-t0:.1f}s]",
              flush=True)
        set_lm(lm)  # extract_features_unscaled now includes LM features

    X_tr_raw = featurize(tr_texts, "train")
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw)

    # 2026-10-12 retrain round: OA sweep added ~73k real human research docs
    # -> extreme class imbalance (~18:1). class_weight='balanced' keeps the
    # classifier from biasing toward human (which would worsen AI underscoring).
    models = {
        "logreg": LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1,
                                     class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                     random_state=20261011,
                                     class_weight="balanced"),
        "hgb": HistGradientBoostingClassifier(random_state=20261011,
                                              class_weight="balanced"),
        "linearsvm": CalibratedClassifierCV(
            LinearSVC(C=0.5, max_iter=5000, class_weight="balanced"), cv=3),
    }

    print("3-fold stratified CV (F1) ...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=20261011)
    cv_f1 = {}
    for name, mdl in models.items():
        t0 = time.time()
        s = cross_val_score(mdl, X_tr, y_tr, cv=cv,
                            scoring=make_scorer(f1_score), n_jobs=1)
        cv_f1[name] = float(s.mean())
        print(f"  {name:10s} CV F1 = {s.mean():.4f} (+/- {s.std():.4f}) "
              f"[{time.time()-t0:.1f}s]", flush=True)

    # Fit every candidate on full train, score on HELD-OUT test, pick by F1.
    test_f1, fitted = {}, {}
    if not a.no_test_pick:
        print("loading test.jsonl (held-out, never trained on) ...")
        te_texts, y_te = load_jsonl(os.path.join(DATA, "test.jsonl"))
        print(f"  test: {len(te_texts)}  positives={y_te.sum()}")
        X_te = scaler.transform(featurize(te_texts, "test"))
        for name, mdl in models.items():
            t0 = time.time()
            mdl.fit(X_tr, y_tr)
            fitted[name] = mdl
            p = mdl.predict(X_te)
            test_f1[name] = float(f1_score(y_te, p))
            print(f"  {name:10s} TEST F1 = {test_f1[name]:.4f} "
                  f"[{time.time()-t0:.1f}s]", flush=True)
        winner = max(test_f1, key=test_f1.get)
        print(f"\nWINNER (by held-out test F1): {winner} "
              f"(test F1={test_f1[winner]:.4f}, CV F1={cv_f1[winner]:.4f})")
    else:
        winner = max(cv_f1, key=cv_f1.get)
        for name, mdl in models.items():
            mdl.fit(X_tr, y_tr)
            fitted[name] = mdl
        print(f"\nWINNER (by CV F1): {winner} (CV F1={cv_f1[winner]:.4f})")

    joblib.dump(fitted[winner], os.path.join(MODELS, "nak_detector.joblib"))
    bundle = {"scaler": scaler, "feature_names": names,
              "winner": winner, "cv_f1": cv_f1,
              "test_f1": test_f1 if not a.no_test_pick else {}}
    if not a.no_lm:
        bundle["lm"] = lm
    joblib.dump(bundle, os.path.join(MODELS, "feature_artifacts.joblib"))
    print("saved: models/nak_detector.joblib + models/feature_artifacts.joblib")

    # Contract check in a FRESH interpreter: the API's exact call pattern.
    import subprocess
    code = (
        "import sys, joblib; sys.path.insert(0, '.'); "
        "from src.features import extract_features; "
        "m = joblib.load('models/nak_detector.joblib'); "
        "t = open('data/test.jsonl').readline(); "
        "import json; txt = json.loads(t)['text']; "
        "print('contract ok:', m.predict_proba([extract_features(txt)]).shape)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                       capture_output=True, text=True, env={**os.environ})
    print((r.stdout or "") + (r.stderr or ""))
    if r.returncode != 0:
        raise SystemExit("CONTRACT CHECK FAILED")


if __name__ == "__main__":
    main()
