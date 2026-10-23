#!/usr/bin/env python3
"""Resumable retrain for NAK AI Detector (survives VM restarts).

- LM: rebuilt on ALL human texts (fast, ~2 min).
- Classifier: stratified 20k OA subsample + existing data (tractable, ~3h).
- Featurization checkpoints to models/_feats/chunk_*.npz; resumes on restart.
- Same model configs + class_weight='balanced' + held-out test pick as train.py.

Usage: .venv/bin/python scripts/resumable_train.py
"""
import argparse
import glob
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import joblib
import numpy as np

from src.features import (extract_features_unscaled, feature_names, set_lm,
                          set_lm_enabled)
from src.lm import build_lm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
FEATS = os.path.join(MODELS, "_feats")
CHUNK = 2500
OA_SUB_N = 20000
SEED = 20261012


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


def build_oa_subsample():
    """Stratified 20k subsample of oa_big_train.jsonl -> data/oa_sub_train.jsonl"""
    out_path = os.path.join(DATA, "oa_sub_train.jsonl")
    if os.path.exists(out_path):
        n = sum(1 for _ in open(out_path, encoding="utf-8"))
        print(f"oa_sub_train.jsonl exists ({n} rows), reusing")
        return
    rng = random.Random(SEED)
    by_src = {}
    for line in open(os.path.join(DATA, "oa_big_train.jsonl"), encoding="utf-8"):
        o = json.loads(line)
        by_src.setdefault(o["source"], []).append(line)
    total = sum(len(v) for v in by_src.values())
    with open(out_path, "w", encoding="utf-8") as f:
        for src, lines in sorted(by_src.items()):
            k = max(1, round(len(lines) / total * OA_SUB_N))
            for ln in rng.sample(lines, min(k, len(lines))):
                f.write(ln)
    n = sum(1 for _ in open(out_path, encoding="utf-8"))
    print(f"wrote oa_sub_train.jsonl ({n} rows, stratified)")


def main():
    set_lm_enabled(True)
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(FEATS, exist_ok=True)
    random.seed(SEED)

    build_oa_subsample()

    print("loading texts ...", flush=True)
    tr_texts, y_tr = load_jsonl(os.path.join(DATA, "train.jsonl"))
    # ai_polished_train.jsonl REMOVED (round 2026-10-12): the rule-based polish
    # simulator was a caricature (marker density 14.5/1k vs real ~1) and did
    # not transfer to real manuscripts. Replaced by REAL polish pairs below.
    for aug in ("real_human_train.jsonl", "wiki_full_train.jsonl",
                "pubmed_train.jsonl", "wiki_short_train.jsonl",
                "arxiv_train.jsonl", "europepmc_train.jsonl",
                "oa_sub_train.jsonl",
                "round_thesis_train.jsonl",     # real thesis prose, label 0
                "round_polish_train.jsonl",     # real ChatGPT polish delta, label 1 (x10)
                "round_nonprose_train.jsonl",   # non-prose negatives, label 0 (x5)
                "family_upsampled_train.jsonl"):  # NEW 2026-10-12: DeepSeek x25 (22 samples) + Claude x14 (40 samples) = 1110 AI-family samples, label 1 — upsampling fixes the 2026-10-12 regression where 40 unupsampled samples were drowned by 4000+ human academic texts (family_claude_train.jsonl superseded)
        p = os.path.join(DATA, aug)
        if os.path.exists(p):
            t, y = load_jsonl(p)
            tr_texts += t
            y_tr = np.concatenate([y_tr, y])
    t, y = load_jsonl(os.path.join(DATA, "real_ai", "real_ai_train.jsonl"))
    tr_texts += t
    y_tr = np.concatenate([y_tr, y])
    print(f"  train: {len(tr_texts)} positives={y_tr.sum()}", flush=True)

    print("building LM on stratified human subsample ...", flush=True)
    t0 = time.time()
    human_texts = [t for t, y in zip(tr_texts, y_tr) if y == 0]
    # LM subsample: 12k stratified docs (full 98k -> GB-sized trigram dict
    # that OOMs both training save and the free-tier deploy). 12k is still
    # 12x the original 1k-doc LM.
    rng_lm = random.Random(SEED + 1)
    if len(human_texts) > 12000:
        human_texts = rng_lm.sample(human_texts, 12000)
    # + wiki LM corpus
    wiki_lm = os.path.join(DATA, "wiki_lm.txt")
    if os.path.exists(wiki_lm):
        with open(wiki_lm, encoding="utf-8") as f:
            human_texts += [l.strip() for l in f if l.strip()]
    lm = build_lm(human_texts)
    # prune singleton trigrams/bigrams (hapax legomena add noise, not signal)
    tri_n0, bi_n0 = len(lm["tri"]), len(lm["bi"])
    lm["tri"] = {k: v for k, v in lm["tri"].items() if v >= 2}
    lm["bi"] = {k: v for k, v in lm["bi"].items() if v >= 2}
    print(f"  vocab={lm['V']} trigrams {tri_n0}->{len(lm['tri'])} "
          f"[{time.time()-t0:.1f}s]", flush=True)
    set_lm(lm)

    # ---- chunked featurization with resume ----
    n = len(tr_texts)
    n_chunks = (n + CHUNK - 1) // CHUNK
    dim = len(feature_names())
    for i in range(n_chunks):
        ckpt = os.path.join(FEATS, f"chunk_{i:03d}.npz")
        if os.path.exists(ckpt):
            print(f"  chunk {i+1}/{n_chunks} cached", flush=True)
            continue
        s = slice(i * CHUNK, min((i + 1) * CHUNK, n))
        t0 = time.time()
        X = np.stack([extract_features_unscaled(t) for t in tr_texts[s]])
        np.savez_compressed(ckpt, X=X, y=y_tr[s])
        print(f"  chunk {i+1}/{n_chunks} featurized {X.shape} "
              f"[{time.time()-t0:.1f}s]", flush=True)

    print("assembling ...", flush=True)
    Xs, ys = [], []
    for ckpt in sorted(glob.glob(os.path.join(FEATS, "chunk_*.npz"))):
        d = np.load(ckpt)
        Xs.append(d["X"])
        ys.append(d["y"])
    X_tr_raw = np.vstack(Xs)
    y_tr = np.concatenate(ys)
    print(f"  X={X_tr_raw.shape}", flush=True)

    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw)

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
    print("3-fold CV ...", flush=True)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=20261011)
    cv_f1 = {}
    for name, mdl in models.items():
        t0 = time.time()
        s = cross_val_score(mdl, X_tr, y_tr, cv=cv,
                            scoring=make_scorer(f1_score), n_jobs=1)
        cv_f1[name] = float(s.mean())
        print(f"  {name:10s} CV F1={s.mean():.4f} [{time.time()-t0:.1f}s]",
              flush=True)

    print("held-out test ...", flush=True)
    te_texts, y_te = load_jsonl(os.path.join(DATA, "test.jsonl"))
    X_te = scaler.transform(np.stack(
        [extract_features_unscaled(t) for t in te_texts]))
    test_f1, fitted = {}, {}
    for name, mdl in models.items():
        t0 = time.time()
        mdl.fit(X_tr, y_tr)
        fitted[name] = mdl
        test_f1[name] = float(f1_score(y_te, mdl.predict(X_te)))
        print(f"  {name:10s} TEST F1={test_f1[name]:.4f} [{time.time()-t0:.1f}s]",
              flush=True)
    winner = max(test_f1, key=test_f1.get)
    print(f"WINNER: {winner} (test F1={test_f1[winner]:.4f})", flush=True)

    # ATOMIC save: write temp then rename (restart-safe)
    tmp1 = os.path.join(MODELS, "_tmp_model.joblib")
    tmp2 = os.path.join(MODELS, "_tmp_artifacts.joblib")
    joblib.dump(fitted[winner], tmp1)
    os.rename(tmp1, os.path.join(MODELS, "nak_detector.joblib"))
    bundle = {"scaler": scaler, "feature_names": feature_names(),
              "winner": winner, "cv_f1": cv_f1, "test_f1": test_f1, "lm": lm}
    joblib.dump(bundle, tmp2)
    os.rename(tmp2, os.path.join(MODELS, "feature_artifacts.joblib"))
    print("saved (atomic): nak_detector.joblib + feature_artifacts.joblib",
          flush=True)

    # cleanup checkpoints
    for ckpt in glob.glob(os.path.join(FEATS, "chunk_*.npz")):
        os.remove(ckpt)
    print("checkpoints cleaned", flush=True)


if __name__ == "__main__":
    main()
