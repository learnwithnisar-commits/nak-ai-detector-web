#!/usr/bin/env python3
"""Build v3-stack: stacked generalization combining v2's absolute heads
(strong AI-side recall) with v3's pairwise ranker (strong human-side precision).

Meta-features per window: [p_v2_report, p_v2_prose, s_v3_rank].
Per-genre meta-logistic trained on absolute sentence-label windows +
anchors + background (same sets as v3 build).

Saves: models/nak_detector_v3_stack.joblib (+ _meta.json).
NEVER touches live model, v2, or v3 files.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from build_v2_model import (genre_label, doc_windows, window_features,
                            MANIFEST, SENT_LABELS, ANCH)
from src.v2_genre import GENRE_FEATURE_NAMES

# local copy of v3's featurizer (do NOT import build_v3_model: it has no
# __main__ guard and would re-run the full v3 build on import)
_v2b_local = joblib.load(os.path.join(ROOT, "models",
                                      "nak_detector_v2_turnitin.joblib"))
_new_scaler_local = _v2b_local["new_scaler"]


def _v3feats(windows):
    Xb, Xn = window_features(windows)
    return np.hstack([Xb, _new_scaler_local.transform(Xn)])

# load v2 + v3 bundles (read-only)
v2b = joblib.load(os.path.join(ROOT, "models", "nak_detector_v2_turnitin.joblib"))
v3b = joblib.load(os.path.join(ROOT, "models", "nak_detector_v3_pairwise.joblib"))
router, router_scaler = v2b["router"], v2b["router_scaler"]
new_scaler = v2b["new_scaler"]
head_report, head_prose = v2b["head_report"], v2b["head_prose"]
pair_w = np.asarray(v3b["pair_w"], dtype=np.float64)
print("loaded v2 heads + v3 ranker", flush=True)


def anchor_text(key):
    return open(os.path.join(ANCH, key + ".txt"), encoding="utf-8").read()


def meta_features(windows):
    """windows -> (n,3) [p_v2_report, p_v2_prose, s_v3]."""
    X = _v3feats(windows)
    pr = head_report.predict_proba(X)[:, 1]
    pe = head_prose.predict_proba(X)[:, 1]
    s = np.ravel(X @ pair_w)
    return np.column_stack([pr, pe, s])


# ---- rebuild the v3 calibration window sets (same construction) -------------
pdf_to_key = {v["pdf"]: k for k, v in MANIFEST.items() if v.get("pdf")}
by_pdf = {}
for s in SENT_LABELS:
    by_pdf.setdefault(s["pdf"], []).append(s)

sets = {"report": ([], []), "prose": ([], [])}  # genre -> (windows, labels)

def add(windows, labels, genre):
    sets[genre][0].extend(windows)
    sets[genre][1].extend(labels)

for pdf, ss in by_pdf.items():
    key = pdf_to_key.get(pdf)
    if key is None:
        continue
    try:
        g = genre_label(key, anchor_text(key))
    except Exception:
        continue
    txts = [x["sentence_text"] for x in ss]
    labs = [1 if x["label"] == "ai" else 0 for x in ss]
    for i in range(0, max(1, len(txts) - 7), 4):
        wlabs = labs[i:i + 8]
        if len(wlabs) < 6:
            continue
        frac = sum(wlabs) / len(wlabs)
        w = " ".join(txts[i:i + 8])
        if frac >= 0.75:
            add([w], [1], g)
        elif frac <= 0.25:
            add([w], [0], g)

for key, v in MANIFEST.items():
    if not v.get("train_include"):
        continue
    try:
        t = anchor_text(key)
    except Exception:
        continue
    g = genre_label(key, t)
    wins = doc_windows(t)
    if v["class"] == "human":
        add(wins, [0] * len(wins), g)
    elif v["class"] == "ai" and isinstance(v["tt_pct"], int) and v["tt_pct"] >= 60:
        add(wins, [1] * len(wins), g)

def load_jsonl(path, n, min_words=60):
    recs = []
    try:
        for l in open(path, encoding="utf-8"):
            if len(recs) >= n:
                break
            try:
                r = json.loads(l)
            except Exception:
                continue
            t = r.get("text", "")
            if len(t.split()) < min_words:
                continue
            recs.append(t)
    except FileNotFoundError:
        pass
    return recs

bg_ai, bg_hu = {"report": [], "prose": []}, {"report": [], "prose": []}
for p in ["data/chatgpt_technical_engineering.jsonl", "data/chatgpt_natural_sciences.jsonl"]:
    for t in load_jsonl(os.path.join(ROOT, p), 40):
        bg_ai[genre_label("x", t)].append(t)
for t in load_jsonl(os.path.join(ROOT, "data/arxiv_train.jsonl"), 600, 30):
    bg_hu["report"].append(t)
for t in load_jsonl(os.path.join(ROOT, "data/chatgpt_law_humanities.jsonl"), 40):
    bg_ai["prose"].append(t)
n_ph = 0
for l in open(os.path.join(ROOT, "data/train.jsonl"), encoding="utf-8"):
    if n_ph >= 800:
        break
    r = json.loads(l)
    if r.get("genre") in ("personal_essay", "blog_post", "short_story_opening",
                          "story_opening") and r.get("label") == 0:
        bg_hu["prose"].append(r["text"])
        n_ph += 1

def wins_of(docs, stride=8):
    out = []
    for t in docs:
        out.extend(doc_windows(t, stride=stride))
    return out

for g in ("report", "prose"):
    add(wins_of(bg_ai[g]), [1] * len(wins_of(bg_ai[g])), g)
    add(wins_of(bg_hu[g]), [0] * len(wins_of(bg_hu[g])), g)

# ---- train per-genre meta-logistic ------------------------------------------
metas = {}
for g in ("report", "prose"):
    wins, labs = sets[g]
    M = meta_features(wins)
    y = np.asarray(labs)
    sc = StandardScaler().fit(M)
    best = None
    from sklearn.model_selection import cross_val_score
    for C in (0.25, 0.5, 1.0, 2.0):
        clf = LogisticRegression(C=C, max_iter=3000)
        s = cross_val_score(clf, sc.transform(M), y, cv=3, scoring="f1").mean()
        if best is None or s > best[1]:
            best = (C, s)
    meta = LogisticRegression(C=best[0], max_iter=3000).fit(sc.transform(M), y)
    metas[g] = (meta, sc)
    print(f"meta {g}: n={len(y)}, pos={int(y.sum())}, C={best[0]}, "
          f"cv_f1={best[1]:.4f}, train_acc={meta.score(sc.transform(M), y):.4f}",
          flush=True)

bundle = {
    "head_report": head_report,
    "head_prose": head_prose,
    "pair_w": pair_w,
    "metas": {g: {"clf": m[0], "scaler": m[1]} for g, m in metas.items()},
    "router": router,
    "router_scaler": router_scaler,
    "new_scaler": new_scaler,
    "new_names": v2b["new_names"],
    "base_dim": 187,
    "seed": 20261016,
    "note": "v3_stack: per-genre meta-logistic on [p_v2_report, p_v2_prose, s_v3_rank]",
}
out = os.path.join(ROOT, "models", "nak_detector_v3_stack.joblib")
for protected in ("nak_detector.joblib", "nak_detector_v2_turnitin.joblib",
                  "nak_detector_v3_pairwise.joblib"):
    assert os.path.abspath(out) != os.path.abspath(
        os.path.join(ROOT, "models", protected)), "SAFETY"
joblib.dump(bundle, out)
json.dump({"seed": 20261016,
           "n": {g: len(sets[g][1]) for g in ("report", "prose")},
           "note": bundle["note"]},
          open(out.replace(".joblib", "_meta.json"), "w"), indent=1)
print("SAVED", out, flush=True)
