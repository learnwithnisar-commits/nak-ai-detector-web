#!/usr/bin/env python3
"""Build the SECOND NAK model: genre-aware (router + two register heads).

- Router: P(report) from 8 doc-level genre features.
- Head_report: AI-vs-human WITHIN student-report register (sentence labels +
  TT anchors + report-like background).
- Head_prose: AI-vs-human WITHIN prose register (essay anchors + prose background).
- Inference: p = w*p_report + (1-w)*p_prose, w = router(doc).

Saves: models/nak_detector_v2_turnitin.joblib (+ _meta.json).
NEVER touches models/nak_detector.joblib.
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.v2_genre import genre_feature_vector, GENRE_FEATURE_NAMES
from src.r4_features import extract_new_features, feature_names_new
from src.features import extract_features as base_features_187

# r4 kept 10/49 validated features - select by name
_ALL_NEW = feature_names_new()
_KEPT10 = ['r4_proper_cap_1k', 'r4_digit_1k', 'r4_date_1k', 'r4_nominal_1k',
           'r4_past_ed_1k', 'r4_sent_start_bland_ratio', 'r4_sent_len_gini',
           'r4_sent_len_range_cv', 'r4_clause_density', 'r4_repeat_trigram_1k']
_KEPT_IDX = [_ALL_NEW.index(n) for n in _KEPT10]
NEW_NAMES = _KEPT10


def new_features_10(text):
    _, arr = extract_new_features(text)
    return np.asarray([arr[i] for i in _KEPT_IDX], dtype=np.float64)

DATA = os.path.join(ROOT, "data")
ANCH = os.path.join(DATA, "anchors_oct16r2")
MANIFEST = json.load(open(os.path.join(ANCH, "anchors_oct16r2_manifest.json")))
SENT_LABELS = json.load(open(os.path.join(DATA, "sentence_labels_oct16r3.json")))
NEW_NAMES = _KEPT10
print("new feature names (kept 10):", NEW_NAMES, flush=True)

_SENT_SPLIT = re.compile(r"(?<=[.!?\u2026\u06d4\u061f])\s+")


def split_sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def anchor_text(key):
    p = os.path.join(ANCH, key + ".txt")
    return open(p, encoding="utf-8").read()


# ---------------------------------------------------------------- genre labels
PROSE_KW = re.compile(r"cover_?letter|reflect|essay|story|debate|transcript|engl100|a1_ing|competition|upload|research_overview|fifa|outline", re.I)
REPORT_KW = re.compile(r"report|manuscript|deliverable|_pd\d|assignment|project|experiment|thesis|proposal|synopsis|guideline|calculation|che320|nano101|com211|analysis_fin200|ratio_analysis", re.I)
# manual overrides for the 10 rule/keyword conflicts (verified by reading heads)
MANUAL = {
    "40290_report_1": "report",
    "71415_cover_letter_1": "prose",
    "75505_individual_reflections": "prose",
    "arab_design_reflective_writing_paper_1": "prose",
    "assignment2_fpga_plc_report_1": "report",
    "bus201_sm26_cv_coverletter_cover_page": "prose",
    "che320_assignment2_1_1": "report",
    "com211_written_and_visual_report_201945776_1": "report",
    "final_report_1": "report",
    "set_ass_2_final_report_scg_1": "report",
}


def genre_label(key, text):
    if key in MANUAL:
        return MANUAL[key]
    if PROSE_KW.search(key):
        return "prose"
    if REPORT_KW.search(key):
        return "report"
    fv = genre_feature_vector(text)
    return "report" if (fv[0] >= 0.12 or fv[3] >= 2.0 or fv[4] + fv[5] >= 1.5) else "prose"


# ---------------------------------------------------------------- feature cache
def window_features(windows):
    """windows: list of str. Returns (X187_scaled, Xnew10_unscaled)."""
    Xb = np.stack([base_features_187(w) for w in windows])
    Xn = np.stack([new_features_10(w) for w in windows])
    return Xb, Xn


def doc_windows(text, window=8, stride=4):
    sents = split_sentences(text)
    out = []
    for i in range(0, max(1, len(sents) - window + 1), stride):
        chunk = sents[i:i + window]
        if len(chunk) >= 4:
            out.append(" ".join(chunk))
    if not out and sents:
        out.append(" ".join(sents))
    return out


def main():
    rng = np.random.RandomState(20261016)
    # ---- 1. router ---------------------------------------------------------
    rX, ry, rkeys = [], [], []
    genre_of = {}
    for key, v in MANIFEST.items():
        if not v.get("train_include"):
            continue
        try:
            t = anchor_text(key)
        except Exception:
            continue
        g = genre_label(key, t)
        genre_of[key] = g
        rX.append(genre_feature_vector(t))
        ry.append(1 if g == "report" else 0)
        rkeys.append(key)
    rX = np.asarray(rX)
    ry = np.asarray(ry)
    print(f"router train: {len(ry)} docs, reports={ry.sum()}, prose={(1-ry).sum()}", flush=True)
    rscaler = StandardScaler().fit(rX)
    router = LogisticRegression(C=1.0, max_iter=2000).fit(rscaler.transform(rX), ry)
    print(f"router train acc: {router.score(rscaler.transform(rX), ry):.4f}", flush=True)
    gw = {k: float(router.predict_proba(rscaler.transform(
        np.asarray(genre_feature_vector(anchor_text(k))).reshape(1, -1)))[0][1])
        for k in rkeys}
    json.dump({k: {"genre": genre_of[k], "w_report": gw[k],
                   "class": MANIFEST[k]["class"], "tt_pct": MANIFEST[k]["tt_pct"]}
               for k in rkeys},
              open(os.path.join(DATA, "v2_genre_weights.json"), "w"), indent=1)
    print("saved data/v2_genre_weights.json", flush=True)

    # ---- 2. sentence-label windows (majority label per 8-sent window) -------
    by_pdf = {}
    for s in SENT_LABELS:
        by_pdf.setdefault(s["pdf"], []).append(s)
    rep_ai_wins, rep_hu_wins = [], []
    prose_ai_wins, prose_hu_wins = [], []
    for pdf, ss in by_pdf.items():
        key = None
        for k, v in MANIFEST.items():
            if v.get("pdf") == pdf:
                key = k
                break
        g = genre_of.get(key, "report")
        txts = [x["sentence_text"] for x in ss]
        labs = [1 if x["label"] == "ai" else 0 for x in ss]
        for i in range(0, max(1, len(txts) - 7), 4):
            wlabs = labs[i:i + 8]
            if len(wlabs) < 6:
                continue
            frac = sum(wlabs) / len(wlabs)
            if frac >= 0.75:
                lab = "ai"
            elif frac <= 0.25:
                lab = "hu"
            else:
                continue
            w = " ".join(txts[i:i + 8])
            if g == "report":
                (rep_ai_wins if lab == "ai" else rep_hu_wins).append(w)
            else:
                (prose_ai_wins if lab == "ai" else prose_hu_wins).append(w)
    print(f"sentence-label windows: rep AI={len(rep_ai_wins)} HU={len(rep_hu_wins)} | "
          f"prose AI={len(prose_ai_wins)} HU={len(prose_hu_wins)}", flush=True)

    # ---- 3. TT anchor windows ----------------------------------------------
    rep_ai_doc, rep_hu_doc, prose_ai_doc, prose_hu_doc = [], [], [], []
    for key in rkeys:
        v = MANIFEST[key]
        g = genre_of[key]
        wins = doc_windows(anchor_text(key))
        if v["class"] == "human":
            (rep_hu_doc if g == "report" else prose_hu_doc).extend(wins)
        elif v["class"] == "ai" and isinstance(v["tt_pct"], int) and v["tt_pct"] >= 60:
            (rep_ai_doc if g == "report" else prose_ai_doc).extend(wins)
    print(f"anchor doc windows: rep AI={len(rep_ai_doc)} HU={len(rep_hu_doc)} | "
          f"prose AI={len(prose_ai_doc)} HU={len(prose_hu_doc)}", flush=True)

    # ---- 4. background corpora, register-filtered ----------------------------
    def load_jsonl(path, n, min_words=60):
        recs = []
        try:
            for i, l in enumerate(open(path, encoding="utf-8")):
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

    # report-register background: synthetic technical AI + arXiv human
    # (arXiv abstracts are academic/structured -> report register by construction)
    rep_ai_bg, rep_hu_bg, prose_ai_bg, prose_hu_bg = [], [], [], []
    for p in ["data/chatgpt_technical_engineering.jsonl",
              "data/chatgpt_natural_sciences.jsonl"]:
        for t in load_jsonl(os.path.join(ROOT, p), 40):
            (rep_ai_bg if genre_label("x", t) == "report" else prose_ai_bg).append(t)
    rep_hu_bg = load_jsonl(os.path.join(ROOT, "data/arxiv_train.jsonl"), 600, 30)
    # prose-register AI background: law/humanities essays are prose by construction
    prose_ai_bg = load_jsonl(os.path.join(ROOT, "data/chatgpt_law_humanities.jsonl"), 40)
    # use train.jsonl genre tags directly for prose human background
    n_prose_hu = 0
    for l in open(os.path.join(ROOT, "data/train.jsonl"), encoding="utf-8"):
        if n_prose_hu >= 800:
            break
        r = json.loads(l)
        if r.get("genre") in ("personal_essay", "blog_post", "short_story_opening",
                              "story_opening") and r.get("label") == 0:
            prose_hu_bg.append(r["text"])
            n_prose_hu += 1
    print(f"background docs: rep AI={len(rep_ai_bg)} HU={len(rep_hu_bg)} | "
          f"prose AI={len(prose_ai_bg)} HU={len(prose_hu_bg)}", flush=True)

    def wins_of(docs, stride=8):
        out = []
        for t in docs:
            out.extend(doc_windows(t, stride=stride))
        return out

    # ---- assemble head training sets ----------------------------------------
    rep_pos = rep_ai_wins + rep_ai_doc + wins_of(rep_ai_bg)
    rep_neg = rep_hu_wins + rep_hu_doc + wins_of(rep_hu_bg)
    prose_pos = prose_ai_wins + prose_ai_doc + wins_of(prose_ai_bg)
    prose_neg = prose_hu_wins + prose_hu_doc + wins_of(prose_hu_bg)
    print(f"HEAD report: pos={len(rep_pos)} neg={len(rep_neg)}", flush=True)
    print(f"HEAD prose:  pos={len(prose_pos)} neg={len(prose_neg)}", flush=True)

    def featurize_all(windows):
        Xb, Xn = window_features(windows)
        return Xb, Xn

    # fit new-feature scaler on ALL training windows
    all_wins = rep_pos + rep_neg + prose_pos + prose_neg
    _, Xn_all = featurize_all(all_wins)
    new_scaler = StandardScaler().fit(Xn_all)
    print("new-feature scaler fit on", len(all_wins), "windows", flush=True)

    def build(Xwins_pos, Xwins_neg):
        Xb_p, Xn_p = featurize_all(Xwins_pos)
        Xb_n, Xn_n = featurize_all(Xwins_neg)
        Xp = np.hstack([Xb_p, new_scaler.transform(Xn_p)])
        Xn_ = np.hstack([Xb_n, new_scaler.transform(Xn_n)])
        X = np.vstack([Xp, Xn_])
        y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn_))])
        return X, y

    Xr, yr = build(rep_pos, rep_neg)
    Xp_, yp = build(prose_pos, prose_neg)

    # tune C per head by 3-fold CV
    from sklearn.model_selection import cross_val_score
    best = {}
    for name, X, y in [("report", Xr, yr), ("prose", Xp_, yp)]:
        scores = {}
        for C in (0.25, 0.5, 1.0):
            clf = LogisticRegression(C=C, max_iter=3000)
            s = cross_val_score(clf, X, y, cv=3, scoring="f1").mean()
            scores[C] = s
        Cbest = max(scores, key=scores.get)
        best[name] = (Cbest, scores[Cbest])
        print(f"head {name}: CV f1 {scores} -> C={Cbest}", flush=True)

    head_report = LogisticRegression(C=best["report"][0], max_iter=3000).fit(Xr, yr)
    head_prose = LogisticRegression(C=best["prose"][0], max_iter=3000).fit(Xp_, yp)
    print(f"train acc: report={head_report.score(Xr, yr):.4f} prose={head_prose.score(Xp_, yp):.4f}", flush=True)

    bundle = {
        "router": router,
        "router_scaler": rscaler,
        "head_report": head_report,
        "head_prose": head_prose,
        "new_scaler": new_scaler,
        "new_names": NEW_NAMES,
        "genre_names": GENRE_FEATURE_NAMES,
        "base_dim": 187,
        "seed": 20261016,
        "note": "v2_turnitin: genre router + two register heads; trained ONLY on Turnitin-anchored data + sentence labels + register-filtered background",
    }
    out = os.path.join(ROOT, "models", "nak_detector_v2_turnitin.joblib")
    joblib.dump(bundle, out)
    meta = {
        "seed": 20261016,
        "router_train_n": len(ry),
        "head_report_n": [int((yr == 1).sum()), int((yr == 0).sum())],
        "head_prose_n": [int((yp == 1).sum()), int((yp == 0).sum())],
        "best_C": {k: v[0] for k, v in best.items()},
        "cv_f1": {k: round(float(v[1]), 4) for k, v in best.items()},
        "train_acc_report": round(float(head_report.score(Xr, yr)), 4),
        "train_acc_prose": round(float(head_prose.score(Xp_, yp)), 4),
        "note": bundle["note"],
    }
    json.dump(meta, open(out.replace(".joblib", "_meta.json"), "w"), indent=1)
    print("SAVED", out, flush=True)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
