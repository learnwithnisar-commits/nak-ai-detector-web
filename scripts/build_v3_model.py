#!/usr/bin/env python3
"""Build the THIRD NAK model: pairwise/contrastive within-register training.

Idea: the verified Turnitin sentence labels (893 AI + 2263 human sentences,
many AI/human pairs from the SAME documents) let us train a RANKING objective:
  f(x_AI) > f(x_human)  for windows from the same document.
Training on difference vectors d = x_AI - x_human with a linear RankNet
(logistic on differences, no intercept) cancels the doc-level register
component that fooled all absolute classifiers (r1..r4, v2).

Then per-genre Platt calibration maps raw ranking scores to P(AI), reusing
v2's genre router + blend structure at inference:
  p(window) = w * sigmoid(a_r*s + b_r) + (1-w) * sigmoid(a_p*s + b_p)

Saves: models/nak_detector_v3_pairwise.joblib (+ _meta.json).
NEVER touches models/nak_detector.joblib or the v2 file.
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
from sklearn.model_selection import cross_val_score

from build_v2_model import (genre_label, doc_windows, window_features,
                            new_features_10, MANIFEST, SENT_LABELS, ANCH)
from src.v2_genre import GENRE_FEATURE_NAMES


def main():

    rng = np.random.RandomState(20261016)

    # ---- reuse v2's router + new-feature scaler (proven pieces) -----------------
    v2b = joblib.load(os.path.join(ROOT, "models", "nak_detector_v2_turnitin.joblib"))
    router, router_scaler, new_scaler = v2b["router"], v2b["router_scaler"], v2b["new_scaler"]
    print("reused v2 router + new_scaler", flush=True)


    def anchor_text(key):
        return open(os.path.join(ANCH, key + ".txt"), encoding="utf-8").read()


    def feats(windows):
        Xb, Xn = window_features(windows)
        return np.hstack([Xb, new_scaler.transform(Xn)])


    # ---- 1. sentence-label windows per pdf, per genre ---------------------------
    # window label: frac>=0.75 AI, frac<=0.25 human (same as v2)
    win_by_pdf = {}   # pdf -> {"genre": g, "ai": [w...], "hu": [w...]}
    pdf_to_key = {}
    for k, v in MANIFEST.items():
        if v.get("pdf"):
            pdf_to_key[v["pdf"]] = k
    by_pdf = {}
    for s in SENT_LABELS:
        by_pdf.setdefault(s["pdf"], []).append(s)
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
        ai_w, hu_w = [], []
        for i in range(0, max(1, len(txts) - 7), 4):
            wlabs = labs[i:i + 8]
            if len(wlabs) < 6:
                continue
            frac = sum(wlabs) / len(wlabs)
            w = " ".join(txts[i:i + 8])
            if frac >= 0.75:
                ai_w.append(w)
            elif frac <= 0.25:
                hu_w.append(w)
        if ai_w and hu_w:
            win_by_pdf[pdf] = {"genre": g, "ai": ai_w, "hu": hu_w}
    n_ai = sum(len(v["ai"]) for v in win_by_pdf.values())
    n_hu = sum(len(v["hu"]) for v in win_by_pdf.values())
    print(f"mixed-doc windows: pdfs={len(win_by_pdf)} AI={n_ai} HU={n_hu}", flush=True)

    # ---- 2. anchor doc windows (human=0; pure-AI tt>=60 -> 1) -------------------
    anch_wins = {"report": {"ai": [], "hu": []}, "prose": {"ai": [], "hu": []}}
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
            anch_wins[g]["hu"].extend(wins)
        elif v["class"] == "ai" and isinstance(v["tt_pct"], int) and v["tt_pct"] >= 60:
            anch_wins[g]["ai"].extend(wins)
    print("anchor windows:", {g: {k: len(v) for k, v in d.items()}
                               for g, d in anch_wins.items()}, flush=True)

    # ---- 3. background windows (register-filtered, same as v2) ------------------
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

    bg = {"report": {"ai": [], "hu": []}, "prose": {"ai": [], "hu": []}}
    for p in ["data/chatgpt_technical_engineering.jsonl", "data/chatgpt_natural_sciences.jsonl"]:
        for t in load_jsonl(os.path.join(ROOT, p), 40):
            bg[genre_label("x", t)]["ai"].append(t)
    for t in load_jsonl(os.path.join(ROOT, "data/arxiv_train.jsonl"), 600, 30):
        bg["report"]["hu"].append(t)
    for t in load_jsonl(os.path.join(ROOT, "data/chatgpt_law_humanities.jsonl"), 40):
        bg["prose"]["ai"].append(t)
    n_ph = 0
    n_pe_ai = 0
    use_essay_bg = os.environ.get("V3_ESSAY_BG", "") == "1"
    for l in open(os.path.join(ROOT, "data/train.jsonl"), encoding="utf-8"):
        if n_ph >= 800 and (not use_essay_bg or n_pe_ai >= 400):
            break
        r = json.loads(l)
        if r.get("genre") in ("personal_essay", "blog_post", "short_story_opening",
                              "story_opening") and r.get("label") == 0 and n_ph < 800:
            bg["prose"]["hu"].append(r["text"])
            n_ph += 1
        elif use_essay_bg and r.get("genre") in ("personal_essay", "blog_post") \
                and r.get("label") == 1 and n_pe_ai < 400:
            bg["prose"]["ai"].append(r["text"])
            n_pe_ai += 1
    print(f"prose bg: AI={len(bg['prose']['ai'])} (incl {n_pe_ai} train.jsonl essays), "
          f"HU={len(bg['prose']['hu'])}", flush=True)

    def wins_of(docs, stride=8):
        out = []
        for t in docs:
            out.extend(doc_windows(t, stride=stride))
        return out

    bg_wins = {g: {k: wins_of(v) for k, v in d.items()} for g, d in bg.items()}
    print("bg windows:", {g: {k: len(v) for k, v in d.items()}
                          for g, d in bg_wins.items()}, flush=True)

    # ---- 4. featurize all window pools ------------------------------------------
    print("featurizing...", flush=True)
    pdf_feat = {}
    for pdf, d in win_by_pdf.items():
        pdf_feat[pdf] = {"genre": d["genre"],
                         "ai": feats(d["ai"]), "hu": feats(d["hu"])}
    anch_feat = {g: {k: feats(v) for k, v in d.items()} for g, d in anch_wins.items()}
    bg_feat = {g: {k: feats(v) for k, v in d.items()} for g, d in bg_wins.items()}
    dim = next(iter(pdf_feat.values()))["ai"].shape[1]
    print("feature dim:", dim, flush=True)

    # ---- 5. pairwise training set -----------------------------------------------
    pairs_pos, pairs_neg = [], []  # lists of feature rows
    # A. within-document pairs (core): each AI window x up to 4 same-doc human windows
    nA = 0
    for pdf, d in pdf_feat.items():
        xa, xh = d["ai"], d["hu"]
        for i in range(len(xa)):
            js = rng.choice(len(xh), size=min(4, len(xh)), replace=False)
            for j in js:
                pairs_pos.append(xa[i])
                pairs_neg.append(xh[j])
                nA += 1
    print(f"A within-doc pairs: {nA}", flush=True)
    # B. cross-document within-genre pairs: mixed-doc AI vs anchor-human; mixed-doc
    #    human vs pure-AI anchors (same genre) — teaches cross-doc within-register rank
    nB = 0
    for pdf, d in pdf_feat.items():
        g = d["genre"]
        xa, xh = d["ai"], d["hu"]
        ah, aa = anch_feat[g]["hu"], anch_feat[g]["ai"]
        if len(ah):
            for i in rng.choice(len(xa), size=min(len(xa), 200), replace=False):
                j = rng.randint(0, len(ah))
                pairs_pos.append(xa[i]); pairs_neg.append(ah[j]); nB += 1
        if len(aa):
            for i in rng.choice(len(xh), size=min(len(xh), 200), replace=False):
                j = rng.randint(0, len(aa))
                pairs_pos.append(aa[j]); pairs_neg.append(xh[i]); nB += 1
    print(f"B cross-doc pairs: {nB}", flush=True)
    # C. background anchor pairs within genre (absolute-scale anchoring)
    nC = 0
    for g in ("report", "prose"):
        ba, bh = bg_feat[g]["ai"], bg_feat[g]["ai"]
        bh = bg_feat[g]["hu"]
        n = min(200, len(ba), len(bh))
        for i in rng.choice(len(ba), size=n, replace=False):
            j = rng.randint(0, len(bh))
            pairs_pos.append(ba[i]); pairs_neg.append(bh[j]); nC += 1
    print(f"C background pairs: {nC}", flush=True)

    P = np.stack(pairs_pos)
    N = np.stack(pairs_neg)
    # forward pairs (AI - human -> 1) + reversed pairs (human - AI -> 0):
    # makes it a proper 2-class problem so CV works; ranking objective unchanged
    D = np.vstack([P - N, N - P])
    y = np.concatenate([np.ones(len(P)), np.zeros(len(N))])
    print(f"pairwise train: {len(D)} pair-diffs ({len(P)} fwd + {len(N)} rev), "
          f"dim {D.shape[1]}", flush=True)

    # ---- 6. train linear RankNet (logistic on differences, no intercept) --------
    best = None
    for C in (0.1, 0.25, 0.5, 1.0):
        clf = LogisticRegression(C=C, max_iter=5000, fit_intercept=False)
        s = cross_val_score(clf, D, y, cv=3, scoring="accuracy").mean()
        print(f"C={C}: pairwise CV acc={s:.4f}", flush=True)
        if best is None or s > best[1]:
            best = (C, s)
    Cbest = best[0]
    ranker = LogisticRegression(C=Cbest, max_iter=5000, fit_intercept=False).fit(D, y)
    w = ranker.coef_[0]
    print(f"chosen C={Cbest} (CV acc={best[1]:.4f}); train pairwise acc={ranker.score(D, y):.4f}",
          flush=True)

    # ---- 7. per-genre Platt calibration on absolute labels ----------------------
    cal_windows = {"report": ([], []), "prose": ([], [])}  # (texts->scores, labels)
    def add_scored(feat_mat, labels, genre):
        s = feat_mat @ w
        cal_windows[genre][0].extend(s.tolist())
        cal_windows[genre][1].extend(labels)

    for pdf, d in pdf_feat.items():
        g = d["genre"]
        add_scored(d["ai"], [1] * len(d["ai"]), g)
        add_scored(d["hu"], [0] * len(d["hu"]), g)
    for g in ("report", "prose"):
        add_scored(anch_feat[g]["ai"], [1] * len(anch_feat[g]["ai"]), g)
        add_scored(anch_feat[g]["hu"], [0] * len(anch_feat[g]["hu"]), g)
        add_scored(bg_feat[g]["ai"], [1] * len(bg_feat[g]["ai"]), g)
        add_scored(bg_feat[g]["hu"], [0] * len(bg_feat[g]["hu"]), g)

    cals = {}
    cw = os.environ.get("V3_CAL_CW", "")  # e.g. "balanced"
    use_iso = os.environ.get("V3_CAL_ISO", "") == "1"
    if use_iso:
        from sklearn.isotonic import IsotonicRegression
    for g in ("report", "prose"):
        ss = np.asarray(cal_windows[g][0])
        yy = np.asarray(cal_windows[g][1])
        if use_iso:
            order = np.argsort(ss, kind="mergesort")
            ss_s, yy_s = ss[order], yy[order]
            ux = np.unique(ss_s)
            by = np.array([yy_s[ss_s == x].mean() for x in ux])
            iso = IsotonicRegression(out_of_bounds="clip").fit(ux, by)
            cals[g] = ("iso", iso)
            acc = ((iso.predict(ss) >= 0.5) == yy).mean()
            print(f"cal {g} (isotonic): n={len(yy)}, train acc={acc:.4f}",
                  flush=True)
        else:
            cal = LogisticRegression(C=1.0, max_iter=2000,
                                     class_weight=cw or None).fit(
                                         ss.reshape(-1, 1), yy)
            a, b = float(cal.coef_[0][0]), float(cal.intercept_[0])
            cals[g] = ("platt", (a, b))
            print(f"cal {g} (cw={cw or 'none'}): n={len(yy)}, a={a:.4f}, "
                  f"b={b:.4f}, train acc={cal.score(ss.reshape(-1, 1), yy):.4f}",
                  flush=True)

    # ---- 8. save ----------------------------------------------------------------
    bundle = {
        "pair_w": w,
        "cal_report": cals["report"],
        "cal_prose": cals["prose"],
        "router": router,
        "router_scaler": router_scaler,
        "new_scaler": new_scaler,
        "new_names": v2b["new_names"],
        "genre_names": GENRE_FEATURE_NAMES,
        "base_dim": 187,
        "seed": 20261016,
        "pair_C": Cbest,
        "note": "v3_pairwise: linear RankNet on within-doc AI/human differences + "
                "per-genre Platt calibration; trained on Turnitin sentence labels, "
                "anchors, register-filtered background",
    }
    out = os.path.join(ROOT, "models", "nak_detector_v3_pairwise.joblib")
    assert os.path.abspath(out) != os.path.abspath(
        os.path.join(ROOT, "models", "nak_detector.joblib")), "SAFETY: never overwrite live"
    joblib.dump(bundle, out)
    meta = {
        "seed": 20261016,
        "pairs": {"A_within_doc": nA, "B_cross_doc": nB, "C_background": nC,
                  "total": int(len(D))},
        "pair_C": Cbest, "pair_cv_acc": round(float(best[1]), 4),
        "pair_train_acc": round(float(ranker.score(D, y)), 4),
        "cal": {g: dict(
                    {"kind": cals[g][0],
                     "a": round(cals[g][1][0], 4),
                     "b": round(cals[g][1][1], 4)}
                    if cals[g][0] == "platt" else {"kind": "iso"},
                    n=len(cal_windows[g][1]))
                for g in ("report", "prose")},
        "feature_dim": int(dim),
        "note": bundle["note"],
    }
    json.dump(meta, open(out.replace(".joblib", "_meta.json"), "w"), indent=1)
    print("SAVED", out, flush=True)
    print(json.dumps(meta, indent=1))



if __name__ == "__main__":
    main()
