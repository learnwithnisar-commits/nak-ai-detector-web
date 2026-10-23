#!/usr/bin/env python3
"""Build v3b: pairwise/contrastive model on the chrome-stripped v2 dataset.

vs v3 (nak_detector_v3_pairwise):
  - Texts from data/phase2_dataset_v2.jsonl (Phase-6 chrome-stripping fix).
  - Trains ONLY on v2-train docs (v1-locked-test docs quarantined out).
  - Calibration variant selected on v2-val by MAE (not train accuracy).
  - 8-sentence windows, stride 4 (matches app inference; Phase-6 fix #2).

Architecture (same as v3): linear RankNet on within-doc AI/human difference
vectors + per-genre Platt/isotonic calibration + v2 genre router blend.
Inference: app-identical (8-sent windows/stride 4, sentence means).

Saves: models/nak_detector_v3b_turnitin.joblib (+ _meta.json).
NEVER touches models/nak_detector.joblib.
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
                            new_features_10, SENT_LABELS)
from src.v2_genre import GENRE_FEATURE_NAMES  # noqa: F401

DATA = os.path.join(ROOT, "data")
V2_DS = os.path.join(DATA, "phase2_dataset_v2.jsonl")
SPLITS = os.path.join(DATA, "phase4_splits_v2")


def main():
    rng = np.random.RandomState(20261017)
    log = lambda m: print(m, flush=True)

    # ---- reuse v2's router + new-feature scaler --------------------------------
    v2b = joblib.load(os.path.join(ROOT, "models", "nak_detector_v2_turnitin.joblib"))
    router, router_scaler, new_scaler = (v2b["router"], v2b["router_scaler"],
                                         v2b["new_scaler"])
    log("reused v2 router + new_scaler")

    recs = [json.loads(l) for l in open(V2_DS, encoding="utf-8")]
    by_id = {r["doc_id"]: r for r in recs}
    train_ids = json.load(open(os.path.join(SPLITS, "train.json")))
    val_ids = json.load(open(os.path.join(SPLITS, "val.json")))
    log(f"train docs: {len(train_ids)}, val docs: {len(val_ids)}")

    def feats(windows):
        if not windows:
            return np.zeros((0, 197))
        Xb, Xn = window_features(windows)
        return np.hstack([Xb, new_scaler.transform(Xn)])

    # map sentence-label pdf -> v2 train doc
    pdf_to_doc = {}
    for r in recs:
        for p in r.get("all_source_pdfs", []):
            pdf_to_doc[p] = r["doc_id"]
    train_set = set(train_ids)
    by_pdf = {}
    for s in SENT_LABELS:
        d = pdf_to_doc.get(s["pdf"])
        if d in train_set:
            by_pdf.setdefault(s["pdf"], []).append(s)
    log(f"sentence-label pdfs in v2-train: {len(by_pdf)}")

    # ---- 1. sentence-label windows (mixed docs) --------------------------------
    win_by_pdf = {}
    for pdf, ss in by_pdf.items():
        doc_id = pdf_to_doc[pdf]
        key = doc_id
        try:
            g = genre_label(key, by_id[doc_id]["text"])
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
    log(f"mixed-doc windows: pdfs={len(win_by_pdf)} AI={n_ai} HU={n_hu}")

    # ---- 2. pure-doc windows from chrome-stripped v2 train texts ----------------
    anch_wins = {"report": {"ai": [], "hu": []}, "prose": {"ai": [], "hu": []}}
    for did in train_ids:
        r = by_id[did]
        s = r.get("turnitin_ai_score")
        if s is None:
            continue  # *% band: not used as absolute label
        t = r.get("text") or ""
        if len(t.split()) < 50:
            continue
        g = genre_label(did, t)
        wins = doc_windows(t)  # 8 sentences, stride 4
        if s == 0:
            anch_wins[g]["hu"].extend(wins)
        elif s >= 60:
            anch_wins[g]["ai"].extend(wins)
    log("pure-doc windows: " + str(
        {g: {k: len(v) for k, v in d.items()} for g, d in anch_wins.items()}))

    # ---- 3. background (register-filtered, same as v3) --------------------------
    def load_jsonl(path, n, min_words=60):
        recs_ = []
        try:
            for l in open(path, encoding="utf-8"):
                if len(recs_) >= n:
                    break
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                t = r.get("text", "")
                if len(t.split()) < min_words:
                    continue
                recs_.append(t)
        except FileNotFoundError:
            pass
        return recs_

    bg = {"report": {"ai": [], "hu": []}, "prose": {"ai": [], "hu": []}}
    for p in ["data/chatgpt_technical_engineering.jsonl",
              "data/chatgpt_natural_sciences.jsonl"]:
        for t in load_jsonl(os.path.join(ROOT, p), 40):
            bg[genre_label("x", t)]["ai"].append(t)
    for t in load_jsonl(os.path.join(ROOT, "data/arxiv_train.jsonl"), 600, 30):
        bg["report"]["hu"].append(t)
    for t in load_jsonl(os.path.join(ROOT, "data/chatgpt_law_humanities.jsonl"), 40):
        bg["prose"]["ai"].append(t)
    n_ph = 0
    for l in open(os.path.join(ROOT, "data/train.jsonl"), encoding="utf-8"):
        if n_ph >= 800:
            break
        r = json.loads(l)
        if r.get("genre") in ("personal_essay", "blog_post", "short_story_opening",
                              "story_opening") and r.get("label") == 0:
            bg["prose"]["hu"].append(r["text"])
            n_ph += 1

    def wins_of(docs, stride=8):
        out = []
        for t in docs:
            out.extend(doc_windows(t, stride=stride))
        return out

    bg_wins = {g: {k: wins_of(v) for k, v in d.items()} for g, d in bg.items()}
    log("bg windows: " + str(
        {g: {k: len(v) for k, v in d.items()} for g, d in bg_wins.items()}))

    # ---- 4. featurize -----------------------------------------------------------
    log("featurizing...")
    pdf_feat = {}
    for pdf, d in win_by_pdf.items():
        pdf_feat[pdf] = {"genre": d["genre"],
                         "ai": feats(d["ai"]), "hu": feats(d["hu"])}
    anch_feat = {g: {k: feats(v) for k, v in d.items()} for g, d in anch_wins.items()}
    bg_feat = {g: {k: feats(v) for k, v in d.items()} for g, d in bg_wins.items()}
    dim = 197
    if pdf_feat:
        dim = next(iter(pdf_feat.values()))["ai"].shape[1]
    log(f"feature dim: {dim}")

    # ---- 5. pairwise training set -----------------------------------------------
    pairs_pos, pairs_neg = [], []
    nA = 0
    for pdf, d in pdf_feat.items():
        xa, xh = d["ai"], d["hu"]
        for i in range(len(xa)):
            js = rng.choice(len(xh), size=min(4, len(xh)), replace=False)
            for j in js:
                pairs_pos.append(xa[i]); pairs_neg.append(xh[j]); nA += 1
    log(f"A within-doc pairs: {nA}")
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
    log(f"B cross-doc pairs: {nB}")
    nC = 0
    for g in ("report", "prose"):
        ba, bh = bg_feat[g]["ai"], bg_feat[g]["hu"]
        n = min(200, len(ba), len(bh))
        for i in rng.choice(len(ba), size=n, replace=False):
            j = rng.randint(0, len(bh))
            pairs_pos.append(ba[i]); pairs_neg.append(bh[j]); nC += 1
    log(f"C background pairs: {nC}")

    P = np.stack(pairs_pos); N = np.stack(pairs_neg)
    D = np.vstack([P - N, N - P])
    y = np.concatenate([np.ones(len(P)), np.zeros(len(N))])
    log(f"pairwise train: {len(D)} diffs, dim {D.shape[1]}")

    # ---- 6. RankNet --------------------------------------------------------------
    best = None
    for C in (0.1, 0.25, 0.5, 1.0):
        clf = LogisticRegression(C=C, max_iter=5000, fit_intercept=False)
        s = cross_val_score(clf, D, y, cv=3, scoring="accuracy").mean()
        log(f"C={C}: pairwise CV acc={s:.4f}")
        if best is None or s > best[1]:
            best = (C, s)
    Cbest = best[0]
    ranker = LogisticRegression(C=Cbest, max_iter=5000,
                                fit_intercept=False).fit(D, y)
    w = ranker.coef_[0]
    log(f"chosen C={Cbest} CV={best[1]:.4f} train_acc={ranker.score(D, y):.4f}")

    # ---- 7. calibration variants -------------------------------------------------
    cal_windows = {"report": ([], []), "prose": ([], [])}
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

    from sklearn.isotonic import IsotonicRegression
    variants = {}
    for g in ("report", "prose"):
        ss = np.asarray(cal_windows[g][0]); yy = np.asarray(cal_windows[g][1])
        cal = LogisticRegression(C=1.0, max_iter=2000).fit(ss.reshape(-1, 1), yy)
        variants.setdefault("platt", {})[g] = (
            "platt", (float(cal.coef_[0][0]), float(cal.intercept_[0])))
        cal_b = LogisticRegression(C=1.0, max_iter=2000,
                                   class_weight="balanced").fit(ss.reshape(-1, 1), yy)
        variants.setdefault("platt_bal", {})[g] = (
            "platt", (float(cal_b.coef_[0][0]), float(cal_b.intercept_[0])))
        order = np.argsort(ss, kind="mergesort")
        ss_s, yy_s = ss[order], yy[order]
        ux = np.unique(ss_s)
        by = np.array([yy_s[ss_s == x].mean() for x in ux])
        iso = IsotonicRegression(out_of_bounds="clip").fit(ux, by)
        variants.setdefault("iso", {})[g] = ("iso", iso)
    log(f"calibration variants: {list(variants)}")

    # ---- 8. val selection by MAE --------------------------------------------------
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from v3_model import V3Model  # noqa: E402

    def score_val(cal_bundle):
        tmp = "/tmp/v3b_val_candidate.joblib"
        bundle = {
            "pair_w": w, "cal_report": cal_bundle["report"],
            "cal_prose": cal_bundle["prose"], "router": router,
            "router_scaler": router_scaler, "new_scaler": new_scaler,
            "new_names": v2b["new_names"], "genre_names": GENRE_FEATURE_NAMES,
            "base_dim": 187,
        }
        joblib.dump(bundle, tmp)
        m = V3Model(tmp)
        errs, details = [], []
        for did in val_ids:
            r = by_id[did]
            ref = r["turnitin_ai_score"]
            pred = m.score_doc(r["text"])["ai_pct"]
            errs.append(abs(pred - ref))
            details.append((did, ref, pred))
        os.remove(tmp)
        return float(np.mean(errs)), details

    results = {}
    for vname, cals in variants.items():
        mae, _ = score_val(cals)
        results[vname] = mae
        log(f"val MAE [{vname}]: {mae:.2f}")
    best_v = min(results, key=results.get)
    log(f"SELECTED calibration: {best_v} (val MAE {results[best_v]:.2f})")
    _, val_details = score_val(variants[best_v])
    for did, ref, pred in val_details:
        log(f"  val {did}: ref={ref} pred={pred:.1f} err={abs(pred-ref):.1f}")

    # ---- 9. save ------------------------------------------------------------------
    cals = variants[best_v]
    bundle = {
        "pair_w": w, "cal_report": cals["report"], "cal_prose": cals["prose"],
        "router": router, "router_scaler": router_scaler, "new_scaler": new_scaler,
        "new_names": v2b["new_names"], "genre_names": GENRE_FEATURE_NAMES,
        "base_dim": 187, "seed": 20261017, "pair_C": Cbest,
        "cal_variant": best_v,
        "note": "v3b: RankNet on chrome-stripped v2-train (8-sent windows) + "
                "per-genre calibration selected on v2-val; sentence-highlight labels",
    }
    out = os.path.join(ROOT, "models", "nak_detector_v3b_turnitin.joblib")
    assert os.path.abspath(out) != os.path.abspath(
        os.path.join(ROOT, "models", "nak_detector.joblib")), "SAFETY: never overwrite live"
    joblib.dump(bundle, out)
    meta = {
        "seed": 20261017,
        "pairs": {"A_within_doc": nA, "B_cross_doc": nB, "C_background": nC,
                  "total": int(len(D))},
        "pair_C": Cbest, "pair_cv_acc": round(float(best[1]), 4),
        "pair_train_acc": round(float(ranker.score(D, y)), 4),
        "cal_variant": best_v, "val_mae": round(results[best_v], 2),
        "val_mae_all": {k: round(v, 2) for k, v in results.items()},
        "feature_dim": int(dim),
        "note": bundle["note"],
    }
    json.dump(meta, open(out.replace(".joblib", "_meta.json"), "w"), indent=1)
    log(f"SAVED {out}")
    log(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
