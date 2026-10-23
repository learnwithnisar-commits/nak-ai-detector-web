#!/usr/bin/env python3
"""Turnitin-methods round, Experiment B: retrain the ORIGINAL architecture
(sklearn LogisticRegression on 187 production features, 8-sent windows) with
paraphrase-augmented AI data, then apply Turnitin-style aggregation
(% sentences with p >= tau, tau tuned on v2 VAL for the new model).

Training data (quarantined):
  (a) v2 chrome-stripped train docs: score>=20 -> AI 1x; score==0 -> human 2x; *% excluded
  (b) sentence-highlight label windows (8-sent, majority label), dropping the
      5 v1-test PDFs found contaminated in the v1_retrain round
  (c) 8 verified paragraphs (4 human incl. third-party listeria / 4 AI) at 2x
  (d) 40 paraphrase-augmented AI samples (10 paragraph + 30 window) from
      data/paraphrase_aug.jsonl: paragraphs at 2x, windows at 1x

C selected on v2 VAL ONLY; tau (+deadzone on/off) tuned on v2 VAL ONLY.
v1 locked test scored EXACTLY ONCE at the end.

NEVER touches models/nak_detector.joblib. Candidate ONLY to
models/nak_detector_v1_ttmethods.joblib. No deploy.
"""
import glob
import hashlib
import json
import os
import sys
import time

PROJ = os.path.expanduser("~/workspace/ai-detector")
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.features import extract_features, feature_names
from app.scan import split_sentences, SentenceScorer

t0 = time.time()
SEED = 20261017
np.random.seed(SEED)
LIVE_MD5 = "9eeba80e03d06c7db8b35796965746c9"
TEST_SHA_PREFIX = "89f8b799"
OUT_MODEL = os.path.join(PROJ, "models", "nak_detector_v1_ttmethods.joblib")
FEAT_CACHE = os.path.join(PROJ, "data", "ttmethods_feats")
os.makedirs(FEAT_CACHE, exist_ok=True)
CONTAM_PDFS = {'65701_task_1__1.pdf', '84956.pdf', 'BA2_______02.pdf',
               'FIFA_World_Cup_Report__1___1.pdf', 'Final_Transcript.pdf'}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(m):
    print(m, flush=True)


live_before = md5(os.path.join(PROJ, "models", "nak_detector.joblib"))
assert live_before == LIVE_MD5, f"live model already modified! {live_before}"
test_before = sha256(os.path.join(PROJ, "data", "phase4_splits", "test.json"))
assert test_before.startswith(TEST_SHA_PREFIX), "v1 test manifest changed!"
log(f"pre-run integrity ok [{time.time()-t0:.0f}s]")

DIM = len(feature_names())
assert DIM == 187, f"expected 187, got {DIM}"
log(f"feature dim: {DIM}")


def windows(text, w=8, s=4, min_words=20):
    sents = split_sentences(text) or [text]
    out = []
    if len(sents) < 3:
        return []
    for i in range(0, max(1, len(sents) - w + 1), s):
        seg = " ".join(sents[i:i + w])
        if len(seg.split()) >= min_words:
            out.append(seg)
    return out


# ---------- load records ----------
recs = {}
with open("data/phase2_dataset_v2.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            o = json.loads(line)
            recs[o["doc_id"]] = o
train_ids = json.load(open("data/phase4_splits_v2/train.json"))
val_ids = json.load(open("data/phase4_splits_v2/val.json"))
log(f"v2 train={len(train_ids)} val={len(val_ids)}")

v1_test_ids = set(json.load(open("data/phase4_splits/test.json")))
assert not (v1_test_ids & set(train_ids)), "v1 test doc in v2 train!"

ai_segs, hu_segs = [], []

# (a) v2 train docs
n_ai_docs = n_hu_docs = 0
for did in train_ids:
    r = recs[did]
    sc = r["turnitin_ai_score"]
    txt = r["text"]
    if not txt or len(txt.split()) < 50:
        continue
    if sc is None or (0 < sc < 20):
        continue
    segs = windows(txt)
    if sc >= 20:
        ai_segs.extend(segs); n_ai_docs += 1
    else:
        hu_segs.extend(segs * 2); n_hu_docs += 1
log(f"(a) v2 train docs: AI={n_ai_docs} human={n_hu_docs}")

# (b) sentence labels, quarantined
labels = [x for x in json.load(open("data/sentence_labels_all.json"))
          if x["pdf"] not in CONTAM_PDFS]
by_pdf = {}
for x in labels:
    by_pdf.setdefault(x["pdf"], []).append(x)
n_ai_w = n_hu_w = 0
for pdf, rl in by_pdf.items():
    sents = [x["sentence_text"] for x in rl]
    labs = [1 if x["label"] == "ai" else 0 for x in rl]
    for i in range(0, max(1, len(sents) - 8 + 1), 4):
        ws, wl = sents[i:i + 8], labs[i:i + 8]
        if sum(len(s.split()) for s in ws) < 20:
            continue
        if sum(wl) / len(wl) >= 0.5:
            ai_segs.append(" ".join(ws)); n_ai_w += 1
        else:
            hu_segs.append(" ".join(ws)); n_hu_w += 1
log(f"(b) label windows (quarantined): AI={n_ai_w} human={n_hu_w}")

# (c) verified paragraphs (8)
verified = [("nisar_001_waste", 0), ("nisar_002_story", 0),
            ("nisar_003_school", 0), ("human_004_listeria", 0),
            ("ai_001_waste", 1), ("ai_002_story", 1),
            ("ai_003_school", 1), ("ai_004_listeria", 1)]
for stem, lab in verified:
    p = f"data/verified/{stem}.txt"
    txt = "\n".join(l for l in open(p, encoding="utf-8").read().split("\n")
                    if not l.startswith("#")).strip()
    assert len(txt.split()) >= 50, stem
    (ai_segs if lab else hu_segs).extend([txt] * 2)
log(f"(c) verified paragraphs: {len(verified)} docs at 2x")

# (d) paraphrase augmentation (all label=1 AI)
n_pp_para = n_pp_win = 0
for line in open("data/paraphrase_aug.jsonl", encoding="utf-8"):
    p = json.loads(line)
    assert p["label"] == 1
    txt = p["text"].strip()
    assert len(txt.split()) >= 20, p["id"]
    if p["kind"] == "paragraph":
        ai_segs.extend([txt] * 2); n_pp_para += 1
    else:
        ai_segs.append(txt); n_pp_win += 1
log(f"(d) paraphrases: {n_pp_para} paragraphs@2x + {n_pp_win} windows@1x")

log(f"segments raw: AI={len(ai_segs)} human={len(hu_segs)} [{time.time()-t0:.0f}s]")

# balance: human capped at 3x AI
rng = np.random.RandomState(SEED)
rng.shuffle(hu_segs)
hu_segs = hu_segs[:3 * len(ai_segs)]
seg_texts = ai_segs + hu_segs
seg_y = np.array([1] * len(ai_segs) + [0] * len(hu_segs))
idx = np.arange(len(seg_texts)); rng.shuffle(idx)
seg_texts = [seg_texts[i] for i in idx]; seg_y = seg_y[idx]
log(f"segments final: {len(seg_texts)} pos={int(seg_y.sum())} [{time.time()-t0:.0f}s]")

# ---------- features (own cache dir) ----------
CHUNK = 2000
n = len(seg_texts)
n_chunks = (n + CHUNK - 1) // CHUNK
for i in range(0, n, CHUNK):
    ckpt = os.path.join(FEAT_CACHE, f"chunk_{i//CHUNK:03d}.npz")
    if os.path.exists(ckpt):
        continue
    s = slice(i, min(i + CHUNK, n))
    X = np.stack([extract_features(t) for t in seg_texts[s]])
    assert X.shape[1] == DIM
    np.savez_compressed(ckpt, X=X, y=seg_y[s])
    log(f"  feat chunk {i//CHUNK+1}/{n_chunks} [{time.time()-t0:.0f}s]")
Xs, ys = [], []
for ckpt in sorted(glob.glob(os.path.join(FEAT_CACHE, "chunk_*.npz"))):
    d = np.load(ckpt)
    Xs.append(d["X"]); ys.append(d["y"])
X = np.vstack(Xs); y = np.concatenate(ys)
assert len(X) == len(seg_texts), "feature cache stale — delete data/ttmethods_feats and rerun"
log(f"X={X.shape} pos={int(y.sum())} [{time.time()-t0:.0f}s]")


# ---------- Turnitin-style scorer ----------
def make_scorer(clf):
    def score_text(t):
        f = np.asarray(extract_features(t), dtype=np.float64).reshape(1, -1)
        p = float(clf.predict_proba(f)[0][1])
        return min(max(p, 0.0), 1.0)

    def sent_scores(text):
        scorer = SentenceScorer(score_text)
        for s in split_sentences(text) or [text]:
            if s.strip():
                scorer.feed(s)
        scorer.finish()
        out = []
        while scorer.finalized:
            _, v = scorer.finalized.popleft()
            out.append(v)
        return out

    def doc_pct(sp, tau, deadzone):
        if deadzone:
            sp = [0.0 if 0.35 <= p <= 0.65 else p for p in sp]
        if not sp:
            return 0.0
        return round(100.0 * float(np.mean([1.0 if p >= tau else 0.0 for p in sp])), 1)

    return sent_scores, doc_pct


TAUS = [round(0.30 + 0.05 * i, 2) for i in range(13)]

# precompute val per-sentence scores once per C (model-dependent)
val_texts = [(did, recs[did]["text"], recs[did]["turnitin_ai_score"]) for did in val_ids]


def tune_eval(clf):
    """Tune tau (+deadzone) on v2 VAL; return (best_mae, best_tau, best_dz, val_sps)."""
    sent_scores, doc_pct = make_scorer(clf)
    val_sps = [(did, ref, sent_scores(txt)) for did, txt, ref in val_texts]
    best = None
    for dz in (False, True):
        for tau in TAUS:
            errs, fp, n0 = [], 0, 0
            for _, ref, sp in val_sps:
                pred = doc_pct(sp, tau, dz)
                errs.append(abs(pred - ref))
                if ref == 0:
                    n0 += 1
                    fp += (pred >= 20)
            mae = float(np.mean(errs))
            fpr = fp / n0 if n0 else 0.0
            key = (mae, fpr)
            if best is None or key < (best[0], best[1]):
                best = (mae, fpr, tau, dz)
    return best[0], best[2], best[3], val_sps


# ---------- train C grid, select on VAL ONLY ----------
results = {}
for C in (0.5, 1.0, 2.0):
    clf = LogisticRegression(max_iter=3000, C=C, n_jobs=-1,
                             class_weight="balanced", random_state=SEED)
    clf.fit(X, y)
    mae, tau, dz, _ = tune_eval(clf)
    results[C] = {"mae": mae, "tau": tau, "dz": dz, "clf": clf}
    log(f"C={C}: val MAE={mae:.2f} @ tau={tau} dz={dz} [{time.time()-t0:.0f}s]")
best_C = min(results, key=lambda c: results[c]["mae"])
R = results[best_C]
log(f"SELECTED C={best_C} (val MAE={R['mae']:.2f}, tau={R['tau']}, dz={R['dz']})")

joblib.dump(R["clf"], OUT_MODEL)
log(f"saved candidate: {OUT_MODEL}")

# ---------- FINAL: v1 locked test, EXACTLY ONCE ----------
v1_recs = {}
with open("data/phase2_dataset.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            o = json.loads(line)
            v1_recs[o["doc_id"]] = o
sent_scores, doc_pct = make_scorer(R["clf"])
rows = []
for did in json.load(open("data/phase4_splits/test.json")):
    r = v1_recs[did]
    ref = r["turnitin_ai_score"]
    assert isinstance(ref, (int, float))
    sp = sent_scores(r["text"])
    pred = doc_pct(sp, R["tau"], R["dz"])
    ae = abs(pred - ref)
    rows.append({"doc_id": did, "turnitin_ref": ref, "predicted": pred,
                 "abs_err": round(ae, 1), "signed_err": round(pred - ref, 1),
                 "within_5": ae <= 5})
mae = float(np.mean([r["abs_err"] for r in rows]))
med = float(np.median([r["abs_err"] for r in rows]))
signed = float(np.mean([r["signed_err"] for r in rows]))
w5 = sum(1 for r in rows if r["within_5"])
log("---- v1 LOCKED TEST (n=16, Turnitin-style agg, scored once) ----")
for r in rows:
    log(f"  {r['doc_id']}: ref={r['turnitin_ref']} pred={r['predicted']} ae={r['abs_err']}")
log(f"MAE={mae:.2f} medianAE={med:.2f} signed={signed:+.2f} within5={w5}/16")

# *% compliance on v1 star pool (68)
star_pool = json.load(open("data/phase4_splits/star_band_pool.json"))
star_ids = star_pool if isinstance(star_pool, list) else star_pool["doc_ids"]
in_band, n_star = 0, 0
for did in star_ids:
    r = v1_recs[did]
    assert r["turnitin_ai_band"] == "*%"
    if not r["text"] or len(r["text"].split()) < 50:
        continue
    sp = sent_scores(r["text"])
    pred = doc_pct(sp, R["tau"], R["dz"])
    n_star += 1
    if 1.0 <= pred <= 19.0:
        in_band += 1
compliance = in_band / n_star if n_star else 0
log(f"star-band: {in_band}/{n_star} = {compliance*100:.1f}%")

# ---------- gates ----------
gates = {"MAE < 19.54": mae < 19.54,
         "within-5 >= 4/16": w5 >= 4,
         "star compliance >= 70%": compliance >= 0.70}
verdict = "PASS" if all(gates.values()) else "HOLD"
for g, ok in gates.items():
    log(f"gate [{g}]: {'PASS' if ok else 'FAIL'}")
log(f"VERDICT: {verdict}")

live_after = md5(os.path.join(PROJ, "models", "nak_detector.joblib"))
test_after = sha256(os.path.join(PROJ, "data", "phase4_splits", "test.json"))
assert live_after == LIVE_MD5, "LIVE MODEL MODIFIED!"
assert test_after == test_before, "v1 test manifest MODIFIED!"
log("post-run integrity: live model + v1 test manifest unchanged")

meta = {
    "seed": SEED, "feature_dim": DIM, "selected_C": best_C,
    "tuned_tau": R["tau"], "tuned_deadzone": R["dz"],
    "val_results": {str(c): {"mae": round(v["mae"], 2), "tau": v["tau"], "dz": v["dz"]}
                    for c, v in results.items()},
    "n_segments": len(seg_texts), "n_pos_segments": int(seg_y.sum()),
    "n_ai_docs_v2": n_ai_docs, "n_hu_docs_v2": n_hu_docs,
    "label_windows": {"ai": n_ai_w, "human": n_hu_w},
    "paraphrases": {"paragraphs_2x": n_pp_para, "windows_1x": n_pp_win},
    "locked_test": {"mae": round(mae, 2), "median_ae": round(med, 2),
                    "mean_signed": round(signed, 2), "within_5": w5, "n": 16,
                    "rows": rows},
    "star_band": {"in_band": in_band, "n": n_star,
                  "compliance": round(compliance, 4)},
    "gates": gates, "verdict": verdict,
    "live_md5_before": live_before, "live_md5_after": live_after,
    "v1test_sha256": test_after,
}
with open(os.path.join(PROJ, "models", "nak_detector_v1_ttmethods_meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
with open(os.path.join(PROJ, "data", "phase5_baseline", "v1_ttmethods_locked_test_eval.json"), "w") as f:
    json.dump({"metrics": meta["locked_test"], "rows": rows,
               "star": meta["star_band"], "gates": gates, "verdict": verdict,
               "tuned": {"C": best_C, "tau": R["tau"], "deadzone": R["dz"]}}, f, indent=1)
log(f"DONE verdict={verdict} [{time.time()-t0:.0f}s]")
