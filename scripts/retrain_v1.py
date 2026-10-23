#!/usr/bin/env python3
"""v1_retrain: retrain the ORIGINAL architecture (sklearn LogisticRegression on
187 production-scaled features, 8-sent windows) on the new data combination:
chrome-stripped v2 train + 3,871 Turnitin sentence-highlight labels +
6 verified paragraphs (3 human / 3 AI).

Same pipeline as prior original-arch rounds (segfull_train_oct16_newanchors.py):
  - windows(text, w=8, s=4, min_words=20)
  - extract_features (187-dim, production-scaled via live artifacts)
  - LogisticRegression(max_iter=3000, class_weight='balanced')
  - AI docs 1x, human docs 2x (never >2x), human segs capped at 3x AI segs
  - *%-band excluded (no hard label)

C is selected on v2 VAL ONLY (document-level MAE, app-identical aggregation).
v1 locked test is scored EXACTLY ONCE at the end.

NEVER touches models/nak_detector.joblib. Saves candidate ONLY to
models/nak_detector_v1_retrain.joblib.
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
from app.parse import parse_document

t0 = time.time()
SEED = 20261017
np.random.seed(SEED)
LIVE_MD5 = "9eeba80e03d06c7db8b35796965746c9"
OUT_MODEL = os.path.join(PROJ, "models", "nak_detector_v1_retrain.joblib")
FEAT_CACHE = os.path.join(PROJ, "data", "retrain_v1_feats")
os.makedirs(FEAT_CACHE, exist_ok=True)


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


# ---------- integrity: live model untouched at start ----------
live_before = md5(os.path.join(PROJ, "models", "nak_detector.joblib"))
assert live_before == LIVE_MD5, f"live model already modified! {live_before}"
test_before = sha256(os.path.join(PROJ, "data", "phase4_splits", "test.json"))
assert test_before.startswith("89f8b799"), f"v1 test manifest changed! {test_before[:16]}"
log(f"pre-run: live md5 ok, v1 test sha256 {test_before[:16]}...")

DIM = len(feature_names())
log(f"feature dim: {DIM}")
assert DIM == 187, f"expected 187, got {DIM}"


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


# ---------- load v2 records ----------
recs = {}
with open("data/phase2_dataset_v2.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            o = json.loads(line)
            recs[o["doc_id"]] = o
train_ids = json.load(open("data/phase4_splits_v2/train.json"))
val_ids = json.load(open("data/phase4_splits_v2/val.json"))
log(f"v2 train docs: {len(train_ids)}, val docs: {len(val_ids)}")

# ---------- v1 locked test quarantine ----------
v1_test_ids = set(json.load(open("data/phase4_splits/test.json")))
v1_recs = {}
with open("data/phase2_dataset.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            o = json.loads(line)
            v1_recs[o["doc_id"]] = o
v1_test_pdfs = set()
for did in v1_test_ids:
    r = v1_recs[did]
    for k in ("source_pdf", "all_source_pdfs"):
        v = r.get(k)
        if isinstance(v, list):
            v1_test_pdfs.update(v)
        elif v:
            v1_test_pdfs.add(v)
overlap = v1_test_ids & set(train_ids)
assert not overlap, f"v1 test doc_ids in v2 train: {overlap}"
log(f"quarantine: {len(v1_test_ids)} v1 test docs, {len(v1_test_pdfs)} source PDFs catalogued")

# ---------- build training segments ----------
ai_segs, hu_segs = [], []

# (a) v2 train docs: score>=20 -> AI (1x); score==0 -> human (2x); *% excluded
n_ai_docs = n_hu_docs = 0
for did in train_ids:
    r = recs[did]
    assert r["report_class"] == "ai_report", did
    sc = r["turnitin_ai_score"]
    txt = r["text"]
    if not txt or len(txt.split()) < 50:
        continue
    if sc is None or (0 < sc < 20):
        continue  # *%-band / unscored: no hard label
    segs = windows(txt)
    if sc >= 20:
        ai_segs.extend(segs)
        n_ai_docs += 1
    else:
        hu_segs.extend(segs * 2)  # 2x human counterbalance (max allowed)
        n_hu_docs += 1
log(f"v2 train docs used: AI={n_ai_docs} human={n_hu_docs}")

# (b) sentence-highlight labels -> 8-sent windows, majority label (1x)
labels = json.load(open("data/sentence_labels_all.json"))
label_pdfs = {x["pdf"] for x in labels}
leak = label_pdfs & v1_test_pdfs
if leak:
    log(f"QUARANTINE: dropping {len(leak)} v1-test PDFs from sentence labels: {sorted(leak)}")
    labels = [x for x in labels if x["pdf"] not in v1_test_pdfs]
    label_pdfs = {x["pdf"] for x in labels}
log(f"sentence labels after quarantine: {len(labels)} from {len(label_pdfs)} PDFs")
by_pdf = {}
for x in labels:
    by_pdf.setdefault(x["pdf"], []).append(x)
n_ai_w = n_hu_w = 0
for pdf, recs_l in by_pdf.items():
    sents = [x["sentence_text"] for x in recs_l]
    labs = [1 if x["label"] == "ai" else 0 for x in recs_l]
    for i in range(0, max(1, len(sents) - 8 + 1), 4):
        ws, wl = sents[i:i + 8], labs[i:i + 8]
        if sum(len(s.split()) for s in ws) < 20:
            continue
        lab = 1 if sum(wl) / len(wl) >= 0.5 else 0
        if lab:
            ai_segs.append(" ".join(ws)); n_ai_w += 1
        else:
            hu_segs.append(" ".join(ws)); n_hu_w += 1
log(f"label windows: AI={n_ai_w} human={n_hu_w} from {len(by_pdf)} PDFs")

# (c) verified paragraphs: whole documents, 2x (max allowed)
for stem, lab in [("nisar_001_waste", 0), ("nisar_002_story", 0),
                  ("nisar_003_school", 0), ("ai_001_waste", 1),
                  ("ai_002_story", 1), ("ai_003_school", 1)]:
    p = f"data/verified/{stem}.txt"
    txt = "\n".join(l for l in open(p, encoding="utf-8").read().split("\n")
                    if not l.startswith("#")).strip()
    assert len(txt.split()) >= 50, stem
    assert "19__2" not in stem  # paranoia: verified files are new, not test docs
    if lab:
        ai_segs.extend([txt] * 2)
    else:
        hu_segs.extend([txt] * 2)
log(f"verified paragraphs: 6 docs at 2x included")

log(f"segments raw: AI={len(ai_segs)} human={len(hu_segs)} [{time.time()-t0:.0f}s]")

# balance: human capped at 3x AI (established)
rng = np.random.RandomState(SEED)
rng.shuffle(hu_segs)
hu_segs = hu_segs[:3 * len(ai_segs)]
seg_texts = ai_segs + hu_segs
seg_y = np.array([1] * len(ai_segs) + [0] * len(hu_segs))
idx = np.arange(len(seg_texts)); rng.shuffle(idx)
seg_texts = [seg_texts[i] for i in idx]; seg_y = seg_y[idx]
log(f"segments final: {len(seg_texts)} pos={int(seg_y.sum())} [{time.time()-t0:.0f}s]")

# ---------- features (cached) ----------
CHUNK = 2000
n = len(seg_texts)
for i in range(0, n, CHUNK):
    ckpt = os.path.join(FEAT_CACHE, f"chunk_{i//CHUNK:03d}.npz")
    if os.path.exists(ckpt):
        continue
    s = slice(i, min(i + CHUNK, n))
    X = np.stack([extract_features(t) for t in seg_texts[s]])
    assert X.shape[1] == DIM
    np.savez_compressed(ckpt, X=X, y=seg_y[s])
    log(f"  feat chunk {i//CHUNK+1}/{(n+CHUNK-1)//CHUNK} [{time.time()-t0:.0f}s]")
Xs, ys = [], []
for ckpt in sorted(glob.glob(os.path.join(FEAT_CACHE, "chunk_*.npz"))):
    d = np.load(ckpt)
    Xs.append(d["X"]); ys.append(d["y"])
X = np.vstack(Xs); y = np.concatenate(ys)
assert len(X) == len(seg_texts), "feature cache stale — delete data/retrain_v1_feats and rerun"
log(f"X={X.shape} pos={int(y.sum())} [{time.time()-t0:.0f}s]")

# ---------- app-identical document scorer ----------
def make_scorer(clf):
    def score_text(t):
        f = np.asarray(extract_features(t), dtype=np.float64).reshape(1, -1)
        p = float(clf.predict_proba(f)[0][1])
        return min(max(p, 0.0), 1.0)

    def score_document(text):
        scorer = SentenceScorer(score_text)
        for s in split_sentences(text) or [text]:
            if s.strip():
                scorer.feed(s)
        scorer.finish()
        ss, sc = 0.0, 0
        while scorer.finalized:
            _, v = scorer.finalized.popleft()
            ss += v; sc += 1
        if not sc or len(text.split()) < 300:
            return None
        return round(ss / sc * 100, 1)
    return score_document


def val_mae(clf):
    sc_doc = make_scorer(clf)
    errs = []
    for did in val_ids:
        r = recs[did]
        pred = sc_doc(r["text"])
        if pred is None:
            continue
        errs.append(abs(pred - r["turnitin_ai_score"]))
    return sum(errs) / len(errs), len(errs)


# ---------- train C grid, select on VAL ONLY ----------
results = {}
for C in (0.5, 1.0, 2.0):
    clf = LogisticRegression(max_iter=3000, C=C, n_jobs=-1,
                             class_weight="balanced", random_state=SEED)
    clf.fit(X, y)
    mae, ndocs = val_mae(clf)
    results[C] = (mae, ndocs, clf)
    log(f"C={C}: val MAE={mae:.2f} on {ndocs} docs [{time.time()-t0:.0f}s]")
best_C = min(results, key=lambda c: results[c][0])
best_clf = results[best_C][2]
log(f"SELECTED C={best_C} (val MAE={results[best_C][0]:.2f})")

joblib.dump(best_clf, OUT_MODEL)
log(f"saved candidate: {OUT_MODEL}")

# ---------- FINAL: v1 locked test, EXACTLY ONCE ----------
sc_doc = make_scorer(best_clf)
rows = []
for did in json.load(open("data/phase4_splits/test.json")):
    r = v1_recs[did]
    ref = r["turnitin_ai_score"]
    assert isinstance(ref, (int, float))
    pred = sc_doc(r["text"])
    assert pred is not None, f"no prediction for {did}"
    ae = abs(pred - ref)
    rows.append({"doc_id": did, "turnitin_ref": ref, "predicted": pred,
                 "abs_err": round(ae, 1), "signed_err": round(pred - ref, 1),
                 "within_5": ae <= 5})
mae = sum(r["abs_err"] for r in rows) / len(rows)
med = float(np.median([r["abs_err"] for r in rows]))
signed = sum(r["signed_err"] for r in rows) / len(rows)
w5 = sum(1 for r in rows if r["within_5"])
log("---- v1 LOCKED TEST (n=16, scored once) ----")
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
    pred = sc_doc(r["text"])
    n_star += 1
    if pred is not None and 1.0 <= pred <= 19.0:
        in_band += 1
compliance = in_band / n_star if n_star else 0
log(f"star-band: {in_band}/{n_star} = {compliance*100:.1f}%")

# ---------- gates ----------
BASE_MAE, BASE_W5, BASE_STAR = 19.54, 3, 0.75
gates = {
    "MAE < 19.54": mae < BASE_MAE,
    f"within-5 >= 4/16": w5 >= 4,
    "star compliance >= 70%": compliance >= 0.70,
}
verdict = "PASS" if all(gates.values()) else "HOLD"
for g, ok in gates.items():
    log(f"gate [{g}]: {'PASS' if ok else 'FAIL'}")
log(f"VERDICT: {verdict}")

# ---------- integrity after ----------
live_after = md5(os.path.join(PROJ, "models", "nak_detector.joblib"))
test_after = sha256(os.path.join(PROJ, "data", "phase4_splits", "test.json"))
assert live_after == LIVE_MD5, "LIVE MODEL MODIFIED!"
assert test_after == test_before, "v1 test manifest MODIFIED!"
log("post-run integrity: live model + v1 test manifest unchanged")

meta = {
    "seed": SEED, "feature_dim": DIM, "selected_C": best_C,
    "val_results": {str(c): {"mae": round(v[0], 2), "ndocs": v[1]} for c, v in results.items()},
    "n_segments": len(seg_texts), "n_pos_segments": int(seg_y.sum()),
    "n_ai_docs_v2": n_ai_docs, "n_hu_docs_v2": n_hu_docs,
    "label_windows": {"ai": n_ai_w, "human": n_hu_w},
    "locked_test": {"mae": round(mae, 2), "median_ae": round(med, 2),
                    "mean_signed": round(signed, 2), "within_5": w5, "n": 16,
                    "rows": rows},
    "star_band": {"in_band": in_band, "n": n_star,
                  "compliance": round(compliance, 4)},
    "gates": gates, "verdict": verdict,
    "live_md5_before": live_before, "live_md5_after": live_after,
    "v1test_sha256": test_after,
}
with open(os.path.join(PROJ, "models", "nak_detector_v1_retrain_meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
with open(os.path.join(PROJ, "data", "phase5_baseline", "v1_retrain_locked_test_eval.json"), "w") as f:
    json.dump({"metrics": meta["locked_test"], "rows": rows,
               "star": meta["star_band"], "gates": gates,
               "verdict": verdict}, f, indent=1)
log(f"DONE verdict={verdict} [{time.time()-t0:.0f}s]")
