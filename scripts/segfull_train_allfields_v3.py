#!/usr/bin/env python3
"""Segment-level training: allfields_v3.

v3 FIXES v2's residual overfitting: v2 at 1.5x/2x still had human FPR 50%
because the model saw ChatGPT-technical but never HUMAN-technical in the
same domains (root cause per v2 eval report).

v3 changes:
  - 80 all-field AI samples at 1x (NOT 1.5x)
  - verified petroleum paper at 1x (NOT 2x)
  - NEW: 60 human-technical matched samples at 1x (same domains, human voice)
  - 4,158 arXiv human papers retained as counterbalance

Everything else identical to v1: production-scaled features (187 dim),
single LogisticRegression, drop-in replacement format.

Outputs:
  models/candidates/allfields_v3.joblib   (candidate classifier ONLY)
  models/candidates/allfields_v3_feats/    (feature cache)
  models/candidates/allfields_v3_meta.json (training metadata)

DOES NOT touch the live model.
"""
import json, os, sys, random, glob, time
PROJ = os.path.expanduser("~/workspace/ai-detector")
sys.path.insert(0, PROJ)
import joblib, numpy as np
from src.features import extract_features, feature_names, set_lm
from app.scan import split_sentences
from app.parse import parse_document
from pathlib import Path
from sklearn.linear_model import LogisticRegression

t0 = time.time()
SEED = 20261016
random.seed(SEED); np.random.seed(SEED)

# LM + production scaler come from the same artifacts bundle
art = joblib.load(os.path.join(PROJ, "models", "feature_artifacts.joblib"))
set_lm(art["lm"])
# warm up schema (registers feature names + scaler)
_ = extract_features("Warm-up sentence for schema registration.")
DIM = len(feature_names())
print(f"feature dim: {DIM}", flush=True)

DATA = os.path.join(PROJ, "data")
OUT = os.path.join(PROJ, "models", "candidates")
FEATS = os.path.join(OUT, "allfields_v3_feats")
os.makedirs(FEATS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


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


def _label_to_int(lab):
    if isinstance(lab, int):
        return lab
    s = str(lab).lower().strip()
    if s in ("ai", "1", "true", "yes"):
        return 1
    if s in ("human", "0", "false", "no"):
        return 0
    raise ValueError(f"unknown label: {lab!r}")


def load_jsonl_int(path):
    """Load (texts, int labels). Handles int or 'ai'/'human' string labels."""
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            texts.append(o["text"])
            labels.append(_label_to_int(o["label"]))
    return texts, labels


def load_jsonl_ai(path, upweight):
    """Load AI docs (label 'ai' string or 1), upweighted."""
    texts = []
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            lab = o.get("label", "ai")
            is_ai = (lab == 1) or (str(lab).lower() == "ai")
            assert is_ai, f"non-AI label in {path}: {lab!r}"
            for _ in range(upweight):
                texts.append(o["text"])
            n += 1
    return texts, n


# ---- base corpus (same as v5) ----
AUG = ("real_human_train.jsonl", "wiki_full_train.jsonl", "pubmed_train.jsonl",
       "wiki_short_train.jsonl", "arxiv_train.jsonl", "europepmc_train.jsonl",
       "oa_sub_train.jsonl", "round_thesis_train.jsonl", "round_polish_train.jsonl",
       "round_nonprose_train.jsonl")
tr_texts, tr_y = load_jsonl_int(os.path.join(DATA, "train.jsonl"))
for aug in AUG:
    p = os.path.join(DATA, aug)
    if os.path.exists(p):
        t, y = load_jsonl_int(p)
        tr_texts += t
        tr_y += y
t, y = load_jsonl_int(os.path.join(DATA, "real_ai", "real_ai_train.jsonl"))
tr_texts += t
tr_y += y
print(f"base docs: {len(tr_texts)}", flush=True)

# ---- family files 5x (v5 convention) ----
for _fn in ["family_deepseek_train.jsonl", "family_claude_train.jsonl",
            "family_multistyle_train.jsonl"]:
    _p = os.path.join(DATA, _fn)
    if os.path.exists(_p):
        _texts, _n = load_jsonl_ai(_p, 5)
        tr_texts += _texts
        tr_y = np.concatenate([tr_y, np.ones(len(_texts), dtype=int)])
        print(f"family 5x: {_fn} {_n} docs -> {len(_texts)} copies", flush=True)

# ---- style samples 5x (v5 convention) ----
for _sd in ["chatgpt_styles", "claude_styles", "deepseek_styles", "gemini_styles"]:
    _sp = os.path.join(DATA, _sd, "samples.jsonl")
    if os.path.exists(_sp):
        _texts, _n = load_jsonl_ai(_sp, 5)
        tr_texts += _texts
        tr_y = np.concatenate([tr_y, np.ones(len(_texts), dtype=int)])
        print(f"style 5x: {_sd} {_n} docs -> {len(_texts)} copies", flush=True)

# ---- NEW 2026-10-16: 80 all-field ChatGPT technical samples, 1x ----
# NOTE (v3): 1x. v2's 1.5x still overfit (FPR 50%) - the fix is matched
# human-technical data, not more weight.
_new_texts, _new_n = load_jsonl_ai(os.path.join(DATA, "chatgpt_allfields_80.jsonl"), 1)
tr_texts += _new_texts
tr_y = np.concatenate([tr_y, np.ones(len(_new_texts), dtype=int)])
print(f"NEW allfields 1x: {_new_n} docs -> {len(_new_texts)} copies", flush=True)

# ---- NEW 2026-10-16: 60 human-technical matched samples, 1x (THE v3 FIX) ----
# Same domains as the AI samples, written in human voice. These matched pairs
# teach the true AI-vs-human distinction within technical writing.
_ht_texts, _ht_y = load_jsonl_int(os.path.join(DATA, "human_technical_matched.jsonl"))
assert all(v == 0 for v in _ht_y), "human_technical_matched must be all-human"
tr_texts += _ht_texts
tr_y = np.concatenate([tr_y, np.zeros(len(_ht_texts), dtype=int)])
print(f"NEW human-technical matched 1x: {len(_ht_texts)} docs", flush=True)

# ---- NEW 2026-10-16: verified ChatGPT petroleum paper, 1x ----
_vp, _vn = load_jsonl_ai(os.path.join(DATA, "family_chatgpt_train.jsonl"), 1)
tr_texts += _vp
tr_y = np.concatenate([tr_y, np.ones(len(_vp), dtype=int)])
print(f"NEW verified-petroleum 1x: {_vn} docs -> {len(_vp)} copies", flush=True)

# ---- verified AI anchors 3x (v5 convention) ----
for _f in ["/home/hatch/workspace/user/files/Critical_Reflection.docx",
           "/tmp/bulk2/99141.docx"]:
    _p = Path(_f)
    if _p.exists():
        _txt = " ".join(u["text"] for u in parse_document(_p, _p.suffix.lower()) if u["tag"] == "p")
        tr_texts += [_txt] * 3
        tr_y = np.concatenate([tr_y, np.ones(3, dtype=int)])
        print(f"added AI anchor: {_p.name}", flush=True)

# ---- verified human anchors 3x (v5 convention) ----
for _f in ["/home/hatch/workspace/user/files/318.docx",
           "/home/hatch/workspace/user/files/7163_A1_ing.docx",
           "/tmp/bulk2/Accounting - #9143.docx",
           "/tmp/bulk2/ECON252 - #14679.docx"]:
    _p = Path(_f)
    if _p.exists():
        _txt = " ".join(u["text"] for u in parse_document(_p, _p.suffix.lower()) if u["tag"] == "p")
        tr_texts += [_txt] * 3
        tr_y = np.concatenate([tr_y, np.zeros(3, dtype=int)])
        print(f"added human anchor: {_p.name}", flush=True)

tr_y = np.array(tr_y)
print(f"docs: {len(tr_texts)} pos={int(tr_y.sum())} neg={int((tr_y==0).sum())} [{time.time()-t0:.0f}s]", flush=True)

# ---- segment ----
ai_segs, hu_segs = [], []
for tx, yy in zip(tr_texts, tr_y):
    segs = windows(tx)
    if yy == 1:
        ai_segs.extend(segs)
    else:
        hu_segs.extend(segs)
print(f"segments raw: AI={len(ai_segs)} human={len(hu_segs)} [{time.time()-t0:.0f}s]", flush=True)
random.shuffle(hu_segs)
hu_segs = hu_segs[:3 * len(ai_segs)]
seg_texts = ai_segs + hu_segs
seg_y = np.array([1] * len(ai_segs) + [0] * len(hu_segs))
idx = list(range(len(seg_texts)))
random.shuffle(idx)
seg_texts = [seg_texts[i] for i in idx]
seg_y = seg_y[idx]
print(f"segments: {len(seg_texts)} pos={int(seg_y.sum())} [{time.time()-t0:.0f}s]", flush=True)

# ---- features (production-scaled), cached ----
CHUNK = 5000
n = len(seg_texts)
n_chunks = (n + CHUNK - 1) // CHUNK
for i in range(n_chunks):
    ckpt = os.path.join(FEATS, f"chunk_{i:03d}.npz")
    if os.path.exists(ckpt):
        print(f"  chunk {i+1}/{n_chunks} cached", flush=True)
        continue
    s = slice(i * CHUNK, min((i + 1) * CHUNK, n))
    X = np.stack([extract_features(t) for t in seg_texts[s]])
    assert X.shape[1] == DIM, f"dim mismatch {X.shape[1]} != {DIM}"
    np.savez_compressed(ckpt, X=X, y=seg_y[s])
    print(f"  chunk {i+1}/{n_chunks} {X.shape} [{time.time()-t0:.0f}s]", flush=True)

Xs, ys = [], []
for ckpt in sorted(glob.glob(os.path.join(FEATS, "chunk_*.npz"))):
    d = np.load(ckpt)
    Xs.append(d["X"])
    ys.append(d["y"])
X = np.vstack(Xs)
y = np.concatenate(ys)
print(f"X={X.shape} [{time.time()-t0:.0f}s]", flush=True)

# ---- train (same hyperparams as v5) ----
clf = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1,
                         class_weight="balanced", random_state=SEED)
clf.fit(X, y)
out_model = os.path.join(OUT, "allfields_v3.joblib")
joblib.dump(clf, out_model)
print(f"SAVED {out_model}", flush=True)

# ---- train accuracy (sanity) ----
tr_pred = clf.predict(X)
tr_acc = float((tr_pred == y).mean())
print(f"train acc: {tr_acc:.4f} [{time.time()-t0:.0f}s]", flush=True)

meta = {
    "seed": SEED,
    "feature_dim": DIM,
    "n_docs": len(tr_texts),
    "n_pos_docs": int(tr_y.sum()),
    "n_segments": len(seg_texts),
    "n_pos_segments": int(seg_y.sum()),
    "train_acc": tr_acc,
    "new_allfields_docs": _new_n,
    "new_petroleum_docs": _vn,
    "elapsed_s": round(time.time() - t0, 1),
    "note": "production-scaled features; drop-in replacement format for nak_detector.joblib",
}
with open(os.path.join(OUT, "allfields_v3_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("META:", json.dumps(meta), flush=True)
print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
