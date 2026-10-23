#!/usr/bin/env python3
"""Segment-level training: oct16_newanchors.

v6 = v4 mix + Nisar's 36-PDF batch (2026-10-16) as verified anchors:
  - 8 NEW AI anchors (TT 24-58%) at 1x, label 1
    (mixed docs, deliberately BELOW v4's 1.5x verified-petroleum weight)
  - 9 NEW human anchors (TT 0%) at 2x, label 0
    (human technical counterbalance; model currently false-alarms 25-48% on 4 of them)
  - 14 *-band docs: EXCLUDED from training (no hard label possible), eval-only
  - 23626/23626__1___1 are duplicates (94% overlap): only 23626 in training,
    the other eval-only
  - 4 similarity-only PDFs excluded entirely

Allfields overweighting lesson: NEVER >2x. Candidate is drop-in replacement
format (production-scaled 187-dim features). DOES NOT touch the live model.

v4 FIXES v3's regression: v3 dropped allfields to 1x and Full Paper got 41%
(worse than v2's 66%). v4 rebalances with REAL verified human data.

v4 changes vs v2/v3:
  - 80 all-field AI samples at 1x (NOT 1.5x/5x)
  - verified petroleum paper at 1.5x (NOT 2x/10x)
  - 60 human_technical_matched at 1x (counterbalance)
  - 10 Drive human samples at 2x (REAL verified student writing)
  - 4,158 arXiv human papers retained

Everything else identical to v2: production-scaled features (187 dim),
single LogisticRegression, drop-in replacement format.

Outputs:
  models/candidates/allfields_v4.joblib   (candidate classifier ONLY)
  models/candidates/allfields_v4_feats/    (feature cache)
  models/candidates/allfields_v4_meta.json (training metadata)

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
SEED = 20261017
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
FEATS = os.path.join(OUT, "naxtrain_oct16_feats")
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


def load_jsonl_human(path, upweight):
    """Load human docs, upweighted."""
    texts = []
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            for _ in range(upweight):
                texts.append(o["text"])
            n += 1
    return texts, n


# ---- base corpus (same as v2) ----
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

# ---- v4: 80 all-field ChatGPT samples at 1x (NOT 1.5x/5x) ----
_new_texts, _new_n = load_jsonl_ai(os.path.join(DATA, "chatgpt_allfields_80.jsonl"), 1)
tr_texts += _new_texts
tr_y = np.concatenate([tr_y, np.ones(len(_new_texts), dtype=int)])
print(f"v4 allfields 1x: {_new_n} docs -> {len(_new_texts)} copies", flush=True)

# ---- v4: 60 human technical matched at 1x (NEW counterbalance) ----
_hm_texts, _hm_n = load_jsonl_human(os.path.join(DATA, "human_technical_matched.jsonl"), 1)
tr_texts += _hm_texts
tr_y = np.concatenate([tr_y, np.zeros(len(_hm_texts), dtype=int)])
print(f"v4 human-matched 1x: {_hm_n} docs -> {len(_hm_texts)} copies", flush=True)

# ---- v4: 10 Drive human samples at 2x (REAL verified student writing) ----
_dh_texts, _dh_n = load_jsonl_human(os.path.join(DATA, "drive_human_samples.jsonl"), 2)
tr_texts += _dh_texts
tr_y = np.concatenate([tr_y, np.zeros(len(_dh_texts), dtype=int)])
print(f"v4 drive-human 2x: {_dh_n} docs -> {len(_dh_texts)} copies", flush=True)

# ---- v4: verified ChatGPT petroleum paper at 1.5x (NOT 2x/10x) ----
# load_jsonl_ai only supports int upweight; do 1x then add 50% extra.
_vp_1x, _vn = load_jsonl_ai(os.path.join(DATA, "family_chatgpt_train.jsonl"), 1)
_vp_extra = random.sample(_vp_1x, max(1, len(_vp_1x) // 2))
_vp = _vp_1x + _vp_extra
tr_texts += _vp
tr_y = np.concatenate([tr_y, np.ones(len(_vp), dtype=int)])
print(f"v4 verified-petroleum 1.5x: {_vn} docs -> {len(_vp)} copies", flush=True)

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


# ---- NEW 2026-10-16+: Nisar's verified Turnitin anchors (batch now 44 PDFs) ----
# Single source of truth: data/anchors_new_oct16/anchors_oct16_manifest.json
# Modest weights only (allfields lesson): AI-class at 1x, human-class at 2x.
# *%-band docs and duplicates are train_include=false (eval-only, no hard label).
_man = json.load(open(os.path.join(DATA, "anchors_new_oct16",
                                   "anchors_oct16_manifest.json")))
_added_ai = _added_hu = 0
for _slug, _m in sorted(_man.items()):
    if not _m.get("train_include"):
        continue
    _p = Path(os.path.join(DATA, "anchors_new_oct16", _slug + ".txt"))
    if not _p.exists():
        print(f"ANCHOR TEXT MISSING: {_slug}", flush=True); continue
    _txt = _p.read_text(encoding="utf-8")
    if not _txt.strip():
        print(f"ANCHOR TEXT EMPTY: {_slug}", flush=True); continue
    if _m["class"] == "ai":
        tr_texts += [_txt]          # 1x, label 1
        tr_y = np.concatenate([tr_y, np.ones(1, dtype=int)])
        _added_ai += 1
        print(f"added NEW AI anchor (1x): {_slug}", flush=True)
    elif _m["class"] == "human":
        tr_texts += [_txt] * 2      # 2x, label 0 (technical counterbalance)
        tr_y = np.concatenate([tr_y, np.zeros(2, dtype=int)])
        _added_hu += 1
        print(f"added NEW human anchor (2x): {_slug}", flush=True)
print(f"new anchors added: AI={_added_ai} HUMAN={_added_hu}", flush=True)

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
out_model = os.path.join(OUT, "naxtrain_oct16.joblib")
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
    "v4_allfields_docs": _new_n,
    "v4_human_matched_docs": _hm_n,
    "v4_drive_human_docs": _dh_n,
    "v4_petroleum_docs": _vn,
    "oct16_new_ai_anchors": _added_ai,
    "oct16_new_human_anchors": _added_hu,
    "elapsed_s": round(time.time() - t0, 1),
    "note": "oct16_newanchors: v4 mix + new TT-verified AI anchors 1x + human anchors 2x; *%-band & dups eval-only; production-scaled; drop-in format",
}
with open(os.path.join(OUT, "naxtrain_oct16_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("META:", json.dumps(meta), flush=True)
print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
