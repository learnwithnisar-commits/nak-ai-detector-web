#!/usr/bin/env python3
"""Segment-level training: allfields_v5.

v5 strategy: replace synthetic AI with REAL verified AI from Turnitin highlights.

v5 changes vs v4:
  - PRIMARY AI: 103 verified Turnitin AI samples (15,104 words, real, multi-domain) at 3x
  - 22103 Turnitin anchor (57% AI) at 3x; 22103 paired AI/human at 3x
  - Human: 133 Turnitin human (378K words) at 1x + 10 Drive human at 2x
            + 60 human-technical-matched at 1x + 5 verified human txts at 3x
  - Family AI (deepseek/claude/multistyle real outputs) REDUCED to 2x (was 5x)
  - EXCLUDED: chatgpt_allfields_80.jsonl (synthetic), chatgpt_styles samples,
              family_multistyle_train_v2.jsonl (115/138 chatgpt synthetic)
  - Base corpus (arXiv/Wikipedia/PubMed background) kept at 1x
  - Verified petroleum paper (family_chatgpt_train) kept at 2x

Architecture identical to v4/live: production-scaled 187-dim features,
single LogisticRegression, drop-in replacement format.

Outputs:
  models/candidates/allfields_v5.joblib
  models/candidates/allfields_v5_feats/
  models/candidates/allfields_v5_meta.json

DOES NOT touch the live model (models/nak_detector.joblib).
"""
import json, os, sys, random, glob, time
PROJ = os.path.expanduser("~/workspace/ai-detector")
sys.path.insert(0, PROJ)
import joblib, numpy as np
from src.features import extract_features, feature_names, set_lm
from app.scan import split_sentences
from sklearn.linear_model import LogisticRegression

t0 = time.time()
SEED = 20261017
random.seed(SEED); np.random.seed(SEED)

art = joblib.load(os.path.join(PROJ, "models", "feature_artifacts.joblib"))
set_lm(art["lm"])
_ = extract_features("Warm-up sentence for schema registration.")
DIM = len(feature_names())
print(f"feature dim: {DIM}", flush=True)

DATA = os.path.join(PROJ, "data")
OUT = os.path.join(PROJ, "models", "candidates")
FEATS = os.path.join(OUT, "allfields_v5_feats")
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
    texts, n = [], 0
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
    texts, n = [], 0
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


def load_txt_nocomments(path, upweight):
    """Verified .txt files: strip '#' comment header lines."""
    with open(path, encoding="utf-8") as f:
        body = "\n".join(l for l in f.read().splitlines() if not l.startswith("#"))
    return [body.strip()] * upweight


tr_texts, tr_y = [], []

# ---- 1. base corpus (arXiv/Wikipedia/PubMed background) at 1x ----
AUG = ("real_human_train.jsonl", "wiki_full_train.jsonl", "pubmed_train.jsonl",
       "wiki_short_train.jsonl", "arxiv_train.jsonl", "europepmc_train.jsonl",
       "oa_sub_train.jsonl", "round_thesis_train.jsonl", "round_polish_train.jsonl",
       "round_nonprose_train.jsonl")
b_texts, b_y = load_jsonl_int(os.path.join(DATA, "train.jsonl"))
for aug in AUG:
    p = os.path.join(DATA, aug)
    if os.path.exists(p):
        t, y = load_jsonl_int(p)
        b_texts += t
        b_y += y
t, y = load_jsonl_int(os.path.join(DATA, "real_ai", "real_ai_train.jsonl"))
b_texts += t
b_y += y
tr_texts += b_texts
tr_y = np.concatenate([np.array([], dtype=int), np.array(b_y)])
print(f"base docs: {len(b_texts)} [{time.time()-t0:.0f}s]", flush=True)

# ---- 2. family AI (real DeepSeek/Claude/multistyle outputs) REDUCED to 2x ----
fam_n = 0
for _fn in ["family_deepseek_train.jsonl", "family_claude_train.jsonl",
            "family_multistyle_train.jsonl"]:
    _p = os.path.join(DATA, _fn)
    if os.path.exists(_p):
        _texts, _n = load_jsonl_ai(_p, 2)
        tr_texts += _texts
        tr_y = np.concatenate([tr_y, np.ones(len(_texts), dtype=int)])
        fam_n += _n
print(f"v5 family AI 2x: {fam_n} docs -> {len(_texts) if fam_n else 0} copies [{time.time()-t0:.0f}s]", flush=True)

# ---- 3. non-ChatGPT style samples at 1x (real family outputs, small) ----
sty_n = 0
for _sd in ["claude_styles", "deepseek_styles", "gemini_styles"]:
    _sp = os.path.join(DATA, _sd, "samples.jsonl")
    if os.path.exists(_sp):
        _texts, _n = load_jsonl_ai(_sp, 1)
        tr_texts += _texts
        tr_y = np.concatenate([tr_y, np.ones(len(_texts), dtype=int)])
        sty_n += _n
print(f"v5 style AI 1x (no chatgpt): {sty_n} docs [{time.time()-t0:.0f}s]", flush=True)

# ---- 4. verified petroleum paper at 2x ----
_vp, _vn = load_jsonl_ai(os.path.join(DATA, "family_chatgpt_train.jsonl"), 2)
tr_texts += _vp
tr_y = np.concatenate([tr_y, np.ones(len(_vp), dtype=int)])
print(f"v5 verified-petroleum 2x: {_vn} docs -> {len(_vp)} copies", flush=True)

# ---- 5. PRIMARY AI: 103 verified Turnitin AI samples at 3x ----
# (load only label=='ai' records)
_ai_texts, _ai_n = [], 0
with open(os.path.join(DATA, "turnitin_mass_paired.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if str(o.get("label", "")).lower() == "ai":
            for _ in range(3):
                _ai_texts.append(o["text"])
            _ai_n += 1
tr_texts += _ai_texts
tr_y = np.concatenate([tr_y, np.ones(len(_ai_texts), dtype=int)])
print(f"v5 VERIFIED-AI 3x: {_ai_n} reports -> {len(_ai_texts)} copies [{time.time()-t0:.0f}s]", flush=True)
v5_ai_docs = _ai_n

# ---- 6. 22103 anchor (Turnitin 57% AI) at 3x ----
_a22103, _a22103_n = load_jsonl_ai(os.path.join(DATA, "turnitin_anchors.jsonl"), 3)
tr_texts += _a22103
tr_y = np.concatenate([tr_y, np.ones(len(_a22103), dtype=int)])
print(f"v5 22103-anchor 3x: {_a22103_n} docs -> {len(_a22103)} copies", flush=True)

# ---- 7. 22103 paired AI/human at 3x ----
_p_texts, _p_y = [], []
with open(os.path.join(DATA, "turnitin_22103_paired.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        for _ in range(3):
            _p_texts.append(o["text"])
            _p_y.append(_label_to_int(o["label"]))
tr_texts += _p_texts
tr_y = np.concatenate([tr_y, np.array(_p_y, dtype=int)])
print(f"v5 22103-paired 3x: {len(_p_y)//3} docs -> {len(_p_texts)} copies", flush=True)

# ---- 8. HUMAN: 133 Turnitin human at 1x ----
_hu_texts, _hu_n = [], 0
with open(os.path.join(DATA, "turnitin_mass_paired.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if str(o.get("label", "")).lower() != "ai":
            _hu_texts.append(o["text"])
            _hu_n += 1
tr_texts += _hu_texts
tr_y = np.concatenate([tr_y, np.zeros(len(_hu_texts), dtype=int)])
print(f"v5 turnitin-human 1x: {_hu_n} reports -> {len(_hu_texts)} copies [{time.time()-t0:.0f}s]", flush=True)

# ---- 9. HUMAN: 10 Drive human at 2x ----
_dh_texts, _dh_n = load_jsonl_human(os.path.join(DATA, "drive_human_samples.jsonl"), 2)
tr_texts += _dh_texts
tr_y = np.concatenate([tr_y, np.zeros(len(_dh_texts), dtype=int)])
print(f"v5 drive-human 2x: {_dh_n} docs -> {len(_dh_texts)} copies", flush=True)

# ---- 10. HUMAN: 60 human-technical-matched at 1x ----
_hm_texts, _hm_n = load_jsonl_human(os.path.join(DATA, "human_technical_matched.jsonl"), 1)
tr_texts += _hm_texts
tr_y = np.concatenate([tr_y, np.zeros(len(_hm_texts), dtype=int)])
print(f"v5 human-matched 1x: {_hm_n} docs -> {len(_hm_texts)} copies", flush=True)

# ---- 11. HUMAN: 5 verified txts (Nisar x3 + 2 friends) at 3x ----
_vt = 0
for _vf in ["nisar_001_waste.txt", "nisar_002_story.txt", "nisar_003_school.txt",
            "human_004_listeria.txt", "human_005_science.txt"]:
    _vp_ = os.path.join(DATA, "verified", _vf)
    if os.path.exists(_vp_):
        _copies = load_txt_nocomments(_vp_, 3)
        tr_texts += _copies
        tr_y = np.concatenate([tr_y, np.zeros(len(_copies), dtype=int)])
        _vt += 1
print(f"v5 verified-human-txt 3x: {_vt} files", flush=True)

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

# ---- train ----
clf = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1,
                         class_weight="balanced", random_state=SEED)
clf.fit(X, y)
out_model = os.path.join(OUT, "allfields_v5.joblib")
joblib.dump(clf, out_model)
print(f"SAVED {out_model}", flush=True)

tr_pred = clf.predict(X)
tr_acc = float((tr_pred == y).mean())
print(f"train acc: {tr_acc:.4f} [{time.time()-t0:.0f}s]", flush=True)

meta = {
    "seed": SEED,
    "feature_dim": DIM,
    "n_docs": len(tr_texts),
    "n_pos_docs": int(tr_y.sum()),
    "n_neg_docs": int((tr_y == 0).sum()),
    "n_segments": len(seg_texts),
    "n_pos_segments": int(seg_y.sum()),
    "train_acc": tr_acc,
    "v5_verified_ai_reports": v5_ai_docs,
    "v5_drive_human_docs": _dh_n,
    "v5_human_matched_docs": _hm_n,
    "v5_turnitin_human_reports": _hu_n,
    "v5_verified_txt_files": _vt,
    "elapsed_s": round(time.time() - t0, 1),
    "note": "v5: 103 verified Turnitin AI @3x PRIMARY; synthetic ChatGPT excluded; family 2x; Turnitin human @1x + Drive @2x + verified txt @3x",
}
with open(os.path.join(OUT, "allfields_v5_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("META:", json.dumps(meta), flush=True)
print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
