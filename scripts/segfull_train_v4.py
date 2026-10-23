#!/usr/bin/env python3
"""Segment-level training v2: includes verified human anchors (TT 0%).
Outputs to ~/workspace/ai-detector/models/segments/ (persistent)."""
import json, os, sys, random, glob, time
PROJ = os.path.expanduser("~/workspace/ai-detector")
sys.path.insert(0, PROJ); sys.path.insert(0, os.path.join(PROJ, "scripts"))
import joblib, numpy as np
from src.features import extract_features_unscaled, feature_names, set_lm
from app.scan import split_sentences
from app.parse import parse_document
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

t0=time.time()
SEED=20261012
random.seed(SEED)
art = joblib.load(os.path.join(PROJ, "models", "feature_artifacts.joblib"))
set_lm(art["lm"])
DATA = os.path.join(PROJ, "data")
OUT = os.path.join(PROJ, "models", "segments_v3")
FEATS = os.path.join(OUT, "feats")
os.makedirs(FEATS, exist_ok=True)

def windows(text, w=8, s=4, min_words=20):
    sents = split_sentences(text) or [text]
    out = []
    if len(sents) < 3:
        return []
    for i in range(0, max(1, len(sents)-w+1), s):
        seg = " ".join(sents[i:i+w])
        if len(seg.split()) >= min_words:
            out.append(seg)
    return out

def load_jsonl(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            o=json.loads(line)
            texts.append(o["text"]); labels.append(int(o["label"]))
    return texts, labels

AUG = ("real_human_train.jsonl","wiki_full_train.jsonl","pubmed_train.jsonl",
       "wiki_short_train.jsonl","arxiv_train.jsonl","europepmc_train.jsonl",
       "oa_sub_train.jsonl","round_thesis_train.jsonl","round_polish_train.jsonl",
       "round_nonprose_train.jsonl")  # family files handled separately with register
tr_texts, tr_y = load_jsonl(os.path.join(DATA,"train.jsonl"))
for aug in AUG:
    p=os.path.join(DATA,aug)
    if os.path.exists(p):
        t,y=load_jsonl(p); tr_texts+=t; tr_y+=y
t,y=load_jsonl(os.path.join(DATA,"real_ai","real_ai_train.jsonl"))
tr_texts+=t; tr_y+=y
# family files with register tracking for style balancing
import collections as _co
_fam_regs = []  # parallel to tr_texts for family docs
_reg_counts = _co.Counter()
for _fn in ["family_deepseek_train.jsonl","family_claude_train.jsonl",
           "family_multistyle_train_v5.jsonl"]:
    _p=os.path.join(DATA,_fn)
    if os.path.exists(_p):
        for _line in open(_p):
            _o=json.loads(_line)
            _r=_o.get("register","general")
            _reg_counts[_r]+=1
            # 5x base, plus boost for rare styles (<5 samples -> up to 10x)
            _mult = 5 if _reg_counts[_r]>=0 else 5
            for _ in range(5):
                tr_texts.append(_o["text"]); _fam_regs.append(_r)
        # y for family (all label 1)
        _n = sum(1 for _ in open(_p))
        tr_y = np.concatenate([tr_y, np.ones(_n*5, dtype=int)])
print(f"family registers: {dict(_reg_counts)}", flush=True)
# verified HIGH-AI anchors as AI training (TT 99/64) — 3x each; MA114/XDR/Listeria/Composites held out
for _f in ["/home/hatch/workspace/user/files/Critical_Reflection.docx",
           "/tmp/bulk2/99141.docx"]:
    _p=Path(_f)
    if _p.exists():
        _txt=" ".join(u["text"] for u in parse_document(_p,_p.suffix.lower()) if u["tag"]=="p")
        tr_texts += [_txt]*3
        tr_y = np.concatenate([tr_y, np.ones(3,dtype=int)])
        print(f"added AI anchor: {_p.name}", flush=True)
# verified human anchors (TT 0%) — 3x each
for _f in ["/home/hatch/workspace/user/files/318.docx",
           "/home/hatch/workspace/user/files/7163_A1_ing.docx",
           "/tmp/bulk2/Accounting - #9143.docx",
           "/tmp/bulk2/ECON252 - #14679.docx"]:
    _p=Path(_f)
    if _p.exists():
        _txt=" ".join(u["text"] for u in parse_document(_p,_p.suffix.lower()) if u["tag"]=="p")
        tr_texts += [_txt]*3
        tr_y = np.concatenate([tr_y, np.zeros(3,dtype=int)])
        print(f"added human anchor: {_p.name}", flush=True)
tr_y=np.array(tr_y)
print(f"docs: {len(tr_texts)} pos={tr_y.sum()}", flush=True)

ai_segs, hu_segs = [], []
# build register map for AI docs (family docs are at the end)
_n_fam = len(_fam_regs)
for idx2, (tx, yy) in enumerate(zip(tr_texts, tr_y)):
    segs = windows(tx)
    if yy==1:
        # determine register if family doc
        _r = None
        if idx2 >= len(tr_texts)-_n_fam:
            _r = _fam_regs[idx2-(len(tr_texts)-_n_fam)]
        ai_segs.extend(segs)
        # style boost: rare registers (<5 docs) get 3x segment copies
        if _r and _reg_counts[_r] < 5:
            ai_segs.extend(segs*2)
    else: hu_segs.extend(segs)
print(f"segments raw: AI={len(ai_segs)} human={len(hu_segs)} [{time.time()-t0:.0f}s]", flush=True)
random.shuffle(hu_segs)
hu_segs = hu_segs[:3*len(ai_segs)]
seg_texts = ai_segs + hu_segs
seg_y = np.array([1]*len(ai_segs)+[0]*len(hu_segs))
idx = list(range(len(seg_texts))); random.shuffle(idx)
seg_texts=[seg_texts[i] for i in idx]; seg_y=seg_y[idx]
print(f"segments: {len(seg_texts)} pos={seg_y.sum()} [{time.time()-t0:.0f}s]", flush=True)

CHUNK=5000; n=len(seg_texts); dim=len(feature_names())
n_chunks=(n+CHUNK-1)//CHUNK
for i in range(n_chunks):
    ckpt=os.path.join(FEATS,f"chunk_{i:03d}.npz")
    if os.path.exists(ckpt):
        print(f"  chunk {i+1}/{n_chunks} cached", flush=True); continue
    s=slice(i*CHUNK,min((i+1)*CHUNK,n))
    X=np.stack([extract_features_unscaled(t) for t in seg_texts[s]])
    np.savez_compressed(ckpt,X=X,y=seg_y[s])
    print(f"  chunk {i+1}/{n_chunks} {X.shape} [{time.time()-t0:.0f}s]", flush=True)
Xs,ys=[],[]
for ckpt in sorted(glob.glob(os.path.join(FEATS,"chunk_*.npz"))):
    d=np.load(ckpt); Xs.append(d["X"]); ys.append(d["y"])
X=np.vstack(Xs); y=np.concatenate(ys)
print(f"X={X.shape} [{time.time()-t0:.0f}s]", flush=True)
sc=StandardScaler().fit(X)
clf=LogisticRegression(max_iter=3000,C=1.0,n_jobs=-1,class_weight="balanced",random_state=SEED)
clf.fit(sc.transform(X),y)
joblib.dump(clf,os.path.join(OUT,"seg_candidate_v4.joblib"))
joblib.dump(sc,os.path.join(OUT,"seg_scaler_v4.joblib"))
print(f"SAVED v2 [{time.time()-t0:.0f}s]", flush=True)
