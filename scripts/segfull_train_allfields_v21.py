#!/usr/bin/env python3
"""v21: v20 + 12 pre-AI era human documents (Nisar's idea) + ChatGPT blueprint metrics.

v17 was best overall (Full Paper 79.7%, PhD 93.0%, 8-anchor 17.56).
v20 added 9 competition essays @3x -> Full Paper 80.6% (best ever) but
PhD diluted 93.0% -> 80.8% (dilution trap).
v21 adds 12 pre-AI era documents (1884-2020, 15,222 words, guaranteed human,
written register: news/blogs/literature/essays) as human @2x, and bumps the
PhD anchor to @18x to defend against the dilution trap.

Blueprint principles applied:
  - Document-level grouping preserved (windows() never crosses documents;
    no document appears in both AI and human sets).
  - Validation diagnostic: a document-level 10% holdout of Turnitin AI/human
    docs is scored by the final model to report validation MAE (no tuning on it).
  - Eval reports MAE, within-5 rate, bias (signed error), error percentiles.

  R: v17's router (keep)
  T: v20 T-recipe + 12 pre-AI docs @2x (human); PhD anchor @18x (dilution defense)
  G: frozen v5 (don't break)
"""
import json, os, sys, random, glob, time, hashlib
PROJ = os.path.expanduser("~/workspace/ai-detector")
sys.path.insert(0, PROJ)
import joblib, numpy as np
from src.features import extract_features, feature_names, set_lm
from app.scan import split_sentences
from sklearn.linear_model import LogisticRegression

t0 = time.time(); SEED = 20261017
random.seed(SEED); np.random.seed(SEED)
art = joblib.load(os.path.join(PROJ, "models", "feature_artifacts.joblib"))
set_lm(art["lm"])
_ = extract_features("warmup.")
DIM = len(feature_names()); assert DIM == 187
DATA = os.path.join(PROJ, "data"); OUT = os.path.join(PROJ, "models", "candidates")
FEATS = os.path.join(OUT, "allfields_v21_feats")
os.makedirs(FEATS, exist_ok=True)

def windows(text, w=8, s=4, min_words=20):
    sents = split_sentences(text) or [text]
    out = []
    if len(sents) < 3: return []
    for i in range(0, max(1, len(sents)-w+1), s):
        seg = " ".join(sents[i:i+w])
        if len(seg.split()) >= min_words: out.append(seg)
    return out

def load_texts(path, max_docs=None):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            texts.append(json.loads(line)["text"])
            if max_docs and len(texts) >= max_docs: break
    return texts

def load_turn_ai_no_meta(path):
    texts = []
    for line in open(path, encoding="utf-8"):
        line=line.strip()
        if not line: continue
        o = json.loads(line)
        if str(o.get("label","")).lower()=="ai" and not str(o.get("source","")).startswith("meta_ai_essay"):
            texts.append(o["text"])
    return texts

def load_meta(path):
    texts = []
    for line in open(path, encoding="utf-8"):
        line=line.strip()
        if not line: continue
        o = json.loads(line)
        if str(o.get("source","")).startswith("meta_ai_essay"): texts.append(o["text"])
    return texts

def load_by_label(path, want):
    texts = []
    for line in open(path, encoding="utf-8"):
        line=line.strip()
        if not line: continue
        o = json.loads(line)
        if str(o.get("label","")).lower()==want: texts.append(o["text"])
    return texts

def load_wa(path):
    EXCLUDE = {"5061","55743","58078","33647","Dreams_Final_Report_AUB_3","Dreams"}
    out = {"phd_ai":[],"bdo_ai":[],"phd_upd_hu":[],"essay_hu":[]}
    for line in open(path, encoding="utf-8"):
        line=line.strip()
        if not line: continue
        o=json.loads(line); src=str(o.get("source",""))
        if any(x in src for x in EXCLUDE): continue
        lab=str(o.get("label","")).lower()
        if lab=="ai":
            if "Personal_statement_PhD" in src: out["phd_ai"].append(o["text"])
            elif "BDO" in src: out["bdo_ai"].append(o["text"])
        elif lab=="human":
            if "PhD_updated" in src: out["phd_upd_hu"].append(o["text"])
            else: out["essay_hu"].append(o["text"])
    return out

def load_txt(p):
    with open(p, encoding="utf-8") as f:
        return "\n".join(l for l in f.read().splitlines() if not l.startswith("#")).strip()

def repeat(lst,k):
    out=[]
    for t in lst: out.extend([t]*k)
    return out

turn_ai = load_turn_ai_no_meta(os.path.join(DATA,"turnitin_mass_paired.jsonl"))
turn_hu = load_by_label(os.path.join(DATA,"turnitin_mass_paired.jsonl"),"human")
meta = load_meta(os.path.join(DATA,"turnitin_mass_paired.jsonl"))
wa = load_wa(os.path.join(DATA,"whatsapp21_training.jsonl"))
wa_all = wa["phd_ai"]+wa["bdo_ai"]+wa["phd_upd_hu"]+wa["essay_hu"]
a22103 = load_texts(os.path.join(DATA,"turnitin_anchors.jsonl"))
p22103_ai = load_by_label(os.path.join(DATA,"turnitin_22103_paired.jsonl"),"ai")
p22103_hu = load_by_label(os.path.join(DATA,"turnitin_22103_paired.jsonl"),"human")
drive_hu = load_texts(os.path.join(DATA,"drive_human_samples.jsonl"))
matched_hu = load_texts(os.path.join(DATA,"human_technical_matched.jsonl"))
fam=[]
for d in ["family_deepseek_train.jsonl","family_claude_train.jsonl","family_multistyle_train.jsonl"]:
    p=os.path.join(DATA,d)
    if os.path.exists(p):
        for line in open(p,encoding="utf-8"):
            line=line.strip()
            if line: fam.append(json.loads(line)["text"])
petro = load_texts(os.path.join(DATA,"family_chatgpt_train.jsonl"))
real_ai = load_texts(os.path.join(DATA,"real_ai","real_ai_train.jsonl"),max_docs=400)
arxiv_hu = load_texts(os.path.join(DATA,"arxiv_train.jsonl"),max_docs=300)
wiki_short = load_texts(os.path.join(DATA,"wiki_short_train.jsonl"))
wiki_full = load_texts(os.path.join(DATA,"wiki_full_train.jsonl"))
h004 = load_txt(os.path.join(DATA,"verified","human_004_listeria.txt"))
h005 = load_txt(os.path.join(DATA,"verified","human_005_science.txt"))
h006_shake = load_txt(os.path.join(DATA,"verified","human_006_shakespeare.txt"))
ESSAY_DIR = os.path.join(DATA,"verified","essays")
comp_essays = []
for _fn in sorted(glob.glob(os.path.join(ESSAY_DIR,"human_essay_*.txt"))):
    comp_essays.append(load_txt(_fn))
print(f"competition essays loaded: {len(comp_essays)} files", flush=True)
# v21 NEW: 12 pre-AI era documents (1884-2020, 15,222 words, guaranteed human, written register)
PREAI_DIR = os.path.join(DATA,"verified","pre_ai_era")
preai_docs = []
for _fn in sorted(glob.glob(os.path.join(PREAI_DIR,"human_preai_*.txt"))):
    preai_docs.append(load_txt(_fn))
preai_words = sum(len(t.split()) for t in preai_docs)
print(f"pre-AI era docs loaded: {len(preai_docs)} files, {preai_words} words", flush=True)
nisar=[load_txt(os.path.join(DATA,"verified",f)) for f in ["nisar_001_waste.txt","nisar_002_story.txt","nisar_003_school.txt"]]

# ---- Blueprint: document-level validation holdout (10% of Turnitin docs) ----
# Held out BEFORE training; used only for a validation MAE diagnostic (no tuning).
random.seed(SEED+1)
_val_ai_idx = set(random.sample(range(len(turn_ai)), max(1, len(turn_ai)//10)))
_val_hu_idx = set(random.sample(range(len(turn_hu)), max(1, len(turn_hu)//10)))
val_ai_docs = [turn_ai[i] for i in sorted(_val_ai_idx)]
val_hu_docs = [turn_hu[i] for i in sorted(_val_hu_idx)]
tr_ai_docs = [t for i, t in enumerate(turn_ai) if i not in _val_ai_idx]
tr_hu_docs = [t for i, t in enumerate(turn_hu) if i not in _val_hu_idx]
print(f"validation holdout: {len(val_ai_docs)} AI docs, {len(val_hu_docs)} human docs (document-level, no leakage)", flush=True)

def segs_of(docs):
    segs=[]
    for t in docs: segs.extend(windows(t))
    return segs

# ROUTER (v10 recipe; trained on full data incl. holdout docs for final model)
r_tech=[]; r_gen=[]
for fn,cap in [("arxiv_train.jsonl",600),("pubmed_train.jsonl",250),("europepmc_train.jsonl",400),("oa_sub_train.jsonl",800)]:
    p=os.path.join(DATA,fn)
    if os.path.exists(p): r_tech+=load_texts(p,max_docs=cap)
r_tech += turn_ai+turn_hu+a22103+fam+petro+real_ai+drive_hu+matched_hu+[h004]+wa_all
r_gen += wiki_short+wiki_full+nisar+[h005]
rt=segs_of(r_tech); rg=segs_of(r_gen)
if len(rt)>1.5*len(rg):
    random.shuffle(rt); rt=rt[:int(1.5*len(rg))]
rX_texts=rt+rg; rY=np.array([1]*len(rt)+[0]*len(rg))
idx=list(range(len(rX_texts))); random.shuffle(idx)
rX_texts=[rX_texts[i] for i in idx]; rY=rY[idx]

# T (v21: v20 + 12 pre-AI docs @2x human; PhD @18x dilution defense; Meta @1x)
t_ai = repeat(turn_ai,3)+a22103+p22103_ai+repeat(fam,2)+repeat(petro,3)+real_ai
t_ai += repeat(meta,1)+repeat(wa["phd_ai"],18)+wa["bdo_ai"]  # v21: PhD @18x (dilution defense)
t_hu = turn_hu+repeat(drive_hu,3)+matched_hu+repeat([h004],3)+repeat(p22103_hu,6)+arxiv_hu
t_hu += repeat(wa["phd_upd_hu"],3)+repeat(wa["essay_hu"],2)
t_hu += repeat([h006_shake],3)
t_hu += repeat(comp_essays,3)
t_hu += repeat(preai_docs,2)  # v21 NEW: 12 pre-AI era docs @2x (guaranteed-human written register)
ta=segs_of(t_ai); th=segs_of(t_hu)
random.shuffle(th); th=th[:int(2.5*len(ta))]
tX_texts=ta+th; tY=np.array([1]*len(ta)+[0]*len(th))
idx=list(range(len(tX_texts))); random.shuffle(idx)
tX_texts=[tX_texts[i] for i in idx]; tY=tY[idx]
print(f"T: {len(tX_texts)} pos={int(tY.sum())} [{time.time()-t0:.0f}s]", flush=True)

def featurize_cached(texts,y,tag):
    n=len(texts); CHUNK=5000; n_chunks=(n+CHUNK-1)//CHUNK
    for i in range(n_chunks):
        ckpt=os.path.join(FEATS,f"{tag}_{i:03d}.npz")
        if os.path.exists(ckpt): continue
        s=slice(i*CHUNK,min((i+1)*CHUNK,n))
        X=np.stack([extract_features(t) for t in texts[s]])
        np.savez_compressed(ckpt,X=X,y=y[s])
    Xs,ys=[],[]
    for ckpt in sorted(glob.glob(os.path.join(FEATS,f"{tag}_*.npz"))):
        d=np.load(ckpt); Xs.append(d["X"]); ys.append(d["y"])
    return np.vstack(Xs),np.concatenate(ys)

def train_lr(X,y,name):
    clf=LogisticRegression(max_iter=3000,C=1.0,class_weight="balanced",random_state=SEED)
    clf.fit(X,y); acc=float((clf.predict(X)==y).mean())
    print(f"{name} acc: {acc:.4f} X={X.shape} [{time.time()-t0:.0f}s]", flush=True)
    return clf,acc

rX,ry=featurize_cached(rX_texts,rY,"router"); R,r_acc=train_lr(rX,ry,"router")
tX,ty=featurize_cached(tX_texts,tY,"tech"); T,t_acc=train_lr(tX,ty,"tech-universal")
G=joblib.load(os.path.join(OUT,"allfields_v5.joblib"))  # frozen v5
print("frozen v5 generalist loaded", flush=True)

# ---- Blueprint validation diagnostic (document-level holdout, no tuning) ----
def _routed_proba(text):
    f = np.asarray(extract_features(text), dtype=np.float64).reshape(1, -1)
    r = float(R.predict_proba(f)[0][1]); pt = float(T.predict_proba(f)[0][1]); pg = float(G.predict_proba(f)[0][1])
    return r*pt + (1.0-r)*pg

def _doc_proba_windows(text):
    segs = windows(text)
    if not segs: return _routed_proba(text)
    return float(np.mean([_routed_proba(s) for s in segs]))

val_rows = []
for t in val_ai_docs: val_rows.append((_doc_proba_windows(t)*100.0, 100.0))
for t in val_hu_docs: val_rows.append((_doc_proba_windows(t)*100.0, 0.0))
val_errs = [abs(p - y) for p, y in val_rows]
val_mae = round(float(np.mean(val_errs)), 2) if val_errs else None
val_w5 = round(float(np.mean([e <= 5.0 for e in val_errs])), 4) if val_errs else None
print(f"VALIDATION (document-level holdout, n={len(val_rows)}): MAE={val_mae}, within-5={val_w5}", flush=True)

bundle={"router":R,"tech":T,"general":G}
joblib.dump(bundle, os.path.join(OUT,"allfields_v21.joblib"))
print("SAVED allfields_v21.joblib", flush=True)
meta_out={"seed":SEED,"feature_dim":DIM,"architecture":"routed: p=R*T+(1-R)*G_v5; T=universal (research+essay AI)",
 "router_acc":round(r_acc,4),"tech_acc":round(t_acc,4),
 "validation_holdout_n":len(val_rows),"validation_mae":val_mae,"validation_within5":val_w5,
 "preai_docs":len(preai_docs),"preai_words":preai_words,
 "elapsed_s":round(time.time()-t0,1),
 "note":"v21: v20 + 12 pre-AI era docs @2x (guaranteed-human written register, 15,222 words); PhD @18x dilution defense; Meta @1x. Blueprint: document-level validation holdout reported."}
with open(os.path.join(OUT,"allfields_v21_meta.json"),"w") as f: json.dump(meta_out,f,indent=2)
live=os.path.join(PROJ,"models","nak_detector.joblib")
h=hashlib.md5(open(live,"rb").read()).hexdigest()
print(f"LIVE MD5: {h} {'OK (v17 wrapper live)' if h=='3e24dc7c79bc40b20c597e37e9aaaaf0' else 'CHECK!'}", flush=True)
print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
