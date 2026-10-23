#!/usr/bin/env python3
"""v19: v18 + 10 YouTube human speeches (Nisar's idea).

v18 was close to v17 (Full Paper 79.6%, PhD 91.9%); Shakespeare essay
fixed h005 (50.9%->2.9%) but Wikipedia worsened.
v19 adds 10 human speeches (Trump, Obama, UN leaders, Malala, Greta,
MLK, Churchill, TED, Shehbaz Sharif; ~6.6K words) as human @2x —
diverse human voices to counterbalance Meta AI essays and fix human FPR.

  R: v18's router (keep)
  T: v18 T-recipe + 10 speeches @2x (human voice diversity)
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
FEATS = os.path.join(OUT, "allfields_v19_feats")
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
h006_shake = load_txt(os.path.join(DATA,"verified","human_006_shakespeare.txt"))  # v18: Nisar's Shakespeare essay
# v19 NEW: 10 YouTube human speeches (Nisar's idea) — diverse human voices
SPEECH_DIR = os.path.join(DATA,"verified","speeches")
speeches = []
for _fn in sorted(glob.glob(os.path.join(SPEECH_DIR,"human_speech_*.txt"))):
    speeches.append(load_txt(_fn))
print(f"speeches loaded: {len(speeches)} files", flush=True)
nisar=[load_txt(os.path.join(DATA,"verified",f)) for f in ["nisar_001_waste.txt","nisar_002_story.txt","nisar_003_school.txt"]]

def segs_of(docs):
    segs=[]
    for t in docs: segs.extend(windows(t))
    return segs

# ROUTER (v10 recipe)
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

# T (v19: v17 + Shakespeare human essay @3x for essay counterbalance)
t_ai = repeat(turn_ai,3)+a22103+p22103_ai+repeat(fam,2)+repeat(petro,3)+real_ai  # v17: 22103-AI @1x
t_ai += repeat(meta,2)+repeat(wa["phd_ai"],15)+wa["bdo_ai"]  # v17: PhD @15x (push 92.3->95+)  # v17: essay AI in T
t_hu = turn_hu+repeat(drive_hu,3)+matched_hu+repeat([h004],3)+repeat(p22103_hu,6)+arxiv_hu  # v17: 22103-HU @6x
t_hu += repeat(wa["phd_upd_hu"],3)+repeat(wa["essay_hu"],2)
t_hu += repeat([h006_shake],3)  # v18: Shakespeare human essay @3x (counterbalance Meta essays)
t_hu += repeat(speeches,2)  # v19 NEW: 10 YouTube human speeches @2x (diverse human voices)
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

bundle={"router":R,"tech":T,"general":G}
joblib.dump(bundle, os.path.join(OUT,"allfields_v19.joblib"))
print("SAVED allfields_v19.joblib", flush=True)
meta_out={"seed":SEED,"feature_dim":DIM,"architecture":"routed: p=R*T+(1-R)*G_v5; T=universal (research+essay AI)",
 "router_acc":round(r_acc,4),"tech_acc":round(t_acc,4),"elapsed_s":round(time.time()-t0,1),
 "note":"v19: v18 + 10 YouTube human speeches @2x (Nisar's idea; diverse human voices, fix human FPR)."}
with open(os.path.join(OUT,"allfields_v19_meta.json"),"w") as f: json.dump(meta_out,f,indent=2)
live=os.path.join(PROJ,"models","nak_detector.joblib")
h=hashlib.md5(open(live,"rb").read()).hexdigest()
print(f"LIVE MD5: {h} {'OK' if h=='9eeba80e03d06c7db8b35796965746c9' else 'MISMATCH!'}", flush=True)
print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
