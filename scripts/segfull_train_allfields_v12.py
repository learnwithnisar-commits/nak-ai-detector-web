#!/usr/bin/env python3
"""Segment-level training: allfields_v12 — router learns "research vs essay".

v12 strategy (v11 diagnosis):
  v10/v11 probe revealed the BOTTLENECK: Router sends ALL academic text
  (R=1.0) to T, but T only understands research-paper style:
    ash_coal: R=1.0, T=0.000, G=0.014 -> 0.000 (T blind to ChatGPT essays)
    22103:    R=1.0, T=0.004, G=0.938 -> 0.004 (G knows, but R blocks it!)
    PhD:      R=1.0, T=0.505, G=0.925 -> 0.505 (G knows, R blocks it!)

  v12 FIX: Router learns RESEARCH-PAPER vs ESSAY (not "technical vs general"):
    R->1 (research): arxiv, pubmed, europepmc, oa_sub, petro_ai, real_ai,
                     fam_ai, drive_hu, matched_hu, h004, arxiv_hu
    R->0 (essay):    turnitin AI/human (student assignments), 22103 pairs,
                     Meta essays, WA docs, Nisar x3, h005, wiki

  T (research specialist): petro_ai @3x, real_ai, fam_ai @2x  vs
                           drive_hu @3x, matched_hu, arxiv_hu, h004 @3x
     -> keeps Full Paper 81%+ (research-paper AI vs human)

  G (essay specialist, FRESH): Meta @3x, PhD @5x, turn_ai @2x, BDO, ai_txts @3x
                            vs PhD_updated @3x, essay_hu @3x, turn_hu,
                               Nisar @3x, h005 @3x
     -> fixes ash_coal, PhD, ES1002, Junlan_Li (essay AI vs human)

Outputs:
  models/candidates/allfields_v12.joblib
  models/candidates/allfields_v12_feats/
  models/candidates/allfields_v12_meta.json

DOES NOT touch the live model (models/nak_detector.joblib).
"""
import json, os, sys, random, glob, time, hashlib
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
assert DIM == 187
print(f"feature dim: {DIM}", flush=True)

DATA = os.path.join(PROJ, "data")
OUT = os.path.join(PROJ, "models", "candidates")
FEATS = os.path.join(OUT, "allfields_v12_feats")
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


def load_texts(path, max_docs=None):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)["text"])
            if max_docs and len(texts) >= max_docs:
                break
    return texts


def load_turnitin_ai_no_meta(path):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if str(o.get("label", "")).lower() == "ai" \
                    and not str(o.get("source", "")).startswith("meta_ai_essay"):
                texts.append(o["text"])
    return texts


def load_meta_essays(path):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if str(o.get("source", "")).startswith("meta_ai_essay"):
                texts.append(o["text"])
    return texts


def load_texts_by_label(path, want="ai"):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if str(o.get("label", "")).lower() == want:
                texts.append(o["text"])
    return texts


def load_whatsapp(path):
    EXCLUDE = {"5061", "55743", "58078", "33647", "Dreams_Final_Report_AUB_3", "Dreams"}
    out = {"phd_ai": [], "bdo_ai": [], "phd_upd_hu": [], "essay_hu": []}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            src = str(o.get("source", ""))
            if any(x in src for x in EXCLUDE):
                continue
            lab = str(o.get("label", "")).lower()
            if lab == "ai":
                if "Personal_statement_PhD" in src:
                    out["phd_ai"].append(o["text"])
                elif "BDO" in src:
                    out["bdo_ai"].append(o["text"])
            elif lab == "human":
                if "PhD_updated" in src:
                    out["phd_upd_hu"].append(o["text"])
                else:
                    out["essay_hu"].append(o["text"])
    return out


def load_txt_nocomments(path):
    with open(path, encoding="utf-8") as f:
        body = "\n".join(l for l in f.read().splitlines() if not l.startswith("#"))
    return body.strip()


def repeat(lst, k):
    out = []
    for t in lst:
        out.extend([t] * k)
    return out


# ================= data pools =================
turn_ai = load_turnitin_ai_no_meta(os.path.join(DATA, "turnitin_mass_paired.jsonl"))
turn_hu = load_texts_by_label(os.path.join(DATA, "turnitin_mass_paired.jsonl"), "human")
meta_essays = load_meta_essays(os.path.join(DATA, "turnitin_mass_paired.jsonl"))
wa = load_whatsapp(os.path.join(DATA, "whatsapp21_training.jsonl"))

a22103 = load_texts(os.path.join(DATA, "turnitin_anchors.jsonl"))
p22103_ai = load_texts_by_label(os.path.join(DATA, "turnitin_22103_paired.jsonl"), "ai")
p22103_hu = load_texts_by_label(os.path.join(DATA, "turnitin_22103_paired.jsonl"), "human")
drive_hu = load_texts(os.path.join(DATA, "drive_human_samples.jsonl"))
matched_hu = load_texts(os.path.join(DATA, "human_technical_matched.jsonl"))
fam_ai = []
for d in ["family_deepseek_train.jsonl", "family_claude_train.jsonl",
          "family_multistyle_train.jsonl"]:
    p = os.path.join(DATA, d)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    fam_ai.append(json.loads(line)["text"])
petro_ai = load_texts(os.path.join(DATA, "family_chatgpt_train.jsonl"))
real_ai = load_texts(os.path.join(DATA, "real_ai", "real_ai_train.jsonl"), max_docs=400)
arxiv_hu = load_texts(os.path.join(DATA, "arxiv_train.jsonl"), max_docs=300)
arxiv_r = load_texts(os.path.join(DATA, "arxiv_train.jsonl"), max_docs=600)
pubmed_r = load_texts(os.path.join(DATA, "pubmed_train.jsonl"), max_docs=250)
europepmc_r = load_texts(os.path.join(DATA, "europepmc_train.jsonl"), max_docs=400)
oa_sub_r = load_texts(os.path.join(DATA, "oa_sub_train.jsonl"), max_docs=800)
wiki_short = load_texts(os.path.join(DATA, "wiki_short_train.jsonl"))
wiki_full = load_texts(os.path.join(DATA, "wiki_full_train.jsonl"))
h004 = load_txt_nocomments(os.path.join(DATA, "verified", "human_004_listeria.txt"))
h005 = load_txt_nocomments(os.path.join(DATA, "verified", "human_005_science.txt"))
nisar_txts = [load_txt_nocomments(os.path.join(DATA, "verified", f))
              for f in ["nisar_001_waste.txt", "nisar_002_story.txt", "nisar_003_school.txt"]]
ai_txts = [load_txt_nocomments(os.path.join(DATA, "verified", f))
           for f in ["ai_001_waste.txt", "ai_002_story.txt", "ai_003_school.txt",
                     "ai_004_listeria.txt"] if os.path.exists(os.path.join(DATA, "verified", f))]

print(f"pools ready [{time.time()-t0:.0f}s]", flush=True)


def segs_of(docs):
    segs = []
    for t in docs:
        segs.extend(windows(t))
    return segs


# ================= ROUTER: research(1) vs essay(0) [v12 NEW] ===============
r_research = (arxiv_r + pubmed_r + europepmc_r + oa_sub_r
              + petro_ai + real_ai + fam_ai
              + drive_hu + matched_hu + [h004] + arxiv_hu)
r_essay = (turn_ai + turn_hu + a22103 + p22103_ai + p22103_hu
           + meta_essays + wa["phd_ai"] + wa["bdo_ai"]
           + wa["phd_upd_hu"] + wa["essay_hu"]
           + nisar_txts + [h005] + wiki_short + wiki_full)

r_res_segs = segs_of(r_research)
r_ess_segs = segs_of(r_essay)
print(f"router raw: research={len(r_res_segs)} essay={len(r_ess_segs)}", flush=True)
if len(r_res_segs) > 1.5 * len(r_ess_segs):
    random.shuffle(r_res_segs)
    r_res_segs = r_res_segs[:int(1.5 * len(r_ess_segs))]
elif len(r_ess_segs) > 1.5 * len(r_res_segs):
    random.shuffle(r_ess_segs)
    r_ess_segs = r_ess_segs[:int(1.5 * len(r_res_segs))]
rX_texts = r_res_segs + r_ess_segs
rY = np.array([1] * len(r_res_segs) + [0] * len(r_ess_segs))
idx = list(range(len(rX_texts))); random.shuffle(idx)
rX_texts = [rX_texts[i] for i in idx]; rY = rY[idx]
print(f"router: {len(rX_texts)} pos={int(rY.sum())} [{time.time()-t0:.0f}s]", flush=True)

# ================= T: research AI vs research human ========================
t_ai_docs = (repeat(petro_ai, 3) + real_ai + repeat(fam_ai, 2)
             + repeat(turn_ai, 2))   # turnitin AI with technical content
t_hu_docs = (repeat(drive_hu, 3) + matched_hu + arxiv_hu
             + repeat([h004], 3) + turn_hu)

t_ai_segs = segs_of(t_ai_docs)
t_hu_segs = segs_of(t_hu_docs)
random.shuffle(t_hu_segs)
t_hu_segs = t_hu_segs[:int(2.5 * len(t_ai_segs))]
tX_texts = t_ai_segs + t_hu_segs
tY = np.array([1] * len(t_ai_segs) + [0] * len(t_hu_segs))
idx = list(range(len(tX_texts))); random.shuffle(idx)
tX_texts = [tX_texts[i] for i in idx]; tY = tY[idx]
print(f"tech: {len(tX_texts)} pos={int(tY.sum())} [{time.time()-t0:.0f}s]", flush=True)

# ================= G: essay AI vs essay human [v12 NEW] =====================
g_ai_docs = (repeat(meta_essays, 3) + repeat(wa["phd_ai"], 5)
             + repeat(turn_ai, 2) + wa["bdo_ai"] + repeat(ai_txts, 3))
g_hu_docs = (repeat(wa["phd_upd_hu"], 3) + repeat(wa["essay_hu"], 3)
             + turn_hu + repeat(nisar_txts, 3) + repeat([h005], 3))

g_ai_segs = segs_of(g_ai_docs)
g_hu_segs = segs_of(g_hu_docs)
random.shuffle(g_hu_segs)
g_hu_segs = g_hu_segs[:int(2.0 * len(g_ai_segs))]
gX_texts = g_ai_segs + g_hu_segs
gY = np.array([1] * len(g_ai_segs) + [0] * len(g_hu_segs))
idx = list(range(len(gX_texts))); random.shuffle(idx)
gX_texts = [gX_texts[i] for i in idx]; gY = gY[idx]
print(f"essay-G: {len(gX_texts)} pos={int(gY.sum())} [{time.time()-t0:.0f}s]", flush=True)


def featurize_cached(texts, y, tag):
    n = len(texts); CHUNK = 5000
    n_chunks = (n + CHUNK - 1) // CHUNK
    for i in range(n_chunks):
        ckpt = os.path.join(FEATS, f"{tag}_{i:03d}.npz")
        if os.path.exists(ckpt):
            print(f"  {tag} chunk {i+1}/{n_chunks} cached", flush=True)
            continue
        s = slice(i * CHUNK, min((i + 1) * CHUNK, n))
        X = np.stack([extract_features(t) for t in texts[s]])
        assert X.shape[1] == DIM
        np.savez_compressed(ckpt, X=X, y=y[s])
        print(f"  {tag} chunk {i+1}/{n_chunks} {X.shape} [{time.time()-t0:.0f}s]", flush=True)
    Xs, ys = [], []
    for ckpt in sorted(glob.glob(os.path.join(FEATS, f"{tag}_*.npz"))):
        d = np.load(ckpt); Xs.append(d["X"]); ys.append(d["y"])
    return np.vstack(Xs), np.concatenate(ys)


def train_lr(X, y, name):
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                             random_state=SEED)
    clf.fit(X, y)
    acc = float((clf.predict(X) == y).mean())
    print(f"{name} train acc: {acc:.4f} X={X.shape} [{time.time()-t0:.0f}s]", flush=True)
    return clf, acc


rX, ry = featurize_cached(rX_texts, rY, "router")
R, r_acc = train_lr(rX, ry, "router")
tX, ty = featurize_cached(tX_texts, tY, "tech")
T, t_acc = train_lr(tX, ty, "research-specialist")
gX, gy = featurize_cached(gX_texts, gY, "general")
G, g_acc = train_lr(gX, gy, "essay-specialist")


def routed_proba(text):
    f = np.asarray(extract_features(text), dtype=np.float64).reshape(1, -1)
    r = float(R.predict_proba(f)[0][1])
    pt = float(T.predict_proba(f)[0][1])
    pg = float(G.predict_proba(f)[0][1])
    return r * pt + (1 - r) * pg, r, pt, pg


with open(os.path.join(PROJ, "data", "real_ai", "ash_coal_essay_chatgpt.txt")) as f:
    ash = f.read()
for name, pr in [
    ("ash_coal", ash[:3000]),
    ("phd", " ".join(wa["phd_ai"][0].split()[:800])),
    ("meta", meta_essays[0][:2000]),
    ("petro", petro_ai[0][:2000]),
    ("nisar", nisar_txts[0][:2000]),
    ("wiki", wiki_full[0][:2000] if wiki_full else "test"),
]:
    p, r, pt, pg = routed_proba(pr)
    print(f"  {name}: routed={p:.3f} R_research={r:.3f} T={pt:.3f} G_essay={pg:.3f}", flush=True)

bundle = {"router": R, "tech": T, "general": G}
out_model = os.path.join(OUT, "allfields_v12.joblib")
joblib.dump(bundle, out_model)
print(f"SAVED {out_model}", flush=True)
meta = {
    "seed": SEED, "feature_dim": DIM,
    "architecture": "routed-ensemble: p=R_research*T+(1-R)*G_essay; all fresh",
    "router_acc": round(r_acc, 4), "tech_acc": round(t_acc, 4), "gen_acc": round(g_acc, 4),
    "elapsed_s": round(time.time() - t0, 1),
    "note": ("v12: router=research vs essay; T=research AI/human; "
             "G=essay AI/human (Meta @3x, PhD @5x vs WA humans @3x, Nisar @3x)."),
}
with open(os.path.join(OUT, "allfields_v12_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("META:", json.dumps(meta), flush=True)

live = os.path.join(PROJ, "models", "nak_detector.joblib")
h = hashlib.md5(open(live, "rb").read()).hexdigest()
print(f"LIVE MODEL MD5: {h} {'OK' if h == '9eeba80e03d06c7db8b35796965746c9' else 'MISMATCH!'}",
      flush=True)
print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
