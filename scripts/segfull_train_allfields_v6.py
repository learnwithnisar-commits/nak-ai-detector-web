#!/usr/bin/env python3
"""Segment-level training: allfields_v6 — DOMAIN-ROUTED ENSEMBLE.

v6 strategy (NEW ARCHITECTURE — first departure from single-LogReg):
  The v1..v5 failure mode is identical: one LogisticRegression learns
  "technical/academic register => AI". v6 splits the problem:

  - Router R:      P(technical register | x)   (LogReg, source-labeled)
  - Specialist T:  P(AI | x, technical)        (LogReg, BALANCED tech AI vs tech human)
  - Generalist G:  P(AI | x, general)          (frozen v5 model)
  - Inference:     p(x) = R(x)*T(x) + (1-R(x))*G(x)

  T is trained on a BALANCED technical set so it must learn the genuine
  AI-vs-human boundary inside technical writing instead of using
  "technical" itself as the AI signal.

  G reuses the frozen v5 (it already scores general/plains texts well:
  Nisar x3 -> 0.0-0.5%, science -> 1.5%).

Outputs:
  models/candidates/allfields_v6.joblib   (dict bundle, NOT drop-in)
  models/candidates/allfields_v6_feats/
  models/candidates/allfields_v6_meta.json

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
FEATS = os.path.join(OUT, "allfields_v6_feats")
os.makedirs(FEATS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# frozen generalist = v5
G = joblib.load(os.path.join(OUT, "allfields_v5.joblib"))
print("frozen generalist (v5) loaded", flush=True)


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
    s = str(lab).lower().strip()
    if s in ("ai", "1", "true", "yes"):
        return 1
    if s in ("human", "0", "false", "no"):
        return 0
    raise ValueError(f"unknown label: {lab!r}")


def load_texts(path, max_docs=None):
    """Load raw texts from a jsonl (uses 'text' field)."""
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


def load_txt_nocomments(path):
    with open(path, encoding="utf-8") as f:
        body = "\n".join(l for l in f.read().splitlines() if not l.startswith("#"))
    return body.strip()


def repeat(lst, k):
    out = []
    for t in lst:
        out.extend([t] * k)
    return out


# ================= ROUTER DATA: technical(1) vs general(0) =================
r_tech, r_gen = [], []

# technical sources (academic/technical register)
for fn, cap in [("arxiv_train.jsonl", 600), ("pubmed_train.jsonl", 250),
                ("europepmc_train.jsonl", 400), ("oa_sub_train.jsonl", 800)]:
    p = os.path.join(DATA, fn)
    if os.path.exists(p):
        r_tech += load_texts(p, max_docs=cap)
r_tech += load_texts_by_label(os.path.join(DATA, "turnitin_mass_paired.jsonl"), "ai")
r_tech += load_texts_by_label(os.path.join(DATA, "turnitin_mass_paired.jsonl"), "human")
r_tech += load_texts(os.path.join(DATA, "turnitin_anchors.jsonl"))
for d in ["family_deepseek_train.jsonl", "family_claude_train.jsonl",
          "family_multistyle_train.jsonl", "family_chatgpt_train.jsonl"]:
    p = os.path.join(DATA, d)
    if os.path.exists(p):
        # family files are AI-only jsonl
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r_tech.append(json.loads(line)["text"])
r_tech += load_texts(os.path.join(DATA, "drive_human_samples.jsonl"))
r_tech += load_texts(os.path.join(DATA, "human_technical_matched.jsonl"))
r_tech += load_texts(os.path.join(DATA, "real_ai", "real_ai_train.jsonl"), max_docs=400)
r_tech.append(load_txt_nocomments(os.path.join(DATA, "verified", "human_004_listeria.txt")))

# general sources (plain/encyclopedic/personal register)
r_gen += load_texts(os.path.join(DATA, "wiki_short_train.jsonl"))
r_gen += load_texts(os.path.join(DATA, "wiki_full_train.jsonl"))
for vf in ["nisar_001_waste.txt", "nisar_002_story.txt", "nisar_003_school.txt",
           "human_005_science.txt"]:
    p = os.path.join(DATA, "verified", vf)
    if os.path.exists(p):
        r_gen.append(load_txt_nocomments(p))

print(f"router docs: tech={len(r_tech)} gen={len(r_gen)} [{time.time()-t0:.0f}s]", flush=True)


def segs_of(docs):
    segs = []
    for t in docs:
        segs.extend(windows(t))
    return segs


r_tech_segs = segs_of(r_tech)
r_gen_segs = segs_of(r_gen)
print(f"router segs raw: tech={len(r_tech_segs)} gen={len(r_gen_segs)}", flush=True)
# balance: cap majority at 1.5x minority
if len(r_tech_segs) > 1.5 * len(r_gen_segs):
    random.shuffle(r_tech_segs)
    r_tech_segs = r_tech_segs[:int(1.5 * len(r_gen_segs))]
elif len(r_gen_segs) > 1.5 * len(r_tech_segs):
    random.shuffle(r_gen_segs)
    r_gen_segs = r_gen_segs[:int(1.5 * len(r_tech_segs))]
rX_texts = r_tech_segs + r_gen_segs
rY = np.array([1] * len(r_tech_segs) + [0] * len(r_gen_segs))
idx = list(range(len(rX_texts))); random.shuffle(idx)
rX_texts = [rX_texts[i] for i in idx]; rY = rY[idx]
print(f"router segs balanced: {len(rX_texts)} pos={int(rY.sum())} [{time.time()-t0:.0f}s]", flush=True)

# ================= T DATA: technical AI(1) vs technical human(0) ===========
t_ai_docs = []
t_ai_docs += repeat(load_texts_by_label(os.path.join(DATA, "turnitin_mass_paired.jsonl"), "ai"), 3)
t_ai_docs += repeat(load_texts(os.path.join(DATA, "turnitin_anchors.jsonl")), 3)  # 22103
t_ai_docs += repeat(load_texts_by_label(os.path.join(DATA, "turnitin_22103_paired.jsonl"), "ai"), 3)
for d in ["family_deepseek_train.jsonl", "family_claude_train.jsonl",
          "family_multistyle_train.jsonl"]:
    p = os.path.join(DATA, d)
    if os.path.exists(p):
        docs = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line)["text"])
        t_ai_docs += repeat(docs, 2)
t_ai_docs += repeat(load_texts(os.path.join(DATA, "family_chatgpt_train.jsonl")), 3)  # petroleum
t_ai_docs += load_texts(os.path.join(DATA, "real_ai", "real_ai_train.jsonl"), max_docs=400)

t_hu_docs = []
t_hu_docs += load_texts_by_label(os.path.join(DATA, "turnitin_mass_paired.jsonl"), "human")
t_hu_docs += repeat(load_texts(os.path.join(DATA, "drive_human_samples.jsonl")), 3)
t_hu_docs += load_texts(os.path.join(DATA, "human_technical_matched.jsonl"))
t_hu_docs += repeat([load_txt_nocomments(os.path.join(DATA, "verified", "human_004_listeria.txt"))], 3)
t_hu_docs += repeat(load_texts_by_label(os.path.join(DATA, "turnitin_22103_paired.jsonl"), "human"), 2)
t_hu_docs += load_texts(os.path.join(DATA, "arxiv_train.jsonl"), max_docs=300)  # real papers=human

print(f"T docs: ai={len(t_ai_docs)} human={len(t_hu_docs)} [{time.time()-t0:.0f}s]", flush=True)
t_ai_segs = segs_of(t_ai_docs)
t_hu_segs = segs_of(t_hu_docs)
print(f"T segs raw: ai={len(t_ai_segs)} human={len(t_hu_segs)}", flush=True)
random.shuffle(t_hu_segs)
t_hu_segs = t_hu_segs[:int(2.5 * len(t_ai_segs))]  # cap human at 2.5x AI
tX_texts = t_ai_segs + t_hu_segs
tY = np.array([1] * len(t_ai_segs) + [0] * len(t_hu_segs))
idx = list(range(len(tX_texts))); random.shuffle(idx)
tX_texts = [tX_texts[i] for i in idx]; tY = tY[idx]
print(f"T segs: {len(tX_texts)} pos={int(tY.sum())} [{time.time()-t0:.0f}s]", flush=True)

# ================= feature extraction (cached per subset) =================
def featurize_cached(texts, y, tag):
    n = len(texts)
    CHUNK = 5000
    n_chunks = (n + CHUNK - 1) // CHUNK
    for i in range(n_chunks):
        ckpt = os.path.join(FEATS, f"{tag}_{i:03d}.npz")
        if os.path.exists(ckpt):
            print(f"  {tag} chunk {i+1}/{n_chunks} cached", flush=True)
            continue
        s = slice(i * CHUNK, min((i + 1) * CHUNK, n))
        X = np.stack([extract_features(t) for t in texts[s]])
        assert X.shape[1] == DIM, f"dim mismatch {X.shape[1]} != {DIM}"
        np.savez_compressed(ckpt, X=X, y=y[s])
        print(f"  {tag} chunk {i+1}/{n_chunks} {X.shape} [{time.time()-t0:.0f}s]", flush=True)
    Xs, ys = [], []
    for ckpt in sorted(glob.glob(os.path.join(FEATS, f"{tag}_*.npz"))):
        d = np.load(ckpt)
        Xs.append(d["X"]); ys.append(d["y"])
    return np.vstack(Xs), np.concatenate(ys)


rX, ry = featurize_cached(rX_texts, rY, "router")
print(f"router X={rX.shape} [{time.time()-t0:.0f}s]", flush=True)
tX, ty = featurize_cached(tX_texts, tY, "tech")
print(f"tech X={tX.shape} [{time.time()-t0:.0f}s]", flush=True)

# ================= train =================
R = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1,
                       class_weight="balanced", random_state=SEED)
R.fit(rX, ry)
r_acc = float((R.predict(rX) == ry).mean())
print(f"router train acc: {r_acc:.4f} [{time.time()-t0:.0f}s]", flush=True)

T = LogisticRegression(max_iter=3000, C=1.0, n_jobs=-1,
                       class_weight="balanced", random_state=SEED)
T.fit(tX, ty)
t_acc = float((T.predict(tX) == ty).mean())
print(f"tech-specialist train acc: {t_acc:.4f} [{time.time()-t0:.0f}s]", flush=True)

# sanity: routed score on a technical-AI-ish and technical-human-ish probe
def routed_proba(text):
    f = np.asarray(extract_features(text), dtype=np.float64).reshape(1, -1)
    r = float(R.predict_proba(f)[0][1])
    pt = float(T.predict_proba(f)[0][1])
    pg = float(G.predict_proba(f)[0][1])
    return r * pt + (1 - r) * pg, r, pt, pg

probe_ai = ("In this study we investigate the thermophysical properties of hybrid "
            "nanofluids under turbulent flow conditions. The experimental results "
            "demonstrate a significant enhancement in heat transfer coefficient.")
probe_hu = ("aj hum school mein science ka project banayenge jo pani se bijli banata hai")
for name, pr in [("tech-probe", probe_ai), ("general-probe", probe_hu)]:
    p, r, pt, pg = routed_proba(pr)
    print(f"  {name}: routed={p:.3f} R_tech={r:.3f} T={pt:.3f} G={pg:.3f}", flush=True)

# ================= save bundle =================
bundle = {"router": R, "tech": T, "general": G}
out_model = os.path.join(OUT, "allfields_v6.joblib")
joblib.dump(bundle, out_model)
print(f"SAVED {out_model}", flush=True)

meta = {
    "seed": SEED,
    "feature_dim": DIM,
    "architecture": "routed-ensemble: p = R_tech*T + (1-R_tech)*G; G=frozen v5",
    "router_segs": int(len(rX_texts)),
    "router_train_acc": round(r_acc, 4),
    "tech_segs": int(len(tX_texts)),
    "tech_pos_segs": int(tY.sum()),
    "tech_train_acc": round(t_acc, 4),
    "elapsed_s": round(time.time() - t0, 1),
    "note": ("v6: NEW ARCHITECTURE. Router(technical vs general) + balanced "
             "technical specialist + frozen v5 generalist. Attacks the "
             "'technical=AI' failure mode of v1..v5."),
}
with open(os.path.join(OUT, "allfields_v6_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("META:", json.dumps(meta), flush=True)
print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
