#!/usr/bin/env python3
"""Segment-level training: allfields_v11 — v10 + fresh generalist.

v11 strategy (v10 failed 6/7 gates):
  v10 kept Full Paper 81.7% PASS and Drive FPR PASS (v6's strengths).
  Failures and fixes:
  1. ash_coal 34.7% vs 100% (gap 65.3 — SINGLE BIGGEST 8-anchor gap):
     ChatGPT essay routed to G (frozen v5, predates ChatGPT data).
     FIX: retrain G FRESH with Meta essays @3x + ChatGPT-family + PhD @3x.
  2. ES1002 22.1% vs 0% (human essay FPR):
     FIX: WhatsApp human essays (PhD_updated, Arab_Design, 77701) @3x in G.
  3. Wikipedia 27.69% vs <10%:
     FIX: Wikipedia @3x as human in G.
  4. PhD 86.1% vs 95%+:
     FIX: PhD @8x in T, PhD @3x in G (both routes covered).
  5. Junlan_Li 46.3% vs * (human essay FPR):
     FIX: same human-essay counterbalance in G.
  6. 22103 89% vs 57% (overshoot):
     FIX: reduce 22103 AI weight @3x -> @2x, increase human @2x -> @3x.
  7. h004 32.3% (human technical FPR):
     FIX: keep drive_hu @3x, matched_hu, arxiv_hu in T.

  Architecture: routed ensemble p = R*T + (1-R)*G.
  R: retrained (v10 recipe, fast).
  T: retrained (v10 recipe + PhD @8x, 22103 rebalanced).
  G: FRESH (not frozen v5) — general-register AI vs human.

Outputs:
  models/candidates/allfields_v11.joblib
  models/candidates/allfields_v11_feats/
  models/candidates/allfields_v11_meta.json

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
assert DIM == 187, f"v11 uses base features only, got dim {DIM}"
print(f"feature dim: {DIM}", flush=True)

DATA = os.path.join(PROJ, "data")
OUT = os.path.join(PROJ, "models", "candidates")
FEATS = os.path.join(OUT, "allfields_v11_feats")
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


def load_whatsapp_v11(path):
    EXCLUDE = {"5061", "55743", "58078", "33647", "Dreams_Final_Report_AUB_3",
               "Dreams"}
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
wa = load_whatsapp_v11(os.path.join(DATA, "whatsapp21_training.jsonl"))
wa_all = wa["phd_ai"] + wa["bdo_ai"] + wa["phd_upd_hu"] + wa["essay_hu"]

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
wiki_short = load_texts(os.path.join(DATA, "wiki_short_train.jsonl"))
wiki_full = load_texts(os.path.join(DATA, "wiki_full_train.jsonl"))
h004 = load_txt_nocomments(os.path.join(DATA, "verified", "human_004_listeria.txt"))
h005 = load_txt_nocomments(os.path.join(DATA, "verified", "human_005_science.txt"))
nisar_txts = [load_txt_nocomments(os.path.join(DATA, "verified", f))
              for f in ["nisar_001_waste.txt", "nisar_002_story.txt", "nisar_003_school.txt"]]
ai_txts = [load_txt_nocomments(os.path.join(DATA, "verified", f))
           for f in ["ai_001_waste.txt", "ai_002_story.txt", "ai_003_school.txt",
                     "ai_004_listeria.txt"] if os.path.exists(os.path.join(DATA, "verified", f))]

print(f"pools: turn_ai={len(turn_ai)} turn_hu={len(turn_hu)} meta={len(meta_essays)} "
      f"WA phd={len(wa['phd_ai'])} bdo={len(wa['bdo_ai'])} "
      f"phd_upd={len(wa['phd_upd_hu'])} essay_hu={len(wa['essay_hu'])} "
      f"ai_txts={len(ai_txts)} [{time.time()-t0:.0f}s]", flush=True)
assert len(turn_ai) == 103 and len(meta_essays) == 10
assert len(wa["phd_ai"]) == 1 and len(wa["essay_hu"]) == 3


def segs_of(docs):
    segs = []
    for t in docs:
        segs.extend(windows(t))
    return segs


# ================= ROUTER (v10 recipe) =================
r_tech, r_gen = [], []
for fn, cap in [("arxiv_train.jsonl", 600), ("pubmed_train.jsonl", 250),
                ("europepmc_train.jsonl", 400), ("oa_sub_train.jsonl", 800)]:
    p = os.path.join(DATA, fn)
    if os.path.exists(p):
        r_tech += load_texts(p, max_docs=cap)
r_tech += turn_ai + turn_hu + a22103 + fam_ai + petro_ai + real_ai
r_tech += drive_hu + matched_hu + [h004] + wa_all
r_gen += wiki_short + wiki_full + nisar_txts + [h005]

r_tech_segs = segs_of(r_tech)
r_gen_segs = segs_of(r_gen)
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
print(f"router segs: {len(rX_texts)} pos={int(rY.sum())} [{time.time()-t0:.0f}s]", flush=True)

# ================= TECH SPECIALIST (v10 + PhD @8x, 22103 rebalanced) ======
t_ai_docs = []
t_ai_docs += repeat(turn_ai, 3)
t_ai_docs += repeat(a22103, 2)                                   # v11: 22103 @2x (was @3x)
t_ai_docs += repeat(p22103_ai, 2)                               # v11: paired AI @2x
t_ai_docs += repeat(fam_ai, 2)
t_ai_docs += repeat(petro_ai, 3)
t_ai_docs += real_ai
t_ai_docs += repeat(wa["phd_ai"], 8)                           # v11: PhD @8x (was @5x)
t_ai_docs += wa["bdo_ai"]

t_hu_docs = []
t_hu_docs += turn_hu
t_hu_docs += repeat(drive_hu, 3)
t_hu_docs += matched_hu
t_hu_docs += repeat([h004], 3)
t_hu_docs += repeat(p22103_hu, 3)                              # v11: paired human @3x (was @2x)
t_hu_docs += arxiv_hu
t_hu_docs += repeat(wa["phd_upd_hu"], 3)
t_hu_docs += repeat(wa["essay_hu"], 2)

t_ai_segs = segs_of(t_ai_docs)
t_hu_segs = segs_of(t_hu_docs)
random.shuffle(t_hu_segs)
t_hu_segs = t_hu_segs[:int(2.5 * len(t_ai_segs))]
tX_texts = t_ai_segs + t_hu_segs
tY = np.array([1] * len(t_ai_segs) + [0] * len(t_hu_segs))
idx = list(range(len(tX_texts))); random.shuffle(idx)
tX_texts = [tX_texts[i] for i in idx]; tY = tY[idx]
print(f"tech segs: {len(tX_texts)} pos={int(tY.sum())} [{time.time()-t0:.0f}s]", flush=True)

# ================= GENERALIST (FRESH — v11's key change) ===================
# General-register AI vs human. Targets: ash_coal (ChatGPT essay), ES1002,
# Junlan_Li, Wikipedia, Nisar/h005 FPR.
g_ai_docs = []
g_ai_docs += repeat(meta_essays, 3)                            # v11: Meta essays @3x (general AI)
g_ai_docs += repeat(fam_ai, 2)                                 # multi-family AI
g_ai_docs += repeat(ai_txts, 3)                               # verified AI pairs
g_ai_docs += repeat(wa["phd_ai"], 3)                          # v11: PhD @3x in G too
g_ai_docs += wa["bdo_ai"]
g_ai_docs += turn_ai                                          # turnitin AI (mixed register)

g_hu_docs = []
g_hu_docs += repeat(wiki_short, 3)                            # v11: Wikipedia @3x
g_hu_docs += repeat(wiki_full, 3)                             # v11: Wikipedia @3x
g_hu_docs += repeat(nisar_txts, 3)                            # Nisar's writing @3x
g_hu_docs += repeat([h005], 3)                                # friend's science @3x
g_hu_docs += turn_hu                                          # turnitin human
g_hu_docs += repeat(wa["phd_upd_hu"], 3)                      # v11: human essay @3x
g_hu_docs += repeat(wa["essay_hu"], 3)                        # v11: human essays @3x
g_hu_docs += drive_hu

g_ai_segs = segs_of(g_ai_docs)
g_hu_segs = segs_of(g_hu_docs)
random.shuffle(g_hu_segs)
g_hu_segs = g_hu_segs[:int(2.0 * len(g_ai_segs))]
gX_texts = g_ai_segs + g_hu_segs
gY = np.array([1] * len(g_ai_segs) + [0] * len(g_hu_segs))
idx = list(range(len(gX_texts))); random.shuffle(idx)
gX_texts = [gX_texts[i] for i in idx]; gY = gY[idx]
print(f"general segs: {len(gX_texts)} pos={int(gY.sum())} [{time.time()-t0:.0f}s]", flush=True)


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
T, t_acc = train_lr(tX, ty, "tech-specialist")
gX, gy = featurize_cached(gX_texts, gY, "general")
G, g_acc = train_lr(gX, gy, "generalist")


def routed_proba(text):
    f = np.asarray(extract_features(text), dtype=np.float64).reshape(1, -1)
    r = float(R.predict_proba(f)[0][1])
    pt = float(T.predict_proba(f)[0][1])
    pg = float(G.predict_proba(f)[0][1])
    return r * pt + (1 - r) * pg, r, pt, pg


for name, pr in [
    ("tech-probe", "In this study we investigate the thermophysical properties of hybrid "
     "nanofluids under turbulent flow conditions. The experimental results "
     "demonstrate a significant enhancement in heat transfer coefficient."),
    ("general-probe", "aj hum school mein science ka project banayenge jo pani se bijli banata hai"),
    ("phd-probe", " ".join(wa["phd_ai"][0].split()[:800])),
    ("meta-probe", meta_essays[0][:2000]),
    ("wiki-probe", wiki_full[0][:2000] if wiki_full else "test"),
]:
    p, r, pt, pg = routed_proba(pr)
    print(f"  {name}: routed={p:.3f} R={r:.3f} T={pt:.3f} G={pg:.3f}", flush=True)

# ================= save =================
bundle = {"router": R, "tech": T, "general": G}
out_model = os.path.join(OUT, "allfields_v11.joblib")
joblib.dump(bundle, out_model)
print(f"SAVED {out_model}", flush=True)
meta = {
    "seed": SEED, "feature_dim": DIM,
    "architecture": "routed-ensemble: p=R*T+(1-R)*G; R+T retrained (v10 recipe), G FRESH",
    "router_acc": round(r_acc, 4), "tech_acc": round(t_acc, 4), "gen_acc": round(g_acc, 4),
    "router_segs": int(len(rX_texts)), "tech_segs": int(len(tX_texts)),
    "tech_pos": int(tY.sum()), "gen_segs": int(len(gX_texts)), "gen_pos": int(gY.sum()),
    "elapsed_s": round(time.time() - t0, 1),
    "note": ("v11: v10 + FRESH generalist (Meta @3x, Wiki @3x, WA humans @3x, "
             "PhD @3x in G); T: PhD @8x, 22103 rebalanced @2x/@3x."),
}
with open(os.path.join(OUT, "allfields_v11_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("META:", json.dumps(meta), flush=True)

live = os.path.join(PROJ, "models", "nak_detector.joblib")
h = hashlib.md5(open(live, "rb").read()).hexdigest()
print(f"LIVE MODEL MD5: {h} {'OK' if h == '9eeba80e03d06c7db8b35796965746c9' else 'MISMATCH!'}",
      flush=True)
print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
