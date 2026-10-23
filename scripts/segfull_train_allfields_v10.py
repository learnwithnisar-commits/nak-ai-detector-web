#!/usr/bin/env python3
"""Segment-level training: allfields_v10 — back to v6's winning formula + PhD boost.

v10 strategy (learn from v6..v9):
  v6 was BEST: Full Paper 81.5% PASS, Drive FPR PASS. Its data mix trained
  BEFORE the 10 Meta essays were appended to turnitin_mass_paired.jsonl
  (v6 train log has 0 meta mentions; file mtime 06:57 > v6 train 06:38).

  v7/v8/v9 regressed: multi-scale + Meta essays ("essay=AI" poisoned FPR) +
  *% band WhatsApp docs labeled AI (poisoned human FPR catastrophically).

  v10 = v6's EXACT recipe (segments-only, frozen v5 generalist) with
  SURGICAL additions only:
    EXCLUDE: 10 Meta essays, five *% band WhatsApp docs (5061/55743/58078/33647/Dreams)
    ADD AI:    Personal_statement_PhD (100%) @5x, BDO_Lebanon (22%) @1x
    ADD HUMAN: PhD_updated (0%) @3x, Arab_Design (0%) @2x, 77701 (0%) @2x,
               Hospital_AR1 (0%) @2x  (human essay counterbalance -> ES1002 FPR)

  Architecture: v6 routed ensemble, NO new features (v9 proved doc-level
  features don't survive 8-sentence windowing):
    p(x) = R(x)*T(x) + (1-R(x))*G(x),  G = frozen v5

Outputs:
  models/candidates/allfields_v10.joblib
  models/candidates/allfields_v10_feats/
  models/candidates/allfields_v10_meta.json

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
assert DIM == 187, f"v10 uses base features only, got dim {DIM}"
print(f"feature dim: {DIM}", flush=True)

DATA = os.path.join(PROJ, "data")
OUT = os.path.join(PROJ, "models", "candidates")
FEATS = os.path.join(OUT, "allfields_v10_feats")
os.makedirs(FEATS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# frozen generalist = v5 (exactly like v6)
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
    """Turnitin AI highlights, EXCLUDING the 10 Meta essays (v10 fix)."""
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


def load_whatsapp_v10(path):
    """WhatsApp high-confidence docs, v10 weights.

    EXCLUDES the five *% band docs (5061, 55743, 58078, 33647, Dreams):
    81-99% human, must never be AI-labeled.

    Returns dict: phd_ai, bdo_ai, phd_upd_hu, essay_hu (3 docs)
    """
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


# ================= data pools (v6 base + v10 surgical adds) =================
turn_ai = load_turnitin_ai_no_meta(os.path.join(DATA, "turnitin_mass_paired.jsonl"))
turn_hu = load_texts_by_label(os.path.join(DATA, "turnitin_mass_paired.jsonl"), "human")
wa = load_whatsapp_v10(os.path.join(DATA, "whatsapp21_training.jsonl"))
wa_all = wa["phd_ai"] + wa["bdo_ai"] + wa["phd_upd_hu"] + wa["essay_hu"]

# v6 base pools
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

print(f"pools: turn_ai(no-meta)={len(turn_ai)} turn_hu={len(turn_hu)} "
      f"WA phd_ai={len(wa['phd_ai'])} bdo={len(wa['bdo_ai'])} "
      f"phd_upd={len(wa['phd_upd_hu'])} essay_hu={len(wa['essay_hu'])} "
      f"drive={len(drive_hu)} fam={len(fam_ai)} [{time.time()-t0:.0f}s]", flush=True)
assert len(turn_ai) == 103, f"expected 103 turnitin AI (no meta), got {len(turn_ai)}"
assert len(turn_hu) == 133, f"expected 133 turnitin human, got {len(turn_hu)}"
assert len(wa["phd_ai"]) == 1 and len(wa["bdo_ai"]) == 1, "WA AI docs missing"
assert len(wa["phd_upd_hu"]) == 1 and len(wa["essay_hu"]) == 3, "WA human docs missing"


def segs_of(docs):
    segs = []
    for t in docs:
        segs.extend(windows(t))
    return segs


# ================= ROUTER: technical(1) vs general(0) [v6 recipe + WA] ======
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
print(f"router segs raw: tech={len(r_tech_segs)} gen={len(r_gen_segs)}", flush=True)
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

# ================= T: technical AI(1) vs technical human(0) =================
# v6 recipe + v10 surgical: PhD @5x, BDO @1x, WA humans @3x/@2x
t_ai_docs = []
t_ai_docs += repeat(turn_ai, 3)
t_ai_docs += repeat(a22103, 3)                                   # 22103
t_ai_docs += repeat(p22103_ai, 3)
t_ai_docs += repeat(fam_ai, 2)
t_ai_docs += repeat(petro_ai, 3)                                # petroleum
t_ai_docs += real_ai
t_ai_docs += repeat(wa["phd_ai"], 5)                            # v10: PhD 100% @5x
t_ai_docs += wa["bdo_ai"]                                       # v10: BDO 22% @1x

t_hu_docs = []
t_hu_docs += turn_hu
t_hu_docs += repeat(drive_hu, 3)
t_hu_docs += matched_hu
t_hu_docs += repeat([h004], 3)
t_hu_docs += repeat(p22103_hu, 2)
t_hu_docs += arxiv_hu                                          # real papers=human
t_hu_docs += repeat(wa["phd_upd_hu"], 3)                       # v10: PhD pair @3x
t_hu_docs += repeat(wa["essay_hu"], 2)                         # v10: human essays @2x

print(f"T docs: ai={len(t_ai_docs)} human={len(t_hu_docs)} [{time.time()-t0:.0f}s]", flush=True)
t_ai_segs = segs_of(t_ai_docs)
t_hu_segs = segs_of(t_hu_docs)
print(f"T segs raw: ai={len(t_ai_segs)} human={len(t_hu_segs)}", flush=True)
random.shuffle(t_hu_segs)
t_hu_segs = t_hu_segs[:int(2.5 * len(t_ai_segs))]
tX_texts = t_ai_segs + t_hu_segs
tY = np.array([1] * len(t_ai_segs) + [0] * len(t_hu_segs))
idx = list(range(len(tX_texts))); random.shuffle(idx)
tX_texts = [tX_texts[i] for i in idx]; tY = tY[idx]
print(f"T segs: {len(tX_texts)} pos={int(tY.sum())} [{time.time()-t0:.0f}s]", flush=True)


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
        assert X.shape[1] == DIM, f"dim mismatch {X.shape[1]} != {DIM}"
        np.savez_compressed(ckpt, X=X, y=y[s])
        print(f"  {tag} chunk {i+1}/{n_chunks} {X.shape} [{time.time()-t0:.0f}s]", flush=True)
    Xs, ys = [], []
    for ckpt in sorted(glob.glob(os.path.join(FEATS, f"{tag}_*.npz"))):
        d = np.load(ckpt); Xs.append(d["X"]); ys.append(d["y"])
    return np.vstack(Xs), np.concatenate(ys)


rX, ry = featurize_cached(rX_texts, rY, "router")
print(f"router X={rX.shape} [{time.time()-t0:.0f}s]", flush=True)
tX, ty = featurize_cached(tX_texts, tY, "tech")
print(f"tech X={tX.shape} [{time.time()-t0:.0f}s]", flush=True)

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


def routed_proba(text):
    f = np.asarray(extract_features(text), dtype=np.float64).reshape(1, -1)
    r = float(R.predict_proba(f)[0][1])
    pt = float(T.predict_proba(f)[0][1])
    pg = float(G.predict_proba(f)[0][1])
    return r * pt + (1 - r) * pg, r, pt, pg


probe_ai = ("In this study we investigate the thermophysical properties of hybrid "
            "nanofluids under turbulent flow conditions. The experimental results "
            "demonstrate a significant enhancement in heat transfer coefficient.")
probe_hu = "aj hum school mein science ka project banayenge jo pani se bijli banata hai"
for name, pr in [("tech-probe", probe_ai), ("general-probe", probe_hu)]:
    p, r, pt, pg = routed_proba(pr)
    print(f"  {name}: routed={p:.3f} R_tech={r:.3f} T={pt:.3f} G={pg:.3f}", flush=True)
# PhD probe (should be HIGH AI)
p, r, pt, pg = routed_proba(" ".join(wa["phd_ai"][0].split()[:800]))
print(f"  phd-probe: routed={p:.3f} R={r:.3f} T={pt:.3f} G={pg:.3f}", flush=True)

# ================= save bundle =================
bundle = {"router": R, "tech": T, "general": G}
out_model = os.path.join(OUT, "allfields_v10.joblib")
joblib.dump(bundle, out_model)
print(f"SAVED {out_model}", flush=True)

meta = {
    "seed": SEED,
    "feature_dim": DIM,
    "architecture": "routed-ensemble: p = R_tech*T + (1-R_tech)*G; G=frozen v5 (v6 recipe)",
    "router_segs": int(len(rX_texts)),
    "router_train_acc": round(r_acc, 4),
    "tech_segs": int(len(tX_texts)),
    "tech_pos_segs": int(tY.sum()),
    "tech_train_acc": round(t_acc, 4),
    "elapsed_s": round(time.time() - t0, 1),
    "note": ("v10: v6's winning recipe (segments-only, frozen v5, NO meta essays) + "
             "surgical WhatsApp adds: PhD 100% @5x, BDO 22% @1x, "
             "PhD_updated @3x, 3 human essays @2x; *% docs excluded."),
}
with open(os.path.join(OUT, "allfields_v10_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("META:", json.dumps(meta), flush=True)

# live-model integrity guard
live = os.path.join(PROJ, "models", "nak_detector.joblib")
h = hashlib.md5(open(live, "rb").read()).hexdigest()
print(f"LIVE MODEL MD5: {h} {'OK' if h == '9eeba80e03d06c7db8b35796965746c9' else 'MISMATCH!'}",
      flush=True)
print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
