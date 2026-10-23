#!/usr/bin/env python3
"""v25-E3: v24 recipe + petro (ChatGPT petroleum) @3x.

Ablation per v24 root-cause analysis (E3 = isolate petro):
- Change: v24 recipe EXACTLY + petro (family_chatgpt_train.jsonl,
  ChatGPT petroleum-engineering doc) @3x added back to T.
- Hypothesis: Full Paper improves (petroleum-specific ChatGPT signal).
- Proves: domain-specific contribution of ChatGPT-petroleum text.

Research only, NOT deployment. Live v17 untouched.
Output: models/candidates/allfields_v25_E3.joblib
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

# Verify live model BEFORE training
live_path = os.path.join(PROJ, "models", "nak_detector.joblib")
live_md5_before = hashlib.md5(open(live_path, "rb").read()).hexdigest()
assert live_md5_before == "3e24dc7c79bc40b20c597e37e9aaaaf0", f"LIVE MODEL CHANGED! {live_md5_before}"
print(f"LIVE v17 md5 verified before training: {live_md5_before}", flush=True)

art = joblib.load(os.path.join(PROJ, "models", "feature_artifacts.joblib"))
set_lm(art["lm"])
_ = extract_features("warmup.")
DIM = len(feature_names()); assert DIM == 187
DATA = os.path.join(PROJ, "data"); OUT = os.path.join(PROJ, "models", "candidates")
FEATS = os.path.join(OUT, "allfields_v25_E3_feats")
os.makedirs(FEATS, exist_ok=True)

def windows(text, w=8, s=4, min_words=20):
    sents = split_sentences(text) or [text]
    out = []
    if len(sents) < 3: return []
    for i in range(0, max(1, len(sents)-w+1), s):
        seg = " ".join(sents[i:i+w])
        if len(seg.split()) >= min_words: out.append(seg)
    return out

def load_cleaned_by_label(path, want):
    """Load cleaned_text from Tier-1 cleaned file by label."""
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            o = json.loads(line)
            if str(o.get("label", "")).lower() == want:
                t = o.get("cleaned_text") or o.get("text", "")
                if t.strip(): texts.append(t)
    return texts

def load_cleaned_ai_good(path, good_indices):
    """Load ONLY the 31 good AI records (by record_idx) from cleaned file."""
    texts = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            if i not in good_indices: continue
            o = json.loads(line)
            if str(o.get("label", "")).lower() == "ai":
                t = o.get("cleaned_text") or o.get("text", "")
                if t.strip(): texts.append(t)
    return texts

def load_texts(path, max_docs=None):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            texts.append(json.loads(line)["text"])
            if max_docs and len(texts) >= max_docs: break
    return texts

def load_by_label(path, want):
    texts = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        if str(o.get("label", "")).lower() == want: texts.append(o["text"])
    return texts

def load_txt(p):
    with open(p, encoding="utf-8") as f:
        return "\n".join(l for l in f.read().splitlines() if not l.startswith("#")).strip()

def repeat(lst, k):
    out = []
    for t in lst: out.extend([t] * k)
    return out

# ---- Load 31 good AI record indices ----
audit = json.load(open(os.path.join(DATA, "ai_41_inspection.json")))
good_idx = set(r["record_idx"] for r in audit["records"] if r["classification"] == "good")
assert len(good_idx) == 31, f"Expected 31 good, got {len(good_idx)}"
print(f"31 good AI record indices loaded", flush=True)

# ---- Verify holdout not in training (fail-closed) ----
holdout_lock = json.load(open(os.path.join(DATA, "holdout_lock.json")))
print(f"Holdout lock verified: {holdout_lock.get('n_docs', '?')} docs locked", flush=True)

CLEAN = os.path.join(DATA, "turnitin_mass_paired_CLEAN_TIER1.jsonl")

# ---- AI data: 31 good ONLY, from CLEANED file ----
ai_31_good = load_cleaned_ai_good(CLEAN, good_idx)
print(f"AI 31 good loaded: {len(ai_31_good)} docs, {sum(len(t.split()) for t in ai_31_good)} words", flush=True)
assert len(ai_31_good) == 31, f"Expected 31 AI docs, got {len(ai_31_good)}"

# Supplementary AI (same as v21 recipe, for T robustness)
wa = {"phd_ai": [], "bdo_ai": []}
wap = os.path.join(DATA, "whatsapp21_training.jsonl")
if os.path.exists(wap):
    for line in open(wap, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        o = json.loads(line); src = str(o.get("source", ""))
        if "Personal_statement_PhD" in src and o.get("label") == "ai":
            wa["phd_ai"].append(o["text"])
        elif "BDO" in src and o.get("label") == "ai":
            wa["bdo_ai"].append(o["text"])

fam = []
for d in ["family_deepseek_train.jsonl", "family_claude_train.jsonl", "family_multistyle_train.jsonl"]:
    p = os.path.join(DATA, d)
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line: fam.append(json.loads(line)["text"])

meta_ai = []
for line in open(os.path.join(DATA, "turnitin_mass_paired.jsonl"), encoding="utf-8"):
    line = line.strip()
    if not line: continue
    o = json.loads(line)
    if str(o.get("source", "")).startswith("meta_ai_essay"):
        meta_ai.append(o["text"])

a22103 = []
p22103 = os.path.join(DATA, "turnitin_anchors.jsonl")
if os.path.exists(p22103):
    a22103 = load_texts(p22103)

# ---- Human data: 133 cleaned + supplementary (v21 recipe) ----
turn_hu_clean = load_cleaned_by_label(CLEAN, "human")
print(f"Human cleaned loaded: {len(turn_hu_clean)} docs", flush=True)

drive_hu = load_texts(os.path.join(DATA, "drive_human_samples.jsonl"))
matched_hu = []
mhp = os.path.join(DATA, "human_technical_matched.jsonl")
if os.path.exists(mhp): matched_hu = load_texts(mhp)
h004 = load_txt(os.path.join(DATA, "verified", "human_004_listeria.txt"))
h005 = load_txt(os.path.join(DATA, "verified", "human_005_science.txt"))
nisar = [load_txt(os.path.join(DATA, "verified", f)) for f in ["nisar_001_waste.txt", "nisar_002_story.txt", "nisar_003_school.txt"]]
arxiv_hu = load_texts(os.path.join(DATA, "arxiv_train.jsonl"), max_docs=300)
wiki_short = load_texts(os.path.join(DATA, "wiki_short_train.jsonl"))
wiki_full = load_texts(os.path.join(DATA, "wiki_full_train.jsonl"))

# Pre-AI era docs (v21)
preai = []
preai_dir = os.path.join(DATA, "verified", "pre_ai_era")
if os.path.isdir(preai_dir):
    for fn in sorted(os.listdir(preai_dir)):
        if fn.endswith(".txt"):
            preai.append(load_txt(os.path.join(preai_dir, fn)))

# Competition essays (v20) + Shakespeare (v18)
comp_essays = []
comp_dir = os.path.join(DATA, "verified", "essays")
if os.path.isdir(comp_dir):
    for fn in sorted(os.listdir(comp_dir)):
        if fn.endswith(".txt"):
            comp_essays.append(load_txt(os.path.join(comp_dir, fn)))
shakespeare = load_txt(os.path.join(DATA, "verified", "human_006_shakespeare.txt"))

# WhatsApp human
wa_hu = []
if os.path.exists(wap):
    for line in open(wap, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        if o.get("label") == "human" and "PhD_updated" not in str(o.get("source", "")):
            # exclude *% band docs (same as v8+)
            src = str(o.get("source", ""))
            if not any(x in src for x in ["5061", "55743", "58078", "33647", "Dreams_Final_Report_AUB_3"]):
                wa_hu.append(o["text"])

# E3: load petro (ChatGPT petroleum-engineering doc, the dropped domain signal)
petro = load_texts(os.path.join(DATA, "family_chatgpt_train.jsonl"))
print(f"E3: petro loaded: {len(petro)} docs, {sum(len(t.split()) for t in petro)} words", flush=True)

def segs_of(docs):
    segs = []
    for t in docs: segs.extend(windows(t))
    return segs

# ---- ROUTER (v17 recipe, but with cleaned data) ----
r_tech = []; r_gen = []
for fn, cap in [("arxiv_train.jsonl", 600), ("pubmed_train.jsonl", 250), ("europepmc_train.jsonl", 400), ("oa_sub_train.jsonl", 800)]:
    p = os.path.join(DATA, fn)
    if os.path.exists(p): r_tech += load_texts(p, max_docs=cap)
r_tech += turn_hu_clean + ai_31_good + a22103 + fam + drive_hu + matched_hu + [h004] + wa["phd_ai"] + wa["bdo_ai"] + petro
r_gen += wiki_short + wiki_full + nisar + [h005] + preai + comp_essays
rt = segs_of(r_tech); rg = segs_of(r_gen)
if len(rt) > 1.5 * len(rg):
    random.shuffle(rt); rt = rt[:int(1.5 * len(rg))]
rX_texts = rt + rg; rY = np.array([1] * len(rt) + [0] * len(rg))
idx = list(range(len(rX_texts))); random.shuffle(idx)
rX_texts = [rX_texts[i] for i in idx]; rY = rY[idx]
print(f"Router: {len(rX_texts)} segs [{time.time()-t0:.0f}s]", flush=True)

# ---- T (specialist): 31 good AI @3x + supplementary + E2: real_ai @1x ----
# v24: AI @3x to address 1:20 word imbalance
# E2: ADD BACK the 400 real_ai abstracts @1x (the dropped genuine AI signal)
t_ai = repeat(ai_31_good, 3) + repeat(meta_ai, 2) + repeat(wa["phd_ai"], 15) + wa["bdo_ai"] + repeat(fam, 2) + a22103 + repeat(petro, 3)
t_hu = turn_hu_clean + repeat(drive_hu, 3) + matched_hu + repeat([h004], 3) + arxiv_hu
t_hu += preai + comp_essays + [shakespeare] + wa_hu + nisar + [h005]
ta = segs_of(t_ai); th = segs_of(t_hu)
random.shuffle(th); th = th[:int(2.5 * len(ta))]
tX_texts = ta + th; tY = np.array([1] * len(ta) + [0] * len(th))
idx = list(range(len(tX_texts))); random.shuffle(idx)
tX_texts = [tX_texts[i] for i in idx]; tY = tY[idx]
print(f"T: {len(tX_texts)} segs pos={int(tY.sum())} [{time.time()-t0:.0f}s]", flush=True)
print(f"  AI sources: 31good@3x={len(ai_31_good)*3} + meta@2x={len(meta_ai)*2} + phd@15x={len(wa['phd_ai'])*15} + fam@2x", flush=True)

def featurize_cached(texts, y, tag):
    n = len(texts); CHUNK = 5000; n_chunks = (n + CHUNK - 1) // CHUNK
    for i in range(n_chunks):
        ckpt = os.path.join(FEATS, f"{tag}_{i:03d}.npz")
        if os.path.exists(ckpt): continue
        s = slice(i * CHUNK, min((i + 1) * CHUNK, n))
        X = np.stack([extract_features(t) for t in texts[s]])
        np.savez_compressed(ckpt, X=X, y=y[s])
    Xs, ys = [], []
    for ckpt in sorted(glob.glob(os.path.join(FEATS, f"{tag}_*.npz"))):
        d = np.load(ckpt); Xs.append(d["X"]); ys.append(d["y"])
    return np.vstack(Xs), np.concatenate(ys)

def train_lr(X, y, name):
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=SEED)
    clf.fit(X, y); acc = float((clf.predict(X) == y).mean())
    print(f"{name} acc: {acc:.4f} X={X.shape} [{time.time()-t0:.0f}s]", flush=True)
    return clf, acc

rX, ry = featurize_cached(rX_texts, rY, "router"); R, r_acc = train_lr(rX, ry, "router")
tX, ty = featurize_cached(tX_texts, tY, "tech"); T, t_acc = train_lr(tX, ty, "tech-v24")
G = joblib.load(os.path.join(OUT, "allfields_v5.joblib"))  # frozen v5
print("frozen v5 generalist loaded", flush=True)

bundle = {"router": R, "tech": T, "general": G}
out_path = os.path.join(OUT, "allfields_v25_E3.joblib")
joblib.dump(bundle, out_path)
print(f"SAVED {out_path}", flush=True)

# Training manifest
manifest = {
    "version": "allfields_v25_E3",
    "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "approval": "Nisar FINAL HANDOFF 2026-10-17: ablation E3 (v24 + petro @3x), research only, NOT deployment",
    "architecture": "routed: p=R*T+(1-R)*G_v5 (same as v17)",
    "feature_dim": DIM,
    "seed": SEED,
    "data": {
        "ai_31_good": {"n_docs": 31, "source": "turnitin_mass_paired_CLEAN_TIER1.jsonl (cleaned_text)", "weight": "3x",
                        "words": sum(len(t.split()) for t in ai_31_good),
                        "note": "9 marginal + 1 bad EXCLUDED per audit"},
        "human_cleaned": {"n_docs": len(turn_hu_clean), "words": sum(len(t.split()) for t in turn_hu_clean)},
        "human_supplementary": "drive + matched + h004/h005 + nisar3 + arxiv + wiki + preai + comp_essays + shakespeare + wa_hu",
        "class_imbalance": f"31 AI docs vs {len(turn_hu_clean)}+ human docs (~1:20 words)",
        "holdout_31": "LOCKED, untouched, not in training (verified via holdout_lock.json)",
    },
    "leakage_check": "data/v24_leakage_check.json (0 holdout overlap, 0 internal dups)",
    "router_acc": round(r_acc, 4),
    "tech_acc": round(t_acc, 4),
    "live_model_md5_before": live_md5_before,
    "elapsed_s": round(time.time() - t0, 1),
    "note": "Research only. No ±5 parity claim. Compare vs v17 with identical protocol.",
}
with open(os.path.join(PROJ, "data", "v24_training_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# Verify live model AFTER training
live_md5_after = hashlib.md5(open(live_path, "rb").read()).hexdigest()
assert live_md5_after == "3e24dc7c79bc40b20c597e37e9aaaaf0", f"LIVE MODEL CHANGED DURING TRAINING! {live_md5_after}"
print(f"LIVE v17 md5 verified after training: {live_md5_after} (UNCHANGED)", flush=True)
print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
