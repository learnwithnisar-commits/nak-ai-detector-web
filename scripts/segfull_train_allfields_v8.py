#!/usr/bin/env python3
"""Segment-level training: allfields_v8 — v6 routed ensemble + FIXED WhatsApp labels.

v8 strategy (builds on v7, fixes v7's catastrophic human-FPR regression):
  Same domain-routed ensemble architecture as v6/v7:
  - Router R:      P(technical register | x)
  - Specialist T:  P(AI | x, technical)
  - Generalist G:  P(AI | x, general)  (fresh)

  v7 LESSON: the five *% band WhatsApp docs (5061, 55743, 58078, 33647,
  Dreams_AUB3) are 81-99% HUMAN but were labeled "ai" — this poisoned human
  FPR catastrophically (Nisar's own text 0.5% -> 80%!).

  v8 FIX: EXCLUDE the five *% docs from AI training entirely.
  Use ONLY high-confidence WhatsApp labels:
  - AI: Personal_statement_PhD (100%), BDO_Lebanon (22% @~0.3 weight)
  - Human: PhD_updated (0%), Arab_Design (0%), 77701 (0%), Hospital_AR1 (0%)
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
FEATS = os.path.join(OUT, "allfields_v8_feats")
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


def load_meta_essays(path):
    """The 10 verified Meta AI essays (source startswith meta_ai_essay)."""
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


def load_whatsapp_v8(path):
    """WhatsApp verified pairs — v8 FIXED labels.

    EXCLUDES the five *% band docs (5061, 55743, 58078, 33647, Dreams_AUB3):
    they are 81-99% human but were labeled "ai" in v7, poisoning human FPR.

    Returns (ai_full, ai_mid, human):
      ai_full: Personal_statement_PhD (100% AI)
      ai_mid:  BDO_Lebanon (22% AI — use at reduced weight)
      human:   PhD_updated, Arab_Design, 77701, Hospital_AR1 (all 0%)
    """
    EXCLUDE = {"5061", "55743", "58078", "33647", "Dreams_Final_Report_AUB_3",
               "Dreams"}
    ai_full, ai_mid, hu = [], [], []
    if not os.path.exists(path):
        return ai_full, ai_mid, hu
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            src = str(o.get("source", ""))
            if any(x in src for x in EXCLUDE):
                continue  # v8 fix: drop *% band docs entirely
            lab = str(o.get("label", "")).lower()
            if lab == "ai":
                if "BDO_Lebanon" in src or "BDO" in src:
                    ai_mid.append(o["text"])
                else:
                    ai_full.append(o["text"])
            elif lab == "human":
                hu.append(o["text"])
    return ai_full, ai_mid, hu


def load_whatsapp(path):
    """Legacy wrapper (kept for compatibility; v8 uses load_whatsapp_v8)."""
    ai_full, ai_mid, hu = load_whatsapp_v8(path)
    return ai_full + ai_mid, hu


def load_txt_nocomments(path):
    with open(path, encoding="utf-8") as f:
        body = "\n".join(l for l in f.read().splitlines() if not l.startswith("#"))
    return body.strip()


def repeat(lst, k):
    out = []
    for t in lst:
        out.extend([t] * k)
    return out


def fam_texts():
    docs = []
    for d in ["family_deepseek_train.jsonl", "family_claude_train.jsonl",
              "family_multistyle_train.jsonl"]:
        p = os.path.join(DATA, d)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        docs.append(json.loads(line)["text"])
    return docs


# ---------- shared doc pools ----------
turn_ai = load_texts_by_label(os.path.join(DATA, "turnitin_mass_paired.jsonl"), "ai")
turn_hu = load_texts_by_label(os.path.join(DATA, "turnitin_mass_paired.jsonl"), "human")
meta_essays = load_meta_essays(os.path.join(DATA, "turnitin_mass_paired.jsonl"))
wa_ai_full, wa_ai_mid, wa_hu = load_whatsapp_v8(os.path.join(DATA, "whatsapp21_training.jsonl"))
wa_ai = wa_ai_full + wa_ai_mid  # combined for router (register signal only)
a22103 = load_texts(os.path.join(DATA, "turnitin_anchors.jsonl"))
p22103_ai = load_texts_by_label(os.path.join(DATA, "turnitin_22103_paired.jsonl"), "ai")
p22103_hu = load_texts_by_label(os.path.join(DATA, "turnitin_22103_paired.jsonl"), "human")
drive_hu = load_texts(os.path.join(DATA, "drive_human_samples.jsonl"))
matched_hu = load_texts(os.path.join(DATA, "human_technical_matched.jsonl"))
fam_ai = fam_texts()
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
print(f"pools: turn_ai={len(turn_ai)} turn_hu={len(turn_hu)} META_ESSAYS={len(meta_essays)} "
      f"WA_AI_FULL={len(wa_ai_full)} WA_AI_MID={len(wa_ai_mid)} WA_HU={len(wa_hu)} "
      f"drive={len(drive_hu)} fam={len(fam_ai)} wiki={len(wiki_short)+len(wiki_full)} "
      f"[{time.time()-t0:.0f}s]", flush=True)
assert len(meta_essays) == 10, f"expected 10 meta essays, got {len(meta_essays)}"
assert len(wa_ai_full) == 1 and len(wa_ai_mid) == 1 and len(wa_hu) == 4, \
    f"expected 1/1/4 whatsapp docs (v8 fixed), got {len(wa_ai_full)}/{len(wa_ai_mid)}/{len(wa_hu)}"


def segs_of(docs):
    segs = []
    for t in docs:
        segs.extend(windows(t))
    return segs


def build_xy(seg_docs_pos, seg_docs_neg, doc_docs_pos, doc_docs_neg,
             neg_cap_mult=2.5, tag=""):
    """Multi-scale: segments (balanced w/ cap) + whole docs (@1x each)."""
    p_segs = segs_of(seg_docs_pos)
    n_segs = segs_of(seg_docs_neg)
    random.shuffle(n_segs)
    n_segs = n_segs[:int(neg_cap_mult * len(p_segs))]
    texts = p_segs + n_segs
    y = [1] * len(p_segs) + [0] * len(n_segs)
    texts += doc_docs_pos + doc_docs_neg
    y += [1] * len(doc_docs_pos) + [0] * len(doc_docs_neg)
    y = np.array(y, dtype=int)
    idx = list(range(len(texts))); random.shuffle(idx)
    texts = [texts[i] for i in idx]; y = y[idx]
    print(f"{tag}: total={len(texts)} pos={int(y.sum())} "
          f"(segs p={len(p_segs)}/n={len(n_segs)}, docs p={len(doc_docs_pos)}/n={len(doc_docs_neg)})",
          flush=True)
    return texts, y


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


# ================= ROUTER (multi-scale) =================
# meta essays are general register -> negative (general) side
# whatsapp docs: mixed registers, add to both sides by label
r_pos_seg = (load_texts(os.path.join(DATA, "arxiv_train.jsonl"), max_docs=600)
             + load_texts(os.path.join(DATA, "pubmed_train.jsonl"), max_docs=250)
             + load_texts(os.path.join(DATA, "europepmc_train.jsonl"), max_docs=400)
             + load_texts(os.path.join(DATA, "oa_sub_train.jsonl"), max_docs=800)
             + turn_ai + turn_hu + a22103 + drive_hu + matched_hu + fam_ai
             + petro_ai + real_ai + [h004] + wa_ai + wa_hu)
r_neg_seg = wiki_short + wiki_full + nisar_txts + [h005] + meta_essays
rX_texts, rY = build_xy(r_pos_seg, r_neg_seg, r_pos_seg, r_neg_seg,
                        neg_cap_mult=1.5, tag="router")
rX, ry = featurize_cached(rX_texts, rY, "router")
R, r_acc = train_lr(rX, ry, "router")

# ================= TECH SPECIALIST (multi-scale) =================
t_pos_seg = (repeat(turn_ai, 3) + a22103 + p22103_ai
             + repeat(fam_ai, 2) + repeat(petro_ai, 3) + real_ai
             + repeat(meta_essays, 2)                      # general-AI signal
             + repeat(wa_ai_full, 3)                      # v8: PhD 100% @3x
             + wa_ai_mid)                                 # v8: BDO 22% @1x (reduced weight)
t_neg_seg = (turn_hu + repeat(drive_hu, 3) + matched_hu + [h004] * 3
             + p22103_hu + arxiv_hu + wiki_full + wiki_short
             + repeat(wa_hu, 3))                          # NEW: whatsapp human counterbalance
t_pos_doc = turn_ai + a22103 + p22103_ai + fam_ai + petro_ai + meta_essays + wa_ai_full + wa_ai_mid
t_neg_doc = turn_hu + drive_hu + matched_hu + [h004, h005] + p22103_hu + wiki_full + wiki_short + wa_hu
tX_texts, tY = build_xy(t_pos_seg, t_neg_seg, t_pos_doc, t_neg_doc,
                        neg_cap_mult=2.5, tag="tech")
tX, ty = featurize_cached(tX_texts, tY, "tech")
T, t_acc = train_lr(tX, ty, "tech-specialist")

# ================= GENERALIST (multi-scale, fresh) =================
# v8: meta essays @4x + PhD 100% @3x + BDO 22% @1x (WhatsApp *% docs EXCLUDED)
g_pos_seg = (turn_ai + fam_ai + repeat(ai_txts, 3) + repeat(meta_essays, 4)
             + repeat(wa_ai_full, 3) + wa_ai_mid)
g_neg_seg = (repeat(wiki_short, 3) + repeat(wiki_full, 3) + repeat(nisar_txts, 3)
             + [h005] * 3 + turn_hu + repeat(wa_hu, 3))
g_pos_doc = turn_ai + fam_ai + ai_txts + repeat(meta_essays, 2) + wa_ai_full + wa_ai_mid
g_neg_doc = wiki_short + wiki_full + nisar_txts + [h005] + turn_hu + wa_hu
gX_texts, gY = build_xy(g_pos_seg, g_neg_seg, g_pos_doc, g_neg_doc,
                        neg_cap_mult=2.0, tag="general")
gX, gy = featurize_cached(gX_texts, gY, "general")
G, g_acc = train_lr(gX, gy, "generalist")

# ================= probes =================
def routed_proba(text):
    f = np.asarray(extract_features(text), dtype=np.float64).reshape(1, -1)
    r = float(R.predict_proba(f)[0][1])
    pt = float(T.predict_proba(f)[0][1])
    pg = float(G.predict_proba(f)[0][1])
    return r * pt + (1 - r) * pg, r, pt, pg

probe_tech_ai = ("In this study we investigate the thermophysical properties of hybrid "
                 "nanofluids under turbulent flow conditions. The experimental results "
                 "demonstrate a significant enhancement in heat transfer coefficient.")
probe_gen = "aj hum school mein science ka project banayenge jo pani se bijli banata hai"
for name, pr in [("tech-probe(seg)", probe_tech_ai), ("general-probe(seg)", probe_gen)]:
    p, r, pt, pg = routed_proba(pr)
    print(f"  {name}: routed={p:.3f} R={r:.3f} T={pt:.3f} G={pg:.3f}", flush=True)
long_hu = " ".join(drive_hu[0].split()[:1500])
p, r, pt, pg = routed_proba(long_hu)
print(f"  long-doc-human: routed={p:.3f} R={r:.3f} T={pt:.3f} G={pg:.3f}", flush=True)
# meta essay probe (should be high AI)
p, r, pt, pg = routed_proba(meta_essays[0])
print(f"  meta-essay-probe: routed={p:.3f} R={r:.3f} T={pt:.3f} G={pg:.3f}", flush=True)

# ================= save =================
bundle = {"router": R, "tech": T, "general": G}
out_model = os.path.join(OUT, "allfields_v8.joblib")
joblib.dump(bundle, out_model)
print(f"SAVED {out_model}", flush=True)
meta = {
    "seed": SEED, "feature_dim": DIM,
    "architecture": "routed-ensemble multi-scale: p=R*T+(1-R)*G; segs+wholedocs",
    "router_acc": round(r_acc, 4), "tech_acc": round(t_acc, 4), "gen_acc": round(g_acc, 4),
    "meta_essays": len(meta_essays),
    "whatsapp_ai_full": len(wa_ai_full), "whatsapp_ai_mid": len(wa_ai_mid),
    "whatsapp_hu": len(wa_hu),
    "elapsed_s": round(time.time() - t0, 1),
    "note": ("v8: v7 architecture + FIXED WhatsApp labels; "
             "*% band docs (5061/55743/58078/33647/Dreams) EXCLUDED from AI training; "
             "PhD 100% @3x, BDO 22% @1x; "
             "4 verified humans as FPR counterbalance."),
}
with open(os.path.join(OUT, "allfields_v8_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("META:", json.dumps(meta), flush=True)

# live-model integrity guard
import hashlib
live = os.path.join(PROJ, "models", "nak_detector.joblib")
h = hashlib.md5(open(live, "rb").read()).hexdigest()
print(f"LIVE MODEL MD5: {h} {'OK' if h == '9eeba80e03d06c7db8b35796965746c9' else 'MISMATCH!'}",
      flush=True)
print(f"DONE [{time.time()-t0:.0f}s]", flush=True)
