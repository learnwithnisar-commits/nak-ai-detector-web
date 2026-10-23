#!/usr/bin/env python3
"""Full evaluation for allfields_v16 routed-ensemble candidate.

Gates (ALL must pass):
1. Full Paper anchor: 75-85% (Turnitin 80%)
2. 22103 anchor: 52-62% (Turnitin 57%)
3. Personal_statement_PhD anchor: 95-100% (Turnitin 100%)  [NEW v7]
4. 8-anchor suite: mean gap < 15, >= 4/8 within ±5
5. Human FPR < 5% on verified humans (Nisar x3 + friends x2) AND Drive samples
6. Wikipedia: < 10% mean

Output: models/candidates/allfields_v16_eval_report.json

DOES NOT touch the live model.
"""
import json, os, sys, time
from pathlib import Path

PROJ = os.path.expanduser("~/workspace/ai-detector")
sys.path.insert(0, PROJ)
import joblib, numpy as np
from src.features import extract_features, set_lm
from app.scan import SentenceScorer, split_sentences
from app.parse import parse_document

t0 = time.time()
DATA = os.path.join(PROJ, "data")
FILES = "/home/hatch/workspace/user/files"

art = joblib.load(os.path.join(PROJ, "models", "feature_artifacts.joblib"))
set_lm(art["lm"])
bundle = joblib.load(os.path.join(PROJ, "models", "candidates", "allfields_v16.joblib"))
R, T, G = bundle["router"], bundle["tech"], bundle["general"]
print("v6 bundle loaded (router + tech-specialist + frozen-v5 generalist)", flush=True)

report = {"candidate": "allfields_v16", "date": time.strftime("%Y-%m-%dT%H:%M:%S")}


def routed_proba(text):
    f = np.asarray(extract_features(text), dtype=np.float64).reshape(1, -1)
    r = float(R.predict_proba(f)[0][1])
    pt = float(T.predict_proba(f)[0][1])
    pg = float(G.predict_proba(f)[0][1])
    return r * pt + (1.0 - r) * pg


def score_text_proba(text):
    return routed_proba(text)


def score_doc_pipeline(path):
    ext = Path(path).suffix.lower()

    def st(t):
        return min(max(routed_proba(t), 0.0), 1.0)

    scorer = SentenceScorer(st)
    sent_sum, sent_count = 0.0, 0
    for unit in parse_document(Path(path), ext):
        if unit["tag"] == "p" and unit["text"].strip():
            for s in split_sentences(unit["text"]) or [unit["text"].strip()]:
                scorer.feed(s)
    scorer.finish()
    while scorer.finalized:
        _, s = scorer.finalized.popleft()
        sent_sum += s; sent_count += 1
    return round(sent_sum / sent_count * 100, 1) if sent_count else 0.0


# ---- 1. Full Paper anchor ----
fp_path = os.path.join(FILES, "Full-Paper-sincere-conf.docx")
fp_score = score_doc_pipeline(fp_path) if os.path.exists(fp_path) else None
fp_pass = fp_score is not None and 75.0 <= fp_score <= 85.0
report["full_paper"] = {
    "turnitin": 80.0, "v16": fp_score,
    "gap": round(abs(fp_score - 80.0), 1) if fp_score is not None else None,
    "target_band": "75-85", "PASS": bool(fp_pass),
}
print(f"Full Paper: v8={fp_score}% (Turnitin 80%) {'PASS' if fp_pass else 'FAIL'}", flush=True)

# ---- 2. 22103 anchor ----
a22103_path = os.path.join(FILES, "22103.docx")
a22103_score = score_doc_pipeline(a22103_path) if os.path.exists(a22103_path) else None
a22103_pass = a22103_score is not None and 52.0 <= a22103_score <= 62.0
report["anchor_22103"] = {
    "turnitin": 57.0, "v16": a22103_score,
    "gap": round(abs(a22103_score - 57.0), 1) if a22103_score is not None else None,
    "target_band": "52-62", "PASS": bool(a22103_pass),
}
print(f"22103: v8={a22103_score}% (Turnitin 57%) {'PASS' if a22103_pass else 'FAIL'}", flush=True)

# ---- 2b. Personal_statement_PhD anchor [NEW v7] ----
phd_path = os.path.join(FILES, "Personal_statement_PhD.docx")
phd_score = score_doc_pipeline(phd_path) if os.path.exists(phd_path) else None
phd_pass = phd_score is not None and 95.0 <= phd_score <= 100.0
report["anchor_phd"] = {
    "turnitin": 100.0, "v16": phd_score,
    "gap": round(abs(phd_score - 100.0), 1) if phd_score is not None else None,
    "target_band": "95-100", "PASS": bool(phd_pass),
}
print(f"PhD_statement: v8={phd_score}% (Turnitin 100%) {'PASS' if phd_pass else 'FAIL'}", flush=True)

# ---- 3. 8-anchor suite ----
ANCHORS = [
    ("Listeria", f"{FILES}/Listeria_goat_MANUSCRIPT_IMPROVED_CLEAN.docx", 54),
    ("XDR_Typhoid", f"{FILES}/XDR_Typhoid_Pakistan_PJMR_Format.docx", 64),
    ("AIGC", f"{FILES}/aigc_download_2027795315876175872_aigc_rate.docx", 43),
    ("ash_coal", f"{PROJ}/data/real_ai/ash_coal_essay_chatgpt.txt", 100),
    ("Competition", f"{FILES}/upload.docx", "*"),
    ("Junlan_Li", f"{FILES}/Essay-_Junlan_Li-25041323.docx", "*"),
    ("Liu_Yueru", f"{FILES}/Liu_Yueru_Vicky__FA1_Yueru__Vicky__LIU.docx", 60),
    ("ES1002", f"{FILES}/ES1002_Essay.docx", 0),
]


def gap_fn(ours, target):
    if target == "*":
        return 0.0 if 0.0 < ours < 20.0 else round(min(abs(ours - 1), abs(ours - 19)), 1)
    return round(abs(ours - target), 1)


anchor_rows = []
for name, path, target in ANCHORS:
    if not os.path.exists(path):
        anchor_rows.append({"name": name, "missing": True, "target": target})
        continue
    pct = score_doc_pipeline(path)
    g = gap_fn(pct, target)
    anchor_rows.append({"name": name, "v16": pct, "target": target, "gap": g})
    print(f"  {name:12s} v8={pct:6.1f} target={str(target):>4} gap={g}", flush=True)

scored = [r for r in anchor_rows if "gap" in r]
mean_gap = round(sum(r["gap"] for r in scored) / len(scored), 2) if scored else 999
within5 = sum(1 for r in scored if r["gap"] <= 5.0)
anchor_pass = mean_gap < 15.0 and within5 >= 4
report["anchors_8"] = {
    "mean_gap": mean_gap, "within_pm5": f"{within5}/{len(scored)}",
    "gate": "mean<15 AND >=4/8 within ±5", "PASS": bool(anchor_pass),
    "per_anchor": anchor_rows,
}
print(f"8-anchor: mean_gap={mean_gap}, within±5={within5}/{len(scored)} {'PASS' if anchor_pass else 'FAIL'}", flush=True)


# ---- 4. Human FPR gates ----
def fpr_gate_jsonl(name, jsonl_path, threshold=20.0):
    texts = [json.loads(l)["text"] for l in open(jsonl_path, encoding="utf-8") if l.strip()]
    pcts = np.array([score_text_proba(t) * 100.0 for t in texts])
    return {
        "n": len(texts), "mean_ai_pct": round(float(np.mean(pcts)), 2),
        "breach_frac_ge20": round(float(np.mean(pcts >= threshold)), 4),
        "max_ai_pct": round(float(np.max(pcts)), 2),
    }


def fpr_gate_txts(name, txt_paths, threshold=20.0):
    texts = []
    for p in txt_paths:
        with open(p, encoding="utf-8") as f:
            body = "\n".join(l for l in f.read().splitlines() if not l.startswith("#"))
        texts.append(body.strip())
    pcts = np.array([score_text_proba(t) * 100.0 for t in texts])
    return {
        "n": len(texts),
        "per_sample": {os.path.basename(p): round(float(pct), 1)
                       for p, pct in zip(txt_paths, pcts)},
        "mean_ai_pct": round(float(np.mean(pcts)), 2),
        "breach_frac_ge20": round(float(np.mean(pcts >= threshold)), 4),
        "max_ai_pct": round(float(np.max(pcts)), 2),
    }


gates = {}

vh = fpr_gate_txts("verified_humans", [
    os.path.join(DATA, "verified", "nisar_001_waste.txt"),
    os.path.join(DATA, "verified", "nisar_002_story.txt"),
    os.path.join(DATA, "verified", "nisar_003_school.txt"),
    os.path.join(DATA, "verified", "human_004_listeria.txt"),
    os.path.join(DATA, "verified", "human_005_science.txt"),
])
vh_pass = vh["breach_frac_ge20"] < 0.05
vh["gate"] = "breach < 5% (Nisar x3 + friends x2)"
vh["PASS"] = bool(vh_pass)
gates["verified_humans"] = vh
print(f"verified_humans: n={vh['n']} breach={vh['breach_frac_ge20']} per={vh['per_sample']} {'PASS' if vh_pass else 'FAIL'}", flush=True)

dh = fpr_gate_jsonl("drive_human", os.path.join(DATA, "drive_human_samples.jsonl"))
dh_pass = dh["breach_frac_ge20"] < 0.05
dh["gate"] = "breach < 5%"
dh["PASS"] = bool(dh_pass)
gates["drive_human"] = dh
print(f"drive_human: n={dh['n']} breach={dh['breach_frac_ge20']} max={dh['max_ai_pct']}% {'PASS' if dh_pass else 'FAIL'}", flush=True)

wk_path = os.path.join(DATA, "wiki_human_eval.jsonl")
if os.path.exists(wk_path):
    wk = fpr_gate_jsonl("wikipedia", wk_path)
    wk_pass = wk["mean_ai_pct"] < 10.0
    wk["gate"] = "mean < 10%"
    wk["PASS"] = bool(wk_pass)
else:
    wk = {"missing": True}
    wk_pass = False
gates["wikipedia"] = wk
print(f"wikipedia: n={wk.get('n')} mean={wk.get('mean_ai_pct')}% {'PASS' if wk_pass else 'FAIL'}", flush=True)

rh_path = os.path.join(DATA, "real_human_eval.jsonl")
if os.path.exists(rh_path):
    rh = fpr_gate_jsonl("real_human", rh_path)
    gates["real_human_info_only"] = rh
    print(f"real_human (info): n={rh['n']} breach={rh['breach_frac_ge20']}", flush=True)

report["human_fpr_gates"] = gates

# ---- overall verdict ----
all_pass = bool(fp_pass and a22103_pass and phd_pass and anchor_pass and vh_pass and dh_pass and wk_pass)
report["overall_verdict"] = "DEPLOY_CANDIDATE" if all_pass else "DO_NOT_DEPLOY"
report["gates_summary"] = {
    "full_paper_75_85": bool(fp_pass),
    "anchor_22103_52_62": bool(a22103_pass),
    "anchor_phd_95_100": bool(phd_pass),
    "anchors_mean_lt15_and_4of8": bool(anchor_pass),
    "verified_humans_fpr_lt5": bool(vh_pass),
    "drive_human_fpr_lt5": bool(dh_pass),
    "wikipedia_lt10": bool(wk_pass),
}
report["live_model_md5"] = "9eeba80e03d06c7db8b35796965746c9"
report["live_model_untouched"] = True
report["elapsed_s"] = round(time.time() - t0, 1)

out = os.path.join(PROJ, "models", "candidates", "allfields_v16_eval_report.json")
json.dump(report, open(out, "w"), indent=1)
print(f"\nVERDICT: {report['overall_verdict']}", flush=True)
print(f"saved {out}", flush=True)
