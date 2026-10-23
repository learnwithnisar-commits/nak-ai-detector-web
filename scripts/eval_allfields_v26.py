#!/usr/bin/env python3
"""Evaluation for allfields_v26 (controlled combination experiment) vs v17 baseline.

Per Nisar's approval 2026-10-17:
- Compare candidate with v17 using IDENTICAL preprocessing, scoring aggregation, evaluation protocol
- Report MAE, within-5, within-10, signed bias, human FPR where valid reference labels exist
- Document-level evaluation (each document = one data point)
- NO ±5 parity claims without independent paired AI test set
- Research only, NOT deployment. Live v17 untouched.

Gates (same as v21 protocol):
1. Full Paper anchor: 75-85% (Turnitin 80%)
2. 22103 anchor: 52-62% (Turnitin 57%)
3. Personal_statement_PhD anchor: 95-100% (Turnitin 100%)
4. 8-anchor suite: mean gap < 15, >= 4/8 within ±5
5. Human FPR < 5% on verified humans AND Drive samples
6. Wikipedia: < 10% mean

Output: models/candidates/allfields_v26_eval_report.json
DOES NOT touch the live model.
"""
import json, os, sys, time, hashlib
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

# Verify live model untouched
live_md5 = hashlib.md5(open(os.path.join(PROJ, "models", "nak_detector.joblib"), "rb").read()).hexdigest()
assert live_md5 == "3e24dc7c79bc40b20c597e37e9aaaaf0", f"LIVE MODEL CHANGED! {live_md5}"
print(f"LIVE v17 md5 verified: {live_md5}", flush=True)

art = joblib.load(os.path.join(PROJ, "models", "feature_artifacts.joblib"))
set_lm(art["lm"])

# Load V26 candidate
BN26 = joblib.load(os.path.join(PROJ, "models", "candidates", "allfields_v26.joblib"))
RN26, TN26, GN26 = BN26["router"], BN26["tech"], BN26["general"]
print("V26 bundle loaded", flush=True)

# Load v17 baseline for side-by-side comparison
b17 = joblib.load(os.path.join(PROJ, "models", "candidates", "allfields_v17.joblib"))
R17, T17, G17 = b17["router"], b17["tech"], b17["general"]
print("v17 baseline bundle loaded", flush=True)

report = {
    "candidate": "allfields_v26",
    "baseline": "allfields_v17",
    "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "approval": "Nisar 2026-10-17: ONE controlled experimental run, research only, NOT deployment",
    "protocol": "identical preprocessing, scoring aggregation (document-level mean), evaluation as v17/v21",
    "honest_note": "Legacy eval docs are in-sample/optimistic (V22 finding). "
                    "No ±5 parity claim without independent paired AI test set.",
}


def make_proba(R, T, G):
    def routed_proba(text):
        f = np.asarray(extract_features(text), dtype=np.float64).reshape(1, -1)
        r = float(R.predict_proba(f)[0][1])
        pt = float(T.predict_proba(f)[0][1])
        pg = float(G.predict_proba(f)[0][1])
        return r * pt + (1.0 - r) * pg
    return routed_proba


proba26 = make_proba(RN26, TN26, GN26)
proba17 = make_proba(R17, T17, G17)


def score_doc_pipeline(path, proba_fn):
    ext = Path(path).suffix.lower()

    def st(t):
        return min(max(proba_fn(t), 0.0), 1.0)

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


def score_text_proba(text, proba_fn):
    return proba_fn(text)


# ---- 1. Full Paper anchor ----
fp_path = os.path.join(FILES, "Full-Paper-sincere-conf.docx")
fp26 = score_doc_pipeline(fp_path, proba26) if os.path.exists(fp_path) else None
fp17 = score_doc_pipeline(fp_path, proba17) if os.path.exists(fp_path) else None
fp_pass = fp26 is not None and 75.0 <= fp26 <= 85.0
report["full_paper"] = {
    "turnitin": 80.0, "V26": fp26, "v17": fp17,
    "gap_V26": round(abs(fp26 - 80.0), 1) if fp26 is not None else None,
    "gap_v17": round(abs(fp17 - 80.0), 1) if fp17 is not None else None,
    "target_band": "75-85", "PASS": bool(fp_pass),
}
print(f"Full Paper: V26={fp26}% v17={fp17}% (Turnitin 80%) {'PASS' if fp_pass else 'FAIL'}", flush=True)

# ---- 2. 22103 anchor ----
a22103_path = os.path.join(FILES, "22103.docx")
a26 = score_doc_pipeline(a22103_path, proba26) if os.path.exists(a22103_path) else None
a17 = score_doc_pipeline(a22103_path, proba17) if os.path.exists(a22103_path) else None
a_pass = a26 is not None and 52.0 <= a26 <= 62.0
report["anchor_22103"] = {
    "turnitin": 57.0, "V26": a26, "v17": a17,
    "gap_V26": round(abs(a26 - 57.0), 1) if a26 is not None else None,
    "gap_v17": round(abs(a17 - 57.0), 1) if a17 is not None else None,
    "target_band": "52-62", "PASS": bool(a_pass),
}
print(f"22103: V26={a26}% v17={a17}% (Turnitin 57%) {'PASS' if a_pass else 'FAIL'}", flush=True)

# ---- 2b. PhD anchor ----
phd_path = os.path.join(FILES, "Personal_statement_PhD.docx")
phd26 = score_doc_pipeline(phd_path, proba26) if os.path.exists(phd_path) else None
phd17 = score_doc_pipeline(phd_path, proba17) if os.path.exists(phd_path) else None
phd_pass = phd26 is not None and 95.0 <= phd26 <= 100.0
report["anchor_phd"] = {
    "turnitin": 100.0, "V26": phd26, "v17": phd17,
    "gap_V26": round(abs(phd26 - 100.0), 1) if phd26 is not None else None,
    "gap_v17": round(abs(phd17 - 100.0), 1) if phd17 is not None else None,
    "target_band": "95-100", "PASS": bool(phd_pass),
}
print(f"PhD: V26={phd26}% v17={phd17}% (Turnitin 100%) {'PASS' if phd_pass else 'FAIL'}", flush=True)

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
    p26 = score_doc_pipeline(path, proba26)
    p17 = score_doc_pipeline(path, proba17)
    g26 = gap_fn(p26, target)
    g17 = gap_fn(p17, target)
    anchor_rows.append({"name": name, "V26": p26, "v17": p17, "target": target,
                        "gap_V26": g26, "gap_v17": g17})
    print(f"  {name:12s} V26={p26:6.1f} v17={p17:6.1f} target={str(target):>4} gapV26={g26} gap17={g17}", flush=True)

scored = [r for r in anchor_rows if "gap_V26" in r]
mean_gap26 = round(sum(r["gap_V26"] for r in scored) / len(scored), 2) if scored else 999
mean_gap17 = round(sum(r["gap_v17"] for r in scored) / len(scored), 2) if scored else 999
within5_26 = sum(1 for r in scored if r["gap_V26"] <= 5.0)
within5_17 = sum(1 for r in scored if r["gap_v17"] <= 5.0)
anchor_pass = mean_gap26 < 15.0 and within5_26 >= 4
report["anchors_8"] = {
    "mean_gap_V26": mean_gap26, "mean_gap_v17": mean_gap17,
    "within_pm5_V26": f"{within5_26}/{len(scored)}", "within_pm5_v17": f"{within5_17}/{len(scored)}",
    "gate": "mean<15 AND >=4/8 within ±5", "PASS": bool(anchor_pass),
    "per_anchor": anchor_rows,
}
print(f"8-anchor: v24 mean_gap={mean_gap26} within±5={within5_26}/{len(scored)} | "
      f"v17 mean_gap={mean_gap17} within±5={within5_17}/{len(scored)} {'PASS' if anchor_pass else 'FAIL'}", flush=True)

# ---- 4. Human FPR gates ----
def fpr_gate_jsonl(jsonl_path, proba_fn, threshold=20.0):
    texts = [json.loads(l)["text"] for l in open(jsonl_path, encoding="utf-8") if l.strip()]
    pcts = np.array([proba_fn(t) * 100.0 for t in texts])
    return {
        "n": len(texts), "mean_ai_pct": round(float(np.mean(pcts)), 2),
        "breach_frac_ge20": round(float(np.mean(pcts >= threshold)), 4),
        "max_ai_pct": round(float(np.max(pcts)), 2),
    }


def fpr_gate_txts(txt_paths, proba_fn, threshold=20.0):
    texts = []
    for p in txt_paths:
        with open(p, encoding="utf-8") as f:
            body = "\n".join(l for l in f.read().splitlines() if not l.startswith("#"))
        texts.append(body.strip())
    pcts = np.array([proba_fn(t) * 100.0 for t in texts])
    return {
        "n": len(texts),
        "per_sample": {os.path.basename(p): round(float(pct), 1) for p, pct in zip(txt_paths, pcts)},
        "mean_ai_pct": round(float(np.mean(pcts)), 2),
        "breach_frac_ge20": round(float(np.mean(pcts >= threshold)), 4),
        "max_ai_pct": round(float(np.max(pcts)), 2),
    }


gates = {}
vh_paths = [
    os.path.join(DATA, "verified", "nisar_001_waste.txt"),
    os.path.join(DATA, "verified", "nisar_002_story.txt"),
    os.path.join(DATA, "verified", "nisar_003_school.txt"),
    os.path.join(DATA, "verified", "human_004_listeria.txt"),
    os.path.join(DATA, "verified", "human_005_science.txt"),
]
vh26 = fpr_gate_txts(vh_paths, proba26)
vh17 = fpr_gate_txts(vh_paths, proba17)
vh_pass = vh26["breach_frac_ge20"] < 0.05
gates["verified_humans"] = {"V26": vh26, "v17": vh17, "gate": "breach < 5%", "PASS": bool(vh_pass)}
print(f"verified_humans: V26 breach={vh26['breach_frac_ge20']} v17 breach={vh17['breach_frac_ge20']} "
      f"{'PASS' if vh_pass else 'FAIL'}", flush=True)

dh26 = fpr_gate_jsonl(os.path.join(DATA, "drive_human_samples.jsonl"), proba26)
dh17 = fpr_gate_jsonl(os.path.join(DATA, "drive_human_samples.jsonl"), proba17)
dh_pass = dh26["breach_frac_ge20"] < 0.05
gates["drive_human"] = {"V26": dh26, "v17": dh17, "gate": "breach < 5%", "PASS": bool(dh_pass)}
print(f"drive_human: V26 breach={dh26['breach_frac_ge20']} max={dh26['max_ai_pct']}% | "
      f"v17 breach={dh17['breach_frac_ge20']} {'PASS' if dh_pass else 'FAIL'}", flush=True)

wk_path = os.path.join(DATA, "wiki_human_eval.jsonl")
if os.path.exists(wk_path):
    wk26 = fpr_gate_jsonl(wk_path, proba26)
    wk17 = fpr_gate_jsonl(wk_path, proba17)
    wk_pass = wk26["mean_ai_pct"] < 10.0
else:
    wk26, wk17, wk_pass = {"missing": True}, {"missing": True}, False
gates["wikipedia"] = {"V26": wk26, "v17": wk17, "gate": "mean < 10%", "PASS": bool(wk_pass)}
print(f"wikipedia: V26 mean={wk26.get('mean_ai_pct')}% v17 mean={wk17.get('mean_ai_pct')}% "
      f"{'PASS' if wk_pass else 'FAIL'}", flush=True)

report["human_fpr_gates"] = gates

# ---- 31-doc human holdout: score but DO NOT TUNE (specificity only) ----
h_docs = []
try:
    holdout = json.load(open(os.path.join(DATA, "holdout_manifest_A.json")))
    def _find_docs(obj, depth=0):
        if depth > 4: return []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("docs", "documents", "items") and isinstance(v, list) and v:
                    # verify list items look like docs
                    if isinstance(v[0], dict) and "text" in v[0]:
                        return v
                r = _find_docs(v, depth + 1)
                if r: return r
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict) and "text" in obj[0]:
            return obj
        return []
    h_docs = _find_docs(holdout) or []
except Exception as e:
    print(f"holdout load skipped: {e}", flush=True)
    h_docs = []
# Holdout docs reference files; load text from file paths
h_texts = []
for d in h_docs:
    if not isinstance(d, dict): continue
    fp = d.get("file")
    if fp:
        full = os.path.join(DATA, fp)
        if os.path.exists(full):
            with open(full, encoding="utf-8") as f:
                body = "\n".join(l for l in f.read().splitlines() if not l.startswith("#")).strip()
                if body: h_texts.append(body)
if h_texts and len(h_texts) > 0:
    h_pcts26, h_pcts17 = [], []
    for t in h_texts:
        h_pcts26.append(proba26(t) * 100.0)
        h_pcts17.append(proba17(t) * 100.0)
    h_pcts26 = np.array(h_pcts26); h_pcts17 = np.array(h_pcts17)
    report["holdout_31_human_specificity"] = {
        "n": len(h_pcts26),
        "V26": {"mean": round(float(np.mean(h_pcts26)), 2), "max": round(float(np.max(h_pcts26)), 2),
                "breach_ge20": round(float(np.mean(h_pcts26 >= 20)), 4)},
        "v17": {"mean": round(float(np.mean(h_pcts17)), 2), "max": round(float(np.max(h_pcts17)), 2),
                "breach_ge20": round(float(np.mean(h_pcts17 >= 20)), 4)},
        "note": "Human-only holdout: measures specificity/FPR only. NOT tuned against. "
                "Does NOT validate ±5 parity.",
    }
    print(f"holdout31: n={len(h_pcts26)} V26 mean={np.mean(h_pcts26):.1f} breach={np.mean(h_pcts26>=20):.3f} | "
          f"v17 mean={np.mean(h_pcts17):.1f} breach={np.mean(h_pcts17>=20):.3f}", flush=True)

# ---- Blueprint metrics: MAE, within-5, within-10, bias (document-level) ----
def bp_metrics(pairs):
    errs = [abs(p - t) for p, t, _ in pairs]
    signed = [p - t for p, t, _ in pairs]
    n = len(pairs)
    return {
        "n_paired": n,
        "mae": round(float(np.mean(errs)), 2) if n else None,
        "within5_rate": round(float(np.mean([e <= 5.0 for e in errs])), 4) if n else None,
        "within5_count": f"{sum(1 for e in errs if e <= 5.0)}/{n}" if n else None,
        "within10_rate": round(float(np.mean([e <= 10.0 for e in errs])), 4) if n else None,
        "within10_count": f"{sum(1 for e in errs if e <= 10.0)}/{n}" if n else None,
        "bias_signed_mean": round(float(np.mean(signed)), 2) if n else None,
        "median_abs_err": round(float(np.median(errs)), 2) if n else None,
        "p90_abs_err": round(float(np.percentile(errs, 90)), 2) if n else None,
        "per_doc": [{"doc": nm, "predicted": round(p, 1), "reference": t,
                     "signed_err": round(p - t, 1), "abs_err": round(abs(p - t), 1),
                     "within5": bool(abs(p - t) <= 5.0), "within10": bool(abs(p - t) <= 10.0)}
                    for p, t, nm in pairs],
    }


bp26_pairs, bp17_pairs = [], []
if fp26 is not None: bp26_pairs.append((fp26, 80.0, "FullPaper"))
if fp17 is not None: bp17_pairs.append((fp17, 80.0, "FullPaper"))
if a26 is not None: bp26_pairs.append((a26, 57.0, "22103"))
if a17 is not None: bp17_pairs.append((a17, 57.0, "22103"))
if phd26 is not None: bp26_pairs.append((phd26, 100.0, "PhD"))
if phd17 is not None: bp17_pairs.append((phd17, 100.0, "PhD"))
for r in scored:
    if r.get("target") != "*" and "V26" in r:
        bp26_pairs.append((r["V26"], float(r["target"]), r["name"]))
        bp17_pairs.append((r["v17"], float(r["target"]), r["name"]))

bp26 = bp_metrics(bp26_pairs)
bp17 = bp_metrics(bp17_pairs)
bp26["note"] = ("Document-level metrics. References are verified Turnitin scores on file. "
                "Legacy docs in-sample/optimistic. Small-n caveat. No ±5 parity claim.")
report["blueprint_metrics_V26"] = bp26
report["blueprint_metrics_v17"] = bp17
print(f"BLUEPRINT V26: n={bp26['n_paired']} MAE={bp26['mae']} within5={bp26['within5_count']} "
      f"within10={bp26['within10_count']} bias={bp26['bias_signed_mean']}", flush=True)
print(f"BLUEPRINT v17: n={bp17['n_paired']} MAE={bp17['mae']} within5={bp17['within5_count']} "
      f"within10={bp17['within10_count']} bias={bp17['bias_signed_mean']}", flush=True)

# ---- overall verdict ----
all_pass = bool(fp_pass and a_pass and phd_pass and anchor_pass and vh_pass and dh_pass and wk_pass)
report["overall_verdict"] = "DEPLOY_CANDIDATE" if all_pass else "DO_NOT_DEPLOY"
report["gates_summary"] = {
    "full_paper_75_85": bool(fp_pass),
    "anchor_22103_52_62": bool(a_pass),
    "anchor_phd_95_100": bool(phd_pass),
    "anchors_mean_lt15_and_4of8": bool(anchor_pass),
    "verified_humans_fpr_lt5": bool(vh_pass),
    "drive_human_fpr_lt5": bool(dh_pass),
    "wikipedia_lt10": bool(wk_pass),
}
report["live_model_md5"] = live_md5
report["live_model_untouched"] = True
report["elapsed_s"] = round(time.time() - t0, 1)
report["final_note"] = ("V26 is a research-only experimental candidate. It was NOT deployed. "
                        "Live v17 unchanged. No ±5 Turnitin parity claimed: that requires an "
                        "independent paired AI test set with verified numeric reference scores.")

out = os.path.join(PROJ, "models", "candidates", "allfields_v26_eval_report.json")
json.dump(report, open(out, "w"), indent=1)
print(f"\nVERDICT: {report['overall_verdict']}", flush=True)
print(f"saved {out}", flush=True)
