#!/usr/bin/env python3
"""Experiment A — Turnitin-style document aggregation on the LIVE model.
INFERENCE ONLY. No training, no fitting.

App-identical per-sentence scores (8-sent windows, stride 4, live model),
then Turnitin-style aggregation: doc_pct = 100 * mean(sent_p >= tau).
tau tuned on v2 VAL ONLY (grid search, MAE; tie-break = lower FPR on 0%-docs).
Also tests the GPTZero-style dead-zone variant: sent_p in [0.35,0.65] -> 0.
Winner applied to v1 locked test EXACTLY ONCE + v1 star-band pool.

NEVER touches models/nak_detector.joblib.
"""
import hashlib
import json
import os
import sys
import time

PROJ = os.path.expanduser("~/workspace/ai-detector")
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import joblib
import numpy as np

from src.features import extract_features
from app.scan import split_sentences, SentenceScorer

LIVE_MD5 = "9eeba80e03d06c7db8b35796965746c9"
TEST_SHA_PREFIX = "89f8b799"
OUT_JSON = os.path.join(PROJ, "data", "phase5_baseline", "ttmethods_expa_eval.json")

t0 = time.time()


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


live_before = md5(os.path.join(PROJ, "models", "nak_detector.joblib"))
assert live_before == LIVE_MD5, "live model modified!"
test_before = sha256(os.path.join(PROJ, "data", "phase4_splits", "test.json"))
assert test_before.startswith(TEST_SHA_PREFIX), "v1 test manifest changed!"
print(f"pre-run integrity ok [{time.time()-t0:.0f}s]", flush=True)

model = joblib.load(os.path.join(PROJ, "models", "nak_detector.joblib"))
print("live model loaded read-only", flush=True)


def score_text(t):
    f = np.asarray(extract_features(t), dtype=np.float64).reshape(1, -1)
    p = float(model.predict_proba(f)[0][1])
    return min(max(p, 0.0), 1.0)


def sent_scores(text):
    """App-identical per-sentence scores (mean of covering window probs)."""
    scorer = SentenceScorer(score_text)  # window=8, stride=4
    for s in split_sentences(text) or [text]:
        if s.strip():
            scorer.feed(s)
    scorer.finish()
    out = []
    while scorer.finalized:
        _, v = scorer.finalized.popleft()
        out.append(v)
    return out


def agg_mean(sp):
    return round(float(np.mean(sp)) * 100, 1) if sp else 0.0


def agg_tt(sp, tau, deadzone=False):
    if deadzone:
        sp = [0.0 if 0.35 <= p <= 0.65 else p for p in sp]
    return round(100.0 * float(np.mean([1.0 if p >= tau else 0.0 for p in sp])), 1) if sp else 0.0


# ---------- load records ----------
recs_v2 = {}
with open("data/phase2_dataset_v2.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            o = json.loads(line)
            recs_v2[o["doc_id"]] = o
recs_v1 = {}
with open("data/phase2_dataset.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            o = json.loads(line)
            recs_v1[o["doc_id"]] = o
val_ids = json.load(open("data/phase4_splits_v2/val.json"))
test_ids = json.load(open("data/phase4_splits/test.json"))
print(f"val={len(val_ids)} test={len(test_ids)}", flush=True)

# ---------- per-sentence scores for val ----------
val_data = []
for did in val_ids:
    r = recs_v2[did]
    sp = sent_scores(r["text"])
    val_data.append({"doc_id": did, "ref": r["turnitin_ai_score"], "sp": sp,
                     "n_words": len(r["text"].split())})
    print(f"  val {did}: ref={r['turnitin_ai_score']} nsent={len(sp)}", flush=True)

# baseline (mean-of-probs) on val
b_errs = [abs(agg_mean(d["sp"]) - d["ref"]) for d in val_data]
print(f"val baseline mean-agg MAE={np.mean(b_errs):.2f}", flush=True)

# ---------- tune tau on val ----------
TAUS = [round(0.30 + 0.05 * i, 2) for i in range(13)]  # 0.30..0.90
results = {}
for dz in (False, True):
    for tau in TAUS:
        errs, fp = [], 0
        n0 = 0
        for d in val_data:
            pred = agg_tt(d["sp"], tau, deadzone=dz)
            errs.append(abs(pred - d["ref"]))
            if d["ref"] == 0:
                n0 += 1
                if pred >= 20:
                    fp += 1
        results[(dz, tau)] = (float(np.mean(errs)), fp / n0 if n0 else 0.0)

# pick: min MAE, tie-break lower FPR
best = min(results, key=lambda k: (results[k][0], results[k][1]))
best_mae, best_fpr = results[best]
print(f"TUNED: deadzone={best[0]} tau={best[1]} val_MAE={best_mae:.2f} val_FPR={best_fpr:.3f}",
      flush=True)
# show top-5 for the record
top5 = sorted(results, key=lambda k: (results[k][0], results[k][1]))[:5]
for k in top5:
    print(f"    dz={k[0]} tau={k[1]} MAE={results[k][0]:.2f} FPR={results[k][1]:.3f}", flush=True)

dz_best, tau_best = best

# ---------- FINAL: v1 locked test, EXACTLY ONCE ----------
rows = []
for did in test_ids:
    r = recs_v1[did]
    ref = r["turnitin_ai_score"]
    assert isinstance(ref, (int, float))
    sp = sent_scores(r["text"])
    pred = agg_tt(sp, tau_best, deadzone=dz_best)
    ae = abs(pred - ref)
    rows.append({"doc_id": did, "turnitin_ref": ref, "predicted": pred,
                 "abs_err": round(ae, 1), "signed_err": round(pred - ref, 1),
                 "within_5": ae <= 5, "n_sent": len(sp)})
mae = float(np.mean([r["abs_err"] for r in rows]))
med = float(np.median([r["abs_err"] for r in rows]))
signed = float(np.mean([r["signed_err"] for r in rows]))
w5 = sum(1 for r in rows if r["within_5"])
print("---- v1 LOCKED TEST (n=16, Turnitin-style agg, scored once) ----", flush=True)
for r in rows:
    print(f"  {r['doc_id']}: ref={r['turnitin_ref']} pred={r['predicted']} ae={r['abs_err']}",
          flush=True)
print(f"MAE={mae:.2f} medianAE={med:.2f} signed={signed:+.2f} within5={w5}/16", flush=True)

# ---------- star-band compliance (v1 68-pool) ----------
star_pool = json.load(open("data/phase4_splits/star_band_pool.json"))
star_ids = star_pool if isinstance(star_pool, list) else star_pool["doc_ids"]
in_band, n_star = 0, 0
for did in star_ids:
    r = recs_v1[did]
    assert r["turnitin_ai_band"] == "*%"
    if not r["text"] or len(r["text"].split()) < 50:
        continue
    sp = sent_scores(r["text"])
    pred = agg_tt(sp, tau_best, deadzone=dz_best)
    n_star += 1
    if 1.0 <= pred <= 19.0:
        in_band += 1
compliance = in_band / n_star if n_star else 0
print(f"star-band: {in_band}/{n_star} = {compliance*100:.1f}%", flush=True)

# ---------- gates vs baseline ----------
BASE_MAE, BASE_W5, BASE_STAR = 19.54, 3, 0.75
gates = {"MAE < 19.54": mae < BASE_MAE,
         "within-5 >= 4/16": w5 >= 4,
         "star compliance >= 70%": compliance >= 0.70}
verdict = "PASS" if all(gates.values()) else "HOLD"
for g, ok in gates.items():
    print(f"gate [{g}]: {'PASS' if ok else 'FAIL'}", flush=True)
print(f"VERDICT: {verdict}", flush=True)

live_after = md5(os.path.join(PROJ, "models", "nak_detector.joblib"))
test_after = sha256(os.path.join(PROJ, "data", "phase4_splits", "test.json"))
assert live_after == LIVE_MD5 and test_after == test_before, "INTEGRITY BREACH"
print("post-run integrity ok", flush=True)

with open(OUT_JSON, "w") as f:
    json.dump({
        "experiment": "A_turnitin_aggregation_live_model",
        "tuned": {"deadzone": dz_best, "tau": tau_best,
                  "val_MAE": round(best_mae, 2), "val_FPR": round(best_fpr, 4)},
        "val_grid_top5": [{"deadzone": k[0], "tau": k[1],
                           "MAE": round(v[0], 2), "FPR": round(v[1], 4)}
                          for k, v in top5],
        "locked_test": {"mae": round(mae, 2), "median_ae": round(med, 2),
                        "mean_signed": round(signed, 2), "within_5": w5,
                        "n": 16, "rows": rows},
        "star_band": {"in_band": in_band, "n": n_star,
                      "compliance": round(compliance, 4)},
        "gates": gates, "verdict": verdict,
    }, f, indent=1)
print(f"saved {OUT_JSON} [{time.time()-t0:.0f}s]", flush=True)
