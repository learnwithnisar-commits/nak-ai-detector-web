#!/usr/bin/env python3
"""Segment-scale robustness: the app scores OVERLAPPING 8-sentence segments
(stride 4) and pools per-sentence means. Validate the model at that scale.

- segment-level accuracy / F1 / FPR (segment inherits its document's label)
- consistency: |doc proba - mean segment proba|, per-doc std of segment probas
- per-genre segment accuracy (academic_paragraph emphasis)
"""
import json
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.features import _sentences, extract_features, set_lm  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
WIN, STRIDE = 8, 4


def segments(text, window=WIN, stride=STRIDE):
    sents = _sentences(text)
    if len(sents) <= window:
        return [" ".join(sents)]
    return [" ".join(sents[i:i + window])
            for i in range(0, len(sents) - window + 1, stride)]


def main():
    bundle = joblib.load(os.path.join(MODELS, "feature_artifacts.joblib"))
    if bundle.get("lm") is not None:
        set_lm(bundle["lm"])
    model = joblib.load(os.path.join(MODELS, "nak_detector.joblib"))
    print(f"model features in: {model.n_features_in_}", flush=True)

    recs = [json.loads(l) for l in open(os.path.join(DATA, "test.jsonl"))
            if l.strip()]

    seg_probs, seg_true, seg_genre = [], [], []
    doc_probs, doc_true = [], []
    per_doc_std, doc_vs_segmean = [], []
    n_seg_total = 0
    for r in recs:
        segs = segments(r["text"])
        n_seg_total += len(segs)
        X = np.stack([extract_features(s) for s in segs])
        p = model.predict_proba(X)[:, 1]
        dp = float(model.predict_proba([extract_features(r["text"])])[0, 1])
        seg_probs.extend(p)
        seg_true.extend([r["label"]] * len(segs))
        seg_genre.extend([r["genre"]] * len(segs))
        doc_probs.append(dp)
        doc_true.append(r["label"])
        per_doc_std.append(float(np.std(p)))
        doc_vs_segmean.append(abs(dp - float(np.mean(p))))
    seg_probs = np.array(seg_probs)
    seg_true = np.array(seg_true)
    seg_pred = (seg_probs >= 0.5).astype(int)

    print(f"\ntest docs: {len(recs)}, total segments: {n_seg_total} "
          f"(avg {n_seg_total/len(recs):.1f}/doc)")
    print("== segment-level (8-sent, stride 4), label = doc label ==")
    print(f"  accuracy: {accuracy_score(seg_true, seg_pred):.4f}")
    print(f"  F1:       {f1_score(seg_true, seg_pred):.4f}")
    tn = ((seg_true == 0) & (seg_pred == 0)).sum()
    fp = ((seg_true == 0) & (seg_pred == 1)).sum()
    fn = ((seg_true == 1) & (seg_pred == 0)).sum()
    tp = ((seg_true == 1) & (seg_pred == 1)).sum()
    print(f"  FPR: {fp/(fp+tn):.4f}  FNR: {fn/(fn+tp):.4f}  "
          f"[tn={tn} fp={fp} fn={fn} tp={tp}]")

    print("== consistency: doc score vs pooled segment scores ==")
    print(f"  mean|doc_proba - mean_seg_proba|: {np.mean(doc_vs_segmean):.4f}")
    print(f"  mean per-doc std of segment probas: {np.mean(per_doc_std):.4f}")
    print(f"  docs with per-doc std > 0.3: "
          f"{sum(s > 0.3 for s in per_doc_std)}/{len(recs)}")

    print("== per-genre SEGMENT accuracy (academic emphasis) ==")
    seg_genre = np.array(seg_genre)
    for g in sorted(set(seg_genre)):
        i = seg_genre == g
        a = accuracy_score(seg_true[i], seg_pred[i])
        f1 = f1_score(seg_true[i], seg_pred[i])
        flag = "   <-- ACADEMIC" if g == "academic_paragraph" else ""
        print(f"  {g:20s} nseg={i.sum():4d} acc={a:.4f} F1={f1:.4f}{flag}")

    # where do segment errors concentrate?
    err_i = seg_pred != seg_true
    if err_i.sum():
        from collections import Counter
        print("  segment errors by genre:",
              dict(Counter(seg_genre[err_i]).most_common()))
        print("  segment errors by true label:",
              dict(Counter(seg_true[err_i].tolist())))

    json.dump({
        "seg_acc": float(accuracy_score(seg_true, seg_pred)),
        "seg_f1": float(f1_score(seg_true, seg_pred)),
        "seg_fpr": float(fp / (fp + tn)),
        "mean_abs_doc_vs_segmean": float(np.mean(doc_vs_segmean)),
        "mean_per_doc_std": float(np.mean(per_doc_std)),
        "per_genre": {g: {"n": int((seg_genre == g).sum()),
                          "acc": float(accuracy_score(
                              seg_true[seg_genre == g],
                              seg_pred[seg_genre == g]))}
                      for g in sorted(set(seg_genre))},
    }, open(os.path.join(MODELS, "segment_metrics.json"), "w"), indent=2)
    print("\nsaved models/segment_metrics.json")


if __name__ == "__main__":
    main()
