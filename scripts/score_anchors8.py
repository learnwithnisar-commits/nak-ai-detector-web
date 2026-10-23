#!/usr/bin/env python3
"""Score the 8 real anchors with the app pipeline (matches app/main.py exactly).

Targets: numeric Turnitin % or "*" (Turnitin 1-19% band; we display "*" for 0<x<20).
Gap metric: numeric -> |ours-target|; "*" -> 0 if 0<ours<20 else distance to band.

Usage: .venv/bin/python scripts/score_anchors8.py [model_joblib] [out_json]
"""
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import joblib
import numpy as np

from app.parse import parse_document
from app.scan import SentenceScorer, count_words, split_sentences
from src.features import extract_features

FILES = "/home/hatch/workspace/user/files"
AD = "/home/hatch/workspace/ai-detector"

# (name, path, target) ; target = number or "*"
ANCHORS = [
    ("Listeria", f"{FILES}/Listeria_goat_MANUSCRIPT_IMPROVED_CLEAN.docx", 54),
    ("XDR_Typhoid", f"{FILES}/XDR_Typhoid_Pakistan_PJMR_Format.docx", 64),
    ("AIGC", f"{FILES}/aigc_download_2027795315876175872_aigc_rate.docx", 43),
    ("ash_coal", f"{AD}/data/real_ai/ash_coal_essay_chatgpt.txt", 100),
    ("Competition", f"{FILES}/upload.docx", "*"),
    ("Junlan_Li", f"{FILES}/Essay-_Junlan_Li-25041323.docx", "*"),
    ("Liu_Yueru", f"{FILES}/Liu_Yueru_Vicky__FA1_Yueru__Vicky__LIU.docx", 60),
    ("ES1002", f"{FILES}/ES1002_Essay.docx", 0),
]


def gap(ours, target):
    if target == "*":
        if 0.0 < ours < 20.0:
            return 0.0
        return round(min(abs(ours - 1.0), abs(ours - 19.0)), 1)
    return round(abs(ours - target), 1)


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "models", "nak_detector.joblib")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        ROOT, "data", "calibration", "anchors8_scored.json")
    model = joblib.load(model_path)

    def score_text(text):
        f = np.asarray(extract_features(text), dtype=np.float64).reshape(1, -1)
        p = float(model.predict_proba(f)[0][1])
        return min(max(p, 0.0), 1.0)

    rows = []
    for name, path, target in ANCHORS:
        if not os.path.exists(path):
            print(f"MISSING {name}: {path}")
            rows.append({"name": name, "missing": True, "target": target,
                         "gap": None})
            continue
        ext = Path(path).suffix.lower()
        scorer = SentenceScorer(score_text)
        sent_sum, sent_count, words = 0.0, 0, 0
        for unit in parse_document(Path(path), ext):
            if unit["tag"] == "p" and unit["text"].strip():
                words += count_words(unit["text"])
                for s in split_sentences(unit["text"]) or [unit["text"].strip()]:
                    scorer.feed(s)
        scorer.finish()
        while scorer.finalized:
            _, s = scorer.finalized.popleft()
            sent_sum += s
            sent_count += 1
        ai_pct = round(sent_sum / sent_count * 100, 1) if sent_count else 0.0
        g = gap(ai_pct, target)
        rows.append({"name": name, "ai_pct": ai_pct, "target": target,
                     "gap": g, "sentences": sent_count, "words": words})
        print(f"{name:12s} ours={ai_pct:6.1f} target={str(target):>4} "
              f"gap={g:5.1f} (n_sent={sent_count})")

    scored = [r for r in rows if r["gap"] is not None]
    mean_gap = round(sum(r["gap"] for r in scored) / len(scored), 2)
    within5 = sum(1 for r in scored if r["gap"] <= 5.0)
    print(f"\nMEAN GAP = {mean_gap} | within ±5: {within5}/{len(scored)}")
    json.dump({"model": model_path, "mean_gap": mean_gap,
               "within5": within5, "n": len(scored), "anchors": rows},
              open(out_path, "w"), indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main()
