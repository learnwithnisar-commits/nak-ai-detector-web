#!/usr/bin/env python3
"""Score anchor documents with the app pipeline (baseline vs new model).

Replicates app/main.py: parse -> sentences -> SentenceScorer(8/4) ->
ai_pct = mean(sentence scores) * 100.
Usage: .venv/bin/python scripts/score_anchors.py [model_joblib]
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

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "models", "nak_detector.joblib")

model = joblib.load(MODEL)

# model file may be raw classifier; features come from src (same as app)
sys.path.insert(0, ROOT)
from src.features import extract_features  # noqa: E402


def score_text(text: str) -> float:
    features = np.asarray(extract_features(text), dtype=np.float64).reshape(1, -1)
    p = float(model.predict_proba(features)[0][1])
    return min(max(p, 0.0), 1.0)


def score_doc(path: str):
    ext = Path(path).suffix.lower()
    scorer = SentenceScorer(score_text)
    sent_sum, sent_count, words = 0.0, 0, 0
    for unit in parse_document(Path(path), ext):
        if unit["tag"] == "p" and unit["text"].strip():
            words += count_words(unit["text"])
            for s in split_sentences(unit["text"]) or [unit["text"].strip()]:
                g = scorer.feed(s)
    scorer.finish()
    # drain finalized
    while scorer.finalized:
        g, s = scorer.finalized.popleft()
        sent_sum += s
        sent_count += 1
    ai_pct = round(sent_sum / sent_count * 100, 1) if sent_count else 0.0
    return {"file": os.path.basename(path), "ai_pct": ai_pct,
            "sentences": sent_count, "words": words}


if __name__ == "__main__":
    anchors = [
        ("/home/hatch/workspace/user/files/Listeria_goat_MANUSCRIPT_IMPROVED_CLEAN.docx", 54),
        ("/home/hatch/workspace/user/files/XDR_Typhoid_Pakistan_PJMR_Format.docx", 64),
        ("/home/hatch/workspace/user/files/aigc_download_2027795315876175872_aigc_rate.docx", 43),
        ("/home/hatch/workspace/ai-detector/data/real_ai/ash_coal_essay_chatgpt.txt", 100),
    ]
    out = []
    for path, tt in anchors:
        if not os.path.exists(path):
            print(f"MISSING {path}")
            continue
        r = score_doc(path)
        r["turnitin_pct"] = tt
        r["gap"] = round(tt - r["ai_pct"], 1)
        out.append(r)
        print(json.dumps(r))
    json.dump(out, open(os.path.join(ROOT, "data", "calibration",
                                     "anchors_scored.json"), "w"), indent=1)
    print("saved data/calibration/anchors_scored.json")
