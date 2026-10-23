#!/usr/bin/env python3
"""Head-to-head: live baseline vs r4 candidate vs v2, app-identical pipeline."""
import json
import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import joblib
import numpy as np

from app.parse import parse_document
from app.scan import SentenceScorer, split_sentences
from app.parse import _flush_para
from src.features import extract_features
from src.r4_features import extract_new_features, feature_names_new
from src.v2_model import V2Model

DATA = os.path.join(ROOT, "data")
ANCH = os.path.join(DATA, "anchors_oct16r2")
MANIFEST = json.load(open(os.path.join(ANCH, "anchors_oct16r2_manifest.json")))

live = joblib.load(os.path.join(ROOT, "models", "nak_detector.joblib"))
r4b = joblib.load(os.path.join(ROOT, "models", "candidates", "naxtrain_oct16r4.joblib"))
r4_clf, r4_sc, r4_names = r4b["clf"], r4b["new_scaler"], r4b["new_names"]
_alln = feature_names_new()
_r4_idx = [_alln.index(n) for n in r4_names]
v2 = V2Model()


def base_p(t):
    f = np.asarray(extract_features(t), dtype=np.float64).reshape(1, -1)
    return float(np.clip(live.predict_proba(f)[0][1], 0, 1))


def r4_p(t):
    xb = np.asarray(extract_features(t), dtype=np.float64).reshape(1, -1)
    _, arr = extract_new_features(t)
    xn = r4_sc.transform(np.asarray([arr[i] for i in _r4_idx], dtype=np.float64).reshape(1, -1))
    return float(np.clip(r4_clf.predict_proba(np.hstack([xb, xn]))[0][1], 0, 1))


def units_of(path):
    ext = Path(path).suffix.lower()
    if ext == ".txt":
        text = open(path, encoding="utf-8").read()
        text = "\n".join(ln for ln in text.split("\n") if not re.match(r"^\s*\d+\s*$", ln))
        units, cur = [], []
        for line in text.split("\n"):
            if line.strip():
                cur.append(line.strip())
            elif cur:
                u = _flush_para(cur); cur = []
                if u: units.append(u)
        if cur:
            u = _flush_para(cur)
            if u: units.append(u)
        return units
    return list(parse_document(Path(path), ext))


def score_units(units, score_fn):
    scorer = SentenceScorer(score_fn)
    ssum, sn = 0.0, 0
    for unit in units:
        if unit["tag"] == "p" and unit["text"].strip():
            for s in split_sentences(unit["text"]) or [unit["text"].strip()]:
                scorer.feed(s)
    scorer.finish()
    while scorer.finalized:
        _, sc = scorer.finalized.popleft()
        ssum += sc; sn += 1
    return round(ssum / sn * 100, 1) if sn else 0.0


def main():
    v2w_cache = {}
    rows = []
    items = [(k, os.path.join(ANCH, k + ".txt"), v["class"], v["tt_pct"])
             for k, v in MANIFEST.items()
             if os.path.exists(os.path.join(ANCH, k + ".txt"))]
    items.append(("ash_coal", os.path.join(DATA, "real_ai", "ash_coal_essay_chatgpt.txt"), "ai", 100))
    for key, path, cls, tt in items:
        units = units_of(path)
        b = score_units(units, base_p)
        r4 = score_units(units, r4_p)
        scored_text = "\n".join(u["text"] for u in units if u["tag"] == "p" and u["text"].strip())
        w = v2.genre_weight(scored_text) if scored_text.strip() else 0.0
        v2v = score_units(units, lambda t: v2.window_proba(t, w))
        rows.append({"key": key, "class": cls, "tt": tt, "base": b, "r4": r4, "v2": v2v, "w": round(w, 2)})
        if len(rows) % 40 == 0:
            print(f"scored {len(rows)}/{len(items)}", flush=True)
    json.dump(rows, open("/tmp/compare_v2.json", "w"), indent=1)

    for name in ("base", "r4", "v2"):
        ai = [r for r in rows if r["class"] == "ai" and isinstance(r["tt"], int)]
        hu = [r for r in rows if r["class"] == "human"]
        st = [r for r in rows if r["class"] == "star"]
        mg = np.mean([abs(r[name] - r["tt"]) for r in ai])
        at5 = sum(1 for r in ai if abs(r[name] - r["tt"]) <= 5)
        hm = np.mean([r[name] for r in hu])
        ib = sum(1 for r in st if 1 <= r[name] <= 19) / max(1, len(st))
        ash = next(r for r in rows if r["key"] == "ash_coal")[name]
        print(f"{name:5s} A_mean_gap={mg:5.2f} at5={at5:2d}/{len(ai)} "
              f"B_human_mean={hm:5.2f} D_inband={ib:.3f} C_ash={ash:5.1f}")


if __name__ == "__main__":
    main()
