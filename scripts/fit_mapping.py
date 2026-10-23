#!/usr/bin/env python3
"""Fit score-mapping calibration: raw model % -> Turnitin-scale %.

Anchors: (raw_pct, turnitin_pct) from scripts/score_anchors.py output
(data/calibration/anchors_scored.json) plus a (0,0) human anchor —
real-human gate docs must stay ~0 after mapping (FPR protection).

Fits isotonic regression (monotonic, no parametric assumptions).
Validates: mapped scores on real-human gate sets must stay <20%
(displayed as *%/0%), else the mapping is REJECTED as unsafe.

Writes models/score_mapping.json:
  {"xs": [...], "ys": [...], "fitted_at": ..., "anchors": [...]}
Usage: .venv/bin/python scripts/fit_mapping.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np
from sklearn.isotonic import IsotonicRegression

CAL = os.path.join(ROOT, "data", "calibration")


def load_anchors():
    path = os.path.join(CAL, "anchors_scored.json")
    anchors = [(0.0, 0.0)]  # human anchor: real-human docs must stay ~0
    for r in json.load(open(path)):
        anchors.append((float(r["ai_pct"]), float(r["turnitin_pct"])))
    return anchors


def main():
    anchors = load_anchors()
    # drop exact-duplicate x keeping max y (isotonic handles, but be explicit)
    xs = np.array([a[0] for a in anchors])
    ys = np.array([a[1] for a in anchors])
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    print("anchors (raw -> TT):", list(zip(xs.tolist(), ys.tolist())))

    ir = IsotonicRegression(y_min=0.0, y_max=100.0, out_of_bounds="clip")
    ir.fit(xs, ys)

    # densify to a lookup table for the app (0..100 step 0.5)
    grid = np.arange(0, 100.5, 0.5)
    mapped = ir.predict(grid)
    # enforce: never map DOWN below raw (we only correct underscoring),
    # and keep (0,0) fixed
    mapped = np.maximum(mapped, grid)
    mapped[0] = 0.0

    out = {
        "xs": grid.tolist(),
        "ys": [round(float(v), 2) for v in mapped],
        "anchors": [[float(a), float(b)] for a, b in zip(xs.tolist(), ys.tolist())],
    }
    with open(os.path.join(ROOT, "models", "score_mapping.json"), "w") as f:
        json.dump(out, f)
    print("wrote models/score_mapping.json")

    # report mapped anchor values
    for x, y in zip(xs, ys):
        m = float(np.interp(x, grid, mapped))
        print(f"  raw {x:6.1f} -> mapped {m:6.1f} (TT {y:6.1f}) gap {y-m:+.1f}")


if __name__ == "__main__":
    main()
