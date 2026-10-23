#!/usr/bin/env python3
"""Scoring interface for the third NAK model (nak_detector_v3_pairwise).

Bundle layout (dict):
  pair_w, cal_report (a,b), cal_prose (a,b), router, router_scaler,
  new_scaler, new_names, genre_names, base_dim

Doc scoring mirrors app/main.py exactly:
  - 8-sentence windows, stride 4 (app.scan.SentenceScorer)
  - raw ranking score s = pair_w . [x187, x10_scaled]
  - per-window p = w*sigmoid(a_r*s+b_r) + (1-w)*sigmoid(a_p*s+b_p),
    w = router(doc)
  - sentence score = mean of covering windows
  - doc AI% = mean(sentence scores) * 100
"""
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.v2_genre import genre_feature_vector  # noqa: E402
from src.r4_features import extract_new_features, feature_names_new  # noqa: E402
from src.features import extract_features as base_features_187  # noqa: E402

_ALL_NEW = feature_names_new()
_KEPT10 = ['r4_proper_cap_1k', 'r4_digit_1k', 'r4_date_1k', 'r4_nominal_1k',
           'r4_past_ed_1k', 'r4_sent_start_bland_ratio', 'r4_sent_len_gini',
           'r4_sent_len_range_cv', 'r4_clause_density', 'r4_repeat_trigram_1k']
_KEPT_IDX = [_ALL_NEW.index(n) for n in _KEPT10]


def _new_features_10(text):
    _, arr = extract_new_features(text)
    return np.asarray([arr[i] for i in _KEPT_IDX], dtype=np.float64)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class V3Model:
    def __init__(self, bundle_path=None):
        bundle_path = bundle_path or os.path.join(
            ROOT, "models", "nak_detector_v3_pairwise.joblib")
        b = joblib.load(bundle_path)
        self.pair_w = np.asarray(b["pair_w"], dtype=np.float64)
        self.cal_report = self._parse_cal(b["cal_report"])
        self.cal_prose = self._parse_cal(b["cal_prose"])
        self.router = b["router"]
        self.router_scaler = b["router_scaler"]
        self.new_scaler = b["new_scaler"]
        self.base_dim = b["base_dim"]
        self.meta = {k: v for k, v in b.items()
                     if k not in ("pair_w", "router", "router_scaler", "new_scaler")}

    @staticmethod
    def _parse_cal(c):
        kind, payload = c[0], c[1]
        if kind == "iso":
            return ("iso", payload)
        (a, b) = payload
        return ("platt", (float(a), float(b)))

    @staticmethod
    def _apply_cal(cal, s):
        kind, payload = cal
        if kind == "iso":
            return float(np.clip(payload.predict([s])[0], 0.0, 1.0))
        a, b = payload
        return float(_sigmoid(a * s + b))

    def genre_weight(self, doc_text):
        """P(report) in [0,1] for a full document."""
        gv = np.asarray(genre_feature_vector(doc_text),
                        dtype=np.float64).reshape(1, -1)
        return float(self.router.predict_proba(
            self.router_scaler.transform(gv))[0][1])

    def window_proba(self, window_text, w):
        """Blended calibrated P(AI) for one 8-sentence window."""
        xb = np.asarray(base_features_187(window_text),
                        dtype=np.float64).reshape(1, -1)
        xn = _new_features_10(window_text).reshape(1, -1)
        xn = self.new_scaler.transform(xn)
        X = np.hstack([xb, xn])
        s = float(np.ravel(X @ self.pair_w)[0])
        pr = self._apply_cal(self.cal_report, s)
        pe = self._apply_cal(self.cal_prose, s)
        p = w * pr + (1.0 - w) * pe
        return min(max(p, 0.0), 1.0)

    def score_doc(self, text):
        """App-identical doc score: mean sentence score * 100."""
        import re as _re
        from app.scan import SentenceScorer, split_sentences
        paras, cur = [], []
        for line in text.split("\n"):
            if line.strip():
                cur.append(line.strip())
            elif cur:
                paras.append(" ".join(cur))
                cur = []
        if cur:
            paras.append(" ".join(cur))
        scored = [p for p in paras if not _re.match(r"^#{1,3}\s+", p)]
        scored_text = "\n".join(scored)
        w = self.genre_weight(scored_text) if scored_text.strip() else 0.0
        scorer = SentenceScorer(lambda t: self.window_proba(t, w))
        sent_sum, sent_count = 0.0, 0
        for para in scored:
            for s in split_sentences(para) or [para.strip()]:
                scorer.feed(s)
        scorer.finish()
        while scorer.finalized:
            _, sc = scorer.finalized.popleft()
            sent_sum += sc
            sent_count += 1
        return {
            "ai_pct": round(sent_sum / sent_count * 100, 1) if sent_count else 0.0,
            "w_report": round(w, 3),
            "sentences": sent_count,
        }
