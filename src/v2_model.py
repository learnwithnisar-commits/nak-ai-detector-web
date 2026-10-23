#!/usr/bin/env python3
"""Scoring interface for the second NAK model (nak_detector_v2_turnitin).

Bundle layout (dict):
  router, router_scaler, head_report, head_prose, new_scaler, new_names,
  genre_names, base_dim

Doc scoring mirrors app/main.py exactly:
  - 8-sentence windows, stride 4 (app.scan.SentenceScorer)
  - per-window p = w*p_report + (1-w)*p_prose, w = router(doc)  [soft blend]
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


class V2Model:
    def __init__(self, bundle_path=None):
        bundle_path = bundle_path or os.path.join(
            ROOT, "models", "nak_detector_v2_turnitin.joblib")
        b = joblib.load(bundle_path)
        self.router = b["router"]
        self.router_scaler = b["router_scaler"]
        self.head_report = b["head_report"]
        self.head_prose = b["head_prose"]
        self.new_scaler = b["new_scaler"]
        self.base_dim = b["base_dim"]
        self.meta = {k: v for k, v in b.items()
                     if k not in ("router", "router_scaler", "head_report",
                                  "head_prose", "new_scaler")}

    def genre_weight(self, doc_text):
        """P(report) in [0,1] for a full document."""
        gv = np.asarray(genre_feature_vector(doc_text),
                        dtype=np.float64).reshape(1, -1)
        return float(self.router.predict_proba(
            self.router_scaler.transform(gv))[0][1])

    def window_proba(self, window_text, w):
        """Blended P(AI) for one 8-sentence window given doc genre weight w."""
        xb = np.asarray(base_features_187(window_text),
                        dtype=np.float64).reshape(1, -1)
        xn = _new_features_10(window_text).reshape(1, -1)
        xn = self.new_scaler.transform(xn)
        X = np.hstack([xb, xn])
        pr = float(self.head_report.predict_proba(X)[0][1])
        pe = float(self.head_prose.predict_proba(X)[0][1])
        p = w * pr + (1.0 - w) * pe
        return min(max(p, 0.0), 1.0)

    def score_doc(self, text, scored_only=True):
        """App-identical doc score: mean sentence score * 100.

        text: full document text. Markdown '# ' heading lines are excluded
        from scoring exactly like app/parse._flush_para (they become h1..h3
        units which the app does not score). Genre is computed on the same
        scored text the heads see.
        """
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
        scored = [ _re.sub(r"^#{1,3}\s+", "", p) if _re.match(r"^#{1,3}\s+", p) else p
                   for p in paras ]
        # drop heading-only paras (app tags them h1..h3 and skips scoring)
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
