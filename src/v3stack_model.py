#!/usr/bin/env python3
"""Scoring interface for v3-stack (nak_detector_v3_stack).

Bundle: head_report, head_prose, pair_w, metas{report,prose:{clf,scaler}},
        router, router_scaler, new_scaler, ...

window: X=[x187, x10]; pr=v2_report(X); pe=v2_prose(X); s=pair_w.X
p = w*meta_report([pr,pe,s]) + (1-w)*meta_prose([pr,pe,s]), w=router(doc)
App-identical aggregation otherwise.
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


class V3StackModel:
    def __init__(self, bundle_path=None):
        bundle_path = bundle_path or os.path.join(
            ROOT, "models", "nak_detector_v3_stack.joblib")
        b = joblib.load(bundle_path)
        self.head_report = b["head_report"]
        self.head_prose = b["head_prose"]
        self.pair_w = np.asarray(b["pair_w"], dtype=np.float64)
        self.metas = b["metas"]
        self.router = b["router"]
        self.router_scaler = b["router_scaler"]
        self.new_scaler = b["new_scaler"]
        self.base_dim = b["base_dim"]

    def genre_weight(self, doc_text):
        gv = np.asarray(genre_feature_vector(doc_text),
                        dtype=np.float64).reshape(1, -1)
        return float(self.router.predict_proba(
            self.router_scaler.transform(gv))[0][1])

    def window_proba(self, window_text, w):
        xb = np.asarray(base_features_187(window_text),
                        dtype=np.float64).reshape(1, -1)
        xn = _new_features_10(window_text).reshape(1, -1)
        xn = self.new_scaler.transform(xn)
        X = np.hstack([xb, xn])
        pr = float(self.head_report.predict_proba(X)[0][1])
        pe = float(self.head_prose.predict_proba(X)[0][1])
        s = float(np.ravel(X @ self.pair_w)[0])
        M = np.array([[pr, pe, s]])
        mr = self.metas["report"]
        me = self.metas["prose"]
        p_r = float(mr["clf"].predict_proba(mr["scaler"].transform(M))[0][1])
        p_e = float(me["clf"].predict_proba(me["scaler"].transform(M))[0][1])
        p = w * p_r + (1.0 - w) * p_e
        return min(max(p, 0.0), 1.0)

    def score_doc(self, text):
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
