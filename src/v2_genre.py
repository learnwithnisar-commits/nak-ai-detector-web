#!/usr/bin/env python3
"""Genre features for the v2 genre-aware model: REPORT (structured technical/
student report) vs PROSE (essay/reflection/letter/continuous prose).

Durable module for the second NAK model (nak_detector_v2_turnitin).
"""
import re

SECTION_WORDS = (
    "abstract introduction background literature review methodology methods materials "
    "procedure experimental results findings discussion analysis conclusion conclusions "
    "recommendations references bibliography appendix appendices acknowledgements"
).split()
_SECTION_RE = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s+)?(%s)\s*$" % "|".join(SECTION_WORDS))
_HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[A-Z]\.?|\(?[ivx]+\)?)\s+\S+")
_FIG_RE = re.compile(r"(?i)\bfigure\s+\d+")
_TAB_RE = re.compile(r"(?i)\btable\s+\d+")
_LIST_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)]|[a-z][.)])\s+\S+")


def _lines(text):
    return [ln for ln in text.split("\n")]


def genre_features_raw(text):
    """Unscaled genre feature dict for *text* (full document)."""
    lines = _lines(text)
    content = [ln for ln in lines if ln.strip()]
    n_lines = max(1, len(content))
    words = text.split()
    n_words = max(1, len(words))
    chars = max(1, len(text))
    per1k_w = 1000.0 / n_words

    # heading-like lines: short, numbered or title/all caps
    head = 0
    for ln in content:
        s = ln.strip()
        if len(s) > 90 or len(s.split()) > 10:
            continue
        if _HEADING_RE.match(s):
            head += 1
        elif s.isupper() and len(s) > 3:
            head += 1
        elif s.istitle() and len(s.split()) <= 6:
            head += 1
    list_lines = sum(1 for ln in content if _LIST_RE.match(ln))
    feats = {
        "g_heading_density": head / n_lines,
        "g_list_density": list_lines / n_lines,
        "g_digit_1k": sum(c.isdigit() for c in text) / chars * 1000.0,
        "g_section_1k": len(_SECTION_RE.findall(text)) * per1k_w,
        "g_fig_1k": len(_FIG_RE.findall(text)) * per1k_w,
        "g_tab_1k": len(_TAB_RE.findall(text)) * per1k_w,
        "g_pipe_1k": text.count("|") * per1k_w,
        "g_caps_1k": sum(1 for w in words if len(w) > 3 and w.isupper()) * per1k_w,
    }
    return feats


GENRE_FEATURE_NAMES = [
    "g_heading_density", "g_list_density", "g_digit_1k", "g_section_1k",
    "g_fig_1k", "g_tab_1k", "g_pipe_1k", "g_caps_1k",
]


def genre_feature_vector(text):
    f = genre_features_raw(text)
    return [f[n] for n in GENRE_FEATURE_NAMES]
