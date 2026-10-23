"""Turnitin-pattern features for NAK v9.

Derived from the deep linguistic analysis of 133 Turnitin AI reports
(data/turnitin_highlight_analysis.md, data/turnitin_pattern_features.json).

8 features:
  POSITIVE (high values -> AI):
    tt_list_enum_density         enumerated-list markers First/Second/Third... per 1k words (STRONG)
    tt_transition_density        transition phrases per 1k words
    tt_conclusion_marker         binary: "in conclusion"/"in summary" present
    tt_balanced_contrast_density balanced frames (not only...but also/while/whereas) per 1k words
  NEGATIVE (high values -> HUMAN; counterweights):
    tt_hedge_density             hedging words per 1k words
    tt_passive_density           passive constructions (be + V-ed) per 1k words
  NEUTRAL (controls; no AI signal in this data — do not interpret as AI tells):
    tt_sent_len_cv               coefficient of variation of sentence length
    tt_ttr200                    type-token ratio over a fixed 200-word window

tt_features(text) -> np.ndarray shape (8,), float64. Deterministic, pure Python.
"""

from __future__ import annotations

import re

import numpy as np

TT_FEATURE_NAMES = [
    "tt_list_enum_density",
    "tt_transition_density",
    "tt_conclusion_marker",
    "tt_balanced_contrast_density",
    "tt_hedge_density",
    "tt_passive_density",
    "tt_sent_len_cv",
    "tt_ttr200",
]

_ENUM_RE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|firstly|secondly|thirdly)\b",
    re.IGNORECASE,
)
_TRANS_RE = re.compile(
    r"\b(?:furthermore|moreover|additionally|consequently|however|nevertheless|"
    r"nonetheless|in contrast|as a result|for example|for instance|in other words|"
    r"ultimately|notably|importantly)\b",
    re.IGNORECASE,
)
_CONCL_RE = re.compile(r"\bin (conclusion|summary)\b", re.IGNORECASE)
_BAL_RE = re.compile(
    r"\b(?:not only|on the one hand|on one hand|while|whereas|on the other hand)\b",
    re.IGNORECASE,
)
_HEDGE_RE = re.compile(
    r"\b(?:may|might|could|would|often|generally|typically|usually|likely|possibly|"
    r"perhaps|seems|appears|relatively|somewhat|largely)\b",
    re.IGNORECASE,
)
_PASS_RE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\s+[a-z]+ed\b",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?\u2026])\s+|\n+")


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def tt_features(text: str) -> np.ndarray:
    """Return the 8 Turnitin-pattern features for *text* (unscaled)."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    words = _words(text)
    n_w = len(words)
    sents = _sentences(text)
    per1k = 1000.0 / n_w if n_w else 0.0

    feats: list[float] = []

    # 1. enumerated-list markers per 1k words (STRONGEST AI signal)
    feats.append(len(_ENUM_RE.findall(text)) * per1k)
    # 2. transition phrases per 1k words
    feats.append(len(_TRANS_RE.findall(text)) * per1k)
    # 3. formulaic closer: binary presence
    feats.append(1.0 if _CONCL_RE.search(text) else 0.0)
    # 4. balanced-contrast frames per 1k words
    feats.append(len(_BAL_RE.findall(text)) * per1k)
    # 5. hedging words per 1k words (NEGATIVE: humans hedge more)
    feats.append(len(_HEDGE_RE.findall(text)) * per1k)
    # 6. passive constructions per 1k words (NEGATIVE: humans more passive)
    feats.append(len(_PASS_RE.findall(text)) * per1k)

    # 7. sentence-length CV (NEUTRAL control)
    sent_lens = [len(_WORD_RE.findall(s)) for s in sents]
    if len(sent_lens) >= 2:
        mu = sum(sent_lens) / len(sent_lens)
        sd = (sum((x - mu) ** 2 for x in sent_lens) / (len(sent_lens) - 1)) ** 0.5
        feats.append(sd / mu if mu else 0.0)
    else:
        feats.append(0.0)

    # 8. TTR over fixed 200-word window (NEUTRAL control)
    win = words[:200]
    feats.append(len(set(win)) / len(win) if win else 0.0)

    vec = np.asarray(feats, dtype=np.float64)
    return np.where(np.isfinite(vec), vec, 0.0)


def tt_feature_names() -> list[str]:
    return list(TT_FEATURE_NAMES)
