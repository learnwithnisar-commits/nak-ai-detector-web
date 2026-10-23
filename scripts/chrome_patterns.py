#!/usr/bin/env python3
"""Shared Turnitin report-chrome patterns for V23 Phase 1.

Single source of truth for chrome detection. Imported by:
- scripts/clean_turnitin_chrome.py (dry-run audit)
- tests/test_chrome_cleaner.py (regression tests)

V23 FIX (2026-10-17): the ai_disclaimer pattern was overbroad.
OLD: r"...we encourage you to learn more about turnitin[’']s ai detection.*?(?=\\n\\n|\\Z)"
     with DOTALL swallowed the ENTIRE record text when the extracted text
     contained no paragraph breaks (13 records, up to 14,668 chars).
NEW: bounded to the canonical two-sentence disclaimer only. All 85 observed
     matches in the corpus contain BOTH sentences, so requiring both loses
     nothing and can never swallow following prose.

Unicode-safe: re.UNICODE is default in Python 3.
"""
import re

# ---------------------------------------------------------------------------
# Canonical AI-report disclaimer (verbatim from corpus, 170 chars):
# "It is essential to understand the limitations of AI detection before
#  making decisions about a student’s work. We encourage you to learn more
#  about Turnitin’s AI detection"
# ---------------------------------------------------------------------------
AI_DISCLAIMER_RE = re.compile(
    r"it is essential to understand the limitations of ai detection"
    r"[^.]{0,200}\."          # first sentence body, bounded, ends at period
    r"\s*we encourage you to learn more about turnitin[’']s ai detection\b",
    re.IGNORECASE,
)
# NOTE: no trailing .* — the match ends exactly at "ai detection".

# The OLD overbroad pattern, kept ONLY for regression testing (must NOT be
# used for cleaning). Tests assert the new pattern never reproduces its
# overmatch behaviour on the 13 known cases.
AI_DISCLAIMER_RE_OLD_OVERBROAD = re.compile(
    r"(?:it is essential to understand the limitations of ai detection.*?)?"
    r"we encourage you to learn more about turnitin[’']s ai detection"
    r".*?(?=\n\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# ---------------------------------------------------------------------------
# Tier-1 patterns: confirmed Turnitin report chrome only.
# (name, regex, rationale)
# ---------------------------------------------------------------------------
TIER1_PATTERNS = [
    (
        "submission_id",
        re.compile(r"trn:oid:::[0-9:]+", re.IGNORECASE),
        "Turnitin submission identifier; never part of student prose.",
    ),
    (
        "page_marker",
        re.compile(r"page \d+ of \d+(?:\s*-\s*[a-z ]*)?", re.IGNORECASE),
        "PDF page header/footer inserted by the report renderer.",
    ),
    (
        "document_details_header",
        re.compile(
            r"document details\s+submission id\b.*?(?=page \d+ of \d+|\d+% overall similarity|$)",
            re.IGNORECASE | re.DOTALL,
        ),
        "Report cover-page metadata block (submission/download dates, file name/size).",
    ),
    (
        "integrity_flags_block",
        re.compile(
            r"integrity flags\b.*?(?=match groups|$)",
            re.IGNORECASE | re.DOTALL,
        ),
        "Report integrity-flag boilerplate.",
    ),
    (
        "ai_disclaimer",
        AI_DISCLAIMER_RE,
        "AI-report disclaimer footer (bounded two-sentence match); report chrome, not student prose.",
    ),
]

# ---------------------------------------------------------------------------
# Tier-2 / uncertain patterns (NOT auto-removed; human review required).
# ---------------------------------------------------------------------------
TIER2_PATTERNS = [
    (
        "overall_similarity_block",
        re.compile(
            r"\d+%\s+overall similarity\b.*?(?=top sources|integrity flags|match groups|$)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "top_sources_block",
        re.compile(
            r"top sources\b.*?(?=integrity flags|match groups|$)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "match_groups_block",
        re.compile(r"match groups\b.*", re.IGNORECASE | re.DOTALL),
    ),
    (
        "file_metadata",
        re.compile(
            r"file name\s+\S+.*?(?:file size|pages|words|characters)\b.*",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
]

# Sentences containing these markers but NOT fully matched above -> uncertain
UNCERTAIN_HINTS = re.compile(
    r"(submission|download date|overall similarity|integrity|match group|"
    r"top source|cited and quoted|missing citation|turnitin[’']s ai)",
    re.IGNORECASE,
)

# Guard: matches longer than this are auto-classified uncertain (may include prose)
LONG_MATCH_GUARD = 1500


def split_sentences_simple(text):
    """Lightweight sentence splitter for audit purposes (not the app's)."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def find_tier1_spans(text):
    """Return [(start, end, pattern_name)] for Tier-1 matches (raw, may overlap)."""
    spans = []
    for name, rx, _rationale in TIER1_PATTERNS:
        for m in rx.finditer(text):
            spans.append((m.start(), m.end(), name))
    return spans


def classify_spans(text, spans):
    """Split spans into verified vs uncertain using the long-match guard.

    Returns (verified_spans, uncertain_spans) where each is a list of
    (start, end, pattern_name). Spans > LONG_MATCH_GUARD chars -> uncertain.
    """
    verified, uncertain = [], []
    for s, e, name in spans:
        if (e - s) > LONG_MATCH_GUARD:
            uncertain.append((s, e, name))
        else:
            verified.append((s, e, name))
    return verified, uncertain


def union_intervals(spans):
    """Merge overlapping (start, end[, ...]) intervals; return merged list."""
    ivs = sorted((s, e) for s, e, *_ in spans)
    if not ivs:
        return []
    merged = []
    cs, ce = ivs[0]
    for s, e in ivs[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    return merged


def resolve_verified_uncertain_overlap(verified, uncertain):
    """Conservative overlap resolution: any verified span touching an
    uncertain span is demoted to uncertain. Returns (verified_clean,
    uncertain_all) that are disjoint by construction."""
    u_merged = union_intervals(uncertain)
    v_clean, demoted = [], []
    for s, e, name in verified:
        if any(s < ue and us < e for us, ue in u_merged):
            demoted.append((s, e, name))
        else:
            v_clean.append((s, e, name))
    uncertain_all = uncertain + demoted
    # sanity: disjoint
    v_merged = union_intervals(v_clean)
    u_merged2 = union_intervals(uncertain_all)
    for vs, ve in v_merged:
        for us, ue in u_merged2:
            assert not (vs < ue and us < ve), "overlap after resolution!"
    return v_clean, uncertain_all


def apply_removal(text, spans):
    """Return text with the given (start, end[, ...]) spans removed."""
    merged = union_intervals(spans)
    out, pos = [], 0
    for s, e in merged:
        out.append(text[pos:s])
        pos = e
    out.append(text[pos:])
    return "".join(out)


def tier1_clean_dryrun(text):
    """Full Tier-1 dry-run pipeline for one text.

    Returns dict with verified/uncertain spans (disjoint), union char counts,
    and the would-be cleaned text. Does NOT modify anything.
    """
    raw = find_tier1_spans(text)
    verified, uncertain = classify_spans(text, raw)
    v_clean, u_all = resolve_verified_uncertain_overlap(verified, uncertain)
    v_union = union_intervals(v_clean)
    u_union = union_intervals(u_all)
    v_chars = sum(e - s for s, e in v_union)
    u_chars = sum(e - s for s, e in u_union)
    cleaned = apply_removal(text, v_clean)
    return {
        "verified_spans": [(s, e, n) for s, e, n in v_clean],
        "uncertain_spans": [(s, e, n) for s, e, n in u_all],
        "verified_chars": v_chars,
        "uncertain_chars": u_chars,
        "total_union_chars": v_chars + u_chars,
        "cleaned_text": cleaned,
        "residual_chars": len(cleaned),
    }
