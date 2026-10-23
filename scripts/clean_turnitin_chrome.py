#!/usr/bin/env python3
"""V23 Phase 1 DRY-RUN: Turnitin report-chrome detection audit.

DOES NOT MODIFY ANY ORIGINAL FILES. Read-only analysis that classifies
each proposed chrome removal as `verified_chrome` or `uncertain`.

Per Nisar's correction: sentence-level regex may remove genuine prose or
miss embedded boilerplate. This dry-run flags uncertain cases for human
review instead of auto-deleting.

Input:  data/turnitin_mass_paired.jsonl (schema: text, label, source, turnitin_ai)
Output: data/chrome_removal_DRYRUN_report.json

Unicode-safe: reads/writes UTF-8, regexes use re.UNICODE (default in py3).
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from chrome_patterns import (
    TIER1_PATTERNS,
    TIER2_PATTERNS,
    UNCERTAIN_HINTS,
    union_intervals,
    split_sentences_simple,
)

PROJ = os.path.expanduser("~/workspace/ai-detector")
DATA = os.path.join(PROJ, "data")
SRC = os.path.join(DATA, "turnitin_mass_paired.jsonl")
OUT = os.path.join(DATA, "chrome_removal_DRYRUN_report.json")

# ---------------------------------------------------------------------------
# 9 chrome patterns — single source of truth in scripts/chrome_patterns.py.
# V23 FIX: ai_disclaimer is now bounded (no more whole-record swallows).
# Each entry: (name, compiled_regex, classification, rationale)
# ---------------------------------------------------------------------------
_RATIONALE = {
    "submission_id": "Turnitin submission identifier; never part of student prose.",
    "document_details_header": "Report cover-page metadata block (submission/download dates, file name/size).",
    "page_marker": "PDF page header/footer inserted by the report renderer.",
    "overall_similarity_block": "Similarity-report summary boilerplate.",
    "top_sources_block": "Similarity-report source-category boilerplate.",
    "integrity_flags_block": "Report integrity-flag boilerplate.",
    "match_groups_block": "Similarity match-group boilerplate (not cited/quoted, missing quotations, etc.).",
    "ai_disclaimer": "AI-report disclaimer footer (bounded two-sentence match); report chrome, not student prose.",
    "file_metadata": "File metadata line from report cover page.",
}
PATTERNS = [
    (name, rx, "verified_chrome", _RATIONALE[name])
    for name, rx, _r in TIER1_PATTERNS
] + [
    (name, rx, "verified_chrome", _RATIONALE[name])
    for name, rx in TIER2_PATTERNS
]


def split_sentences_simple(text):
    """Lightweight sentence splitter for audit purposes (not the app's)."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def audit_record(rec):
    text = rec.get("text", "")
    findings = []
    for name, rx, classification, rationale in PATTERNS:
        for m in rx.finditer(text):
            span = m.group(0)
            # Guard: extremely long matches (>1500 chars) may have swallowed prose
            if len(span) > 1500:
                cls = "uncertain"
                note = "match exceeds 1500 chars; may include genuine prose"
            else:
                cls = classification
                note = rationale
            findings.append(
                {
                    "pattern": name,
                    "classification": cls,
                    "note": note,
                    "match_len": len(span),
                    "match_preview": span[:200],
                }
            )
    # Uncertain: sentences with chrome hints but no verified pattern hit
    matched_spans = []
    for name, rx, _cls, _r in PATTERNS:
        matched_spans += [(m.start(), m.end()) for m in rx.finditer(text)]

    def in_match(pos):
        return any(s <= pos < e for s, e in matched_spans)

    for sent in split_sentences_simple(text):
        if not UNCERTAIN_HINTS.search(sent):
            continue
        idx = text.find(sent)
        if idx >= 0 and in_match(idx):
            continue  # already covered by a verified pattern
        findings.append(
            {
                "pattern": "hint_only",
                "classification": "uncertain",
                "note": "contains chrome hint words but no verified pattern matched; needs human review",
                "match_len": len(sent),
                "match_preview": sent[:200],
            }
        )
    return findings


def union_intervals(spans):
    """Merge overlapping (start, end) intervals; return merged list.

    FIX (V23 audit): the original code summed match_len across all pattern
    hits, but the 9 chrome regexes overlap heavily (e.g. document_details_header
    CONTAINS the submission_id match). That double/triple-counted characters,
    producing flagged > total. Union counting is the correct metric.
    """
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


def main():
    # Safety: confirm source exists and record mtimes BEFORE (validation that we don't modify)
    mtimes_before = {}
    for f in [SRC]:
        mtimes_before[f] = os.path.getmtime(f)

    records = []
    with open(SRC, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARN: line {i} JSON decode error: {e}", file=sys.stderr)
                continue
            # Schema check
            for k in ("text", "label", "source"):
                if k not in rec:
                    print(f"WARN: line {i} missing key {k}", file=sys.stderr)
            records.append(rec)

    per_pattern = Counter()
    per_class = Counter()
    per_label_pattern = Counter()
    examples = defaultdict(list)  # (pattern, classification) -> up to 3 previews
    records_affected = 0
    total_chars_flagged_naive = 0  # legacy buggy metric, kept for comparison
    total_chars_flagged_union = 0  # CORRECT metric: union of spans per record
    union_verified_chars = 0
    union_uncertain_chars = 0
    total_chars_all = 0
    worst_overlap = None

    for ri, rec in enumerate(records):
        total_chars_all += len(rec.get("text", ""))
        findings = audit_record(rec)
        if findings:
            records_affected += 1
        # FIX: union-count spans instead of summing match_len (overlaps double-counted)
        span_tuples = [(0, 0)]  # placeholder replaced below
        span_tuples = []
        for f in findings:
            # recover span positions: re-run patterns to get (start, end)
            span_tuples.append((f, f["match_len"]))
        # Recompute spans with positions for union counting
        text = rec.get("text", "")
        pos_spans = []
        for name, rx, classification, rationale in PATTERNS:
            for m in rx.finditer(text):
                s, e = m.start(), m.end()
                cls = "uncertain" if (e - s) > 1500 else classification
                pos_spans.append((s, e, cls))
        # hint_only spans
        matched = [(s, e) for s, e, _ in pos_spans]
        def _in_match(pos):
            return any(s <= pos < e for s, e in matched)
        for sent in split_sentences_simple(text):
            if not UNCERTAIN_HINTS.search(sent):
                continue
            idx = text.find(sent)
            if idx >= 0 and not _in_match(idx):
                pos_spans.append((idx, idx + len(sent), "uncertain"))
        for f in findings:
            per_pattern[f["pattern"]] += 1
            per_class[f["classification"]] += 1
            per_label_pattern[(rec.get("label"), f["pattern"])] += 1
            total_chars_flagged_naive += f["match_len"]
            key = (f["pattern"], f["classification"])
            if len(examples[key]) < 3:
                examples[key].append(
                    {
                        "record_idx": ri,
                        "source": rec.get("source"),
                        "label": rec.get("label"),
                        "preview": f["match_preview"],
                        "match_len": f["match_len"],
                    }
                )
        merged_all = union_intervals(pos_spans)
        u_all = sum(e - s for s, e in merged_all)
        total_chars_flagged_union += u_all
        merged_v = union_intervals([(s, e, c) for s, e, c in pos_spans if c == "verified_chrome"])
        union_verified_chars += sum(e - s for s, e in merged_v)
        merged_u = union_intervals([(s, e, c) for s, e, c in pos_spans if c == "uncertain"])
        union_uncertain_chars += sum(e - s for s, e in merged_u)
        if pos_spans and u_all > 0:
            ov = sum(f["match_len"] for f in findings)
            factor = ov / u_all
            if worst_overlap is None or factor > worst_overlap[4]:
                worst_overlap = (ri, rec.get("source"), ov, u_all, round(factor, 2))

    # Validation: originals untouched
    mtimes_after = {f: os.path.getmtime(f) for f in mtimes_before}
    unmodified = all(mtimes_before[f] == mtimes_after[f] for f in mtimes_before)

    report = {
        "phase": "V23 Phase 1 DRY-RUN (no files modified)",
        "source_file": SRC,
        "total_records": len(records),
        "records_with_any_finding": records_affected,
        "records_unaffected": len(records) - records_affected,
        "per_pattern_hits": dict(per_pattern),
        "per_classification_hits": dict(per_class),
        "per_label_pattern_hits": {
            f"{label}/{pat}": n for (label, pat), n in per_label_pattern.items()
        },
        "total_chars_flagged": total_chars_flagged_union,
        "total_chars_all": total_chars_all,
        "flagged_char_fraction": round(total_chars_flagged_union / max(total_chars_all, 1), 4),
        "total_chars_flagged_verified_chrome_union": union_verified_chars,
        "total_chars_flagged_uncertain_union": union_uncertain_chars,
        "legacy_naive_total_chars_flagged_BUGGY": total_chars_flagged_naive,
        "legacy_naive_flagged_fraction_BUGGY": round(total_chars_flagged_naive / max(total_chars_all, 1), 4),
        "worst_overlap_record": (
            {
                "record_idx": worst_overlap[0],
                "source": worst_overlap[1],
                "naive_chars": worst_overlap[2],
                "union_chars": worst_overlap[3],
                "overcount_factor": worst_overlap[4],
            }
            if worst_overlap
            else None
        ),
        "originals_unmodified": unmodified,
        "mtimes_before": mtimes_before,
        "mtimes_after": mtimes_after,
        "examples": {
            f"{pat} [{cls}]": exs for (pat, cls), exs in examples.items()
        },
        "notes": [
            "verified_chrome = definitely Turnitin report boilerplate; safe to strip.",
            "uncertain = contains chrome hint words but no verified pattern matched; MUST be human-reviewed before any deletion.",
            "No cleaned copies were created in dry-run mode.",
            "Unicode handled via UTF-8 read and re.UNICODE regex (Python 3 default).",
        ],
    }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"records: {len(records)}, affected: {records_affected}")
    print(f"per_pattern: {dict(per_pattern)}")
    print(f"per_class: {dict(per_class)}")
    print(f"flagged_char_fraction: {report['flagged_char_fraction']}")
    print(f"originals_unmodified: {unmodified}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
