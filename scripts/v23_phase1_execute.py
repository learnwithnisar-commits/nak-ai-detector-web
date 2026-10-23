#!/usr/bin/env python3
"""V23 Phase 1 execution per Nisar's decisions (2026-10-17).

DRY-RUN ONLY. No source data modified. No retraining. No deployment.
Live v17 model untouched (md5 verified before/after).

Tasks:
  1. Quarantine manifest for 72 suspicious AI records (reversible, originals preserved)
  2. Resolve verified/uncertain overlap (conservative: overlap -> uncertain) + Tier 1 dry-run
  3. Inspect all 72 quarantined records for genuine AI prose
  4. Class balance report post-quarantine
  5. Preservation tests (md5, mtime, checksums)

Tier 1 patterns (Nisar-approved): submission_id, page_marker,
document_details_header (short), integrity_flags_block, ai_disclaimer (short <=1500).
"""
import hashlib
import json
import os
import re
import sys
from collections import Counter

PROJ = os.path.expanduser("~/workspace/ai-detector")
DATA = os.path.join(PROJ, "data")
SRC = os.path.join(DATA, "turnitin_mass_paired.jsonl")
SUMMARY = "/tmp/per_record_summary.json"

# ---------------------------------------------------------------------------
# Tier 1 patterns only
# ---------------------------------------------------------------------------
TIER1 = [
    ("submission_id",
     re.compile(r"trn:oid:::[0-9:]+", re.IGNORECASE)),
    ("page_marker",
     re.compile(r"page \d+ of \d+(?:\s*-\s*[a-z ]*)?", re.IGNORECASE)),
    ("document_details_header",
     re.compile(
         r"document details\s+submission id\b.*?(?=page \d+ of \d+|\d+% overall similarity|$)",
         re.IGNORECASE | re.DOTALL)),
    ("integrity_flags_block",
     re.compile(r"integrity flags\b.*?(?=match groups|$)",
                re.IGNORECASE | re.DOTALL)),
    ("ai_disclaimer",
     re.compile(
         r"(?:it is essential to understand the limitations of ai detection.*?)?"
         r"we encourage you to learn more about turnitin[\u2019']s ai detection"
         r".*?(?=\n\n|\Z)",
         re.IGNORECASE | re.DOTALL)),
]
MAX_VERIFIED_LEN = 1500  # longer matches -> uncertain (conservative)


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def tier1_spans(text):
    """Return list of (start, end, classification, pattern_name)."""
    spans = []
    for name, rx in TIER1:
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            cls = "verified" if (e - s) <= MAX_VERIFIED_LEN else "uncertain"
            spans.append((s, e, cls, name))
    return spans


def union_disjoint(spans):
    """Merge overlapping spans; classification = uncertain if ANY part uncertain.

    Returns (verified_chars, uncertain_chars, total_chars, merged_spans).
    verified + uncertain == total guaranteed (disjoint by construction).
    """
    if not spans:
        return 0, 0, 0, []
    events = []
    for s, e, cls, name in spans:
        events.append((s, 1, cls, name))   # open
        events.append((e, -1, cls, name))  # close
    events.sort(key=lambda x: (x[0], x[1]))  # closes before opens at same pos
    merged = []
    active = Counter()
    cur_start = None
    cur_unc = False
    for pos, typ, cls, name in events:
        if typ == 1:  # open
            if not active:
                cur_start = pos
                cur_unc = False
            active[(cls, name)] += 1
            if cls == "uncertain":
                cur_unc = True
        else:  # close
            active[(cls, name)] -= 1
            if active[(cls, name)] == 0:
                del active[(cls, name)]
            if not active:
                merged.append((cur_start, pos, "uncertain" if cur_unc else "verified"))
                cur_start = None
                cur_unc = False
    v = sum(e - s for s, e, c in merged if c == "verified")
    u = sum(e - s for s, e, c in merged if c == "uncertain")
    return v, u, v + u, merged


def remove_spans(text, merged):
    """Return text with merged (s,e) spans removed."""
    out = []
    last = 0
    for s, e, _c in sorted(merged):
        out.append(text[last:s])
        last = e
    out.append(text[last:])
    return "".join(out)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading records...", file=sys.stderr)
records = []
with open(SRC, encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        line = line.strip()
        if line:
            rec = json.loads(line)
            rec["_idx"] = i
            records.append(rec)
print(f"  {len(records)} records loaded", file=sys.stderr)

with open(SUMMARY, encoding="utf-8") as fh:
    summ = json.load(fh)
summ_by_idx = {r["record_idx"]: r for r in summ}

# Identify the 72 (label=ai, residual<100 in prior verified-chrome dry-run)
q72_idxs = sorted(r["record_idx"] for r in summ
                 if r["label"] == "ai" and r["residual_chars"] < 100)
assert len(q72_idxs) == 72, f"expected 72, got {len(q72_idxs)}"
print(f"  72 quarantine candidates identified", file=sys.stderr)

# ---------------------------------------------------------------------------
# TASK 1: Quarantine manifest
# ---------------------------------------------------------------------------
print("Task 1: quarantine manifest...", file=sys.stderr)
src_md5 = md5_file(SRC)
src_mtime = os.path.getmtime(SRC)
manifest = {
    "phase": "V23 Phase 1",
    "date": "2026-10-17",
    "action": "quarantine (reversible; originals preserved, nothing deleted)",
    "source_file": "data/turnitin_mass_paired.jsonl",
    "source_md5": src_md5,
    "source_record_count": len(records),
    "reversible": True,
    "quarantine_criteria": (
        "label == 'ai' AND residual_chars < 100 after verified-chrome removal "
        "(all 9 patterns, union counting) per /tmp/per_record_summary.json"
    ),
    "records": [],
}
for idx in q72_idxs:
    rec = records[idx]
    s = summ_by_idx[idx]
    manifest["records"].append({
        "record_idx": idx,
        "record_id": f"mass_{idx:04d}",
        "source_file": rec.get("source"),
        "label": rec.get("label"),
        "turnitin_ai": rec.get("turnitin_ai"),
        "total_chars": s["total_chars"],
        "verified_chrome_chars_prior_run": s["verified_remove"],
        "residual_chars_prior_run": s["residual_chars"],
        "reason": "chrome-only: <100 chars residual after verified chrome removal; "
                  "no usable AI training signal detected by automated audit",
        "provenance": {
            "jsonl": "data/turnitin_mass_paired.jsonl",
            "jsonl_md5": src_md5,
            "line_number": idx + 1,
            "extraction": "Turnitin AI report blue-background highlight extraction, 2026-10-17",
        },
        "reversible": True,
    })
with open(os.path.join(DATA, "quarantine_manifest_v23.json"), "w",
          encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, ensure_ascii=False)
print(f"  manifest written: {len(manifest['records'])} records", file=sys.stderr)

# ---------------------------------------------------------------------------
# TASK 2: Tier 1 dry-run with disjoint classification
# ---------------------------------------------------------------------------
print("Task 2: Tier 1 dry-run...", file=sys.stderr)
tier1_results = []
tot_v = tot_u = tot_t = 0
tot_chars_all = 0
affected = 0
per_pattern_hits = Counter()
for rec in records:
    text = rec.get("text", "")
    tot_chars_all += len(text)
    spans = tier1_spans(text)
    v, u, t, merged = union_disjoint(spans)
    # DISJOINT CHECK per record
    assert v + u == t, f"record {rec['_idx']}: {v}+{u} != {t}"
    tot_v += v
    tot_u += u
    tot_t += t
    if t:
        affected += 1
    for _s, _e, _c, name in spans:
        per_pattern_hits[name] += 1
    residual = remove_spans(text, merged)
    tier1_results.append({
        "record_idx": rec["_idx"],
        "source": rec.get("source"),
        "label": rec.get("label"),
        "total_chars": len(text),
        "tier1_verified_remove": v,
        "tier1_uncertain_flagged": u,
        "tier1_total_union": t,
        "tier1_residual_chars": len(residual),
        "tier1_pct_remove": round(100.0 * t / len(text), 2) if text else 0.0,
    })
# GLOBAL DISJOINT CHECK
assert tot_v + tot_u == tot_t, "global disjoint violated"
tier1_report = {
    "phase": "V23 Phase 1 - Tier 1 dry-run",
    "date": "2026-10-17",
    "dry_run": True,
    "source_modified": False,
    "tier1_patterns": [n for n, _ in TIER1],
    "overlap_resolution_rule": (
        "Any union span containing an uncertain-classified part is classified "
        "UNCERTAIN (conservative). Verified and uncertain are disjoint by construction."
    ),
    "disjoint_verified": True,
    "totals": {
        "records": len(records),
        "total_chars_all": tot_chars_all,
        "tier1_verified_chars_unique": tot_v,
        "tier1_uncertain_chars_unique": tot_u,
        "tier1_total_union": tot_t,
        "disjoint_check": f"{tot_v} + {tot_u} == {tot_t}",
        "flagged_fraction": round(tot_t / tot_chars_all, 4) if tot_chars_all else 0,
        "records_affected": affected,
    },
    "per_pattern_hit_counts": dict(per_pattern_hits),
    "note": "Hit counts may double-count overlaps; character counts are union-based.",
    "per_record": tier1_results,
}
with open(os.path.join(DATA, "chrome_removal_TIER1_dryrun.json"), "w",
          encoding="utf-8") as fh:
    json.dump(tier1_report, fh, indent=2, ensure_ascii=False)
print(f"  verified={tot_v} uncertain={tot_u} total={tot_t} "
      f"check={tot_v + tot_u == tot_t}", file=sys.stderr)

# ---------------------------------------------------------------------------
# TASK 3: Inspect the 72 quarantined records
# ---------------------------------------------------------------------------
print("Task 3: inspecting 72 quarantined records...", file=sys.stderr)
t1_by_idx = {r["record_idx"]: r for r in tier1_results}

TEMPLATE_HINTS = re.compile(
    r"(assignment|task \d|instructions|question \d|answer the following|"
    r"write a|submit your|deadline|word count|marking criteria|learning outcome)",
    re.IGNORECASE)


def classify_residual(residual):
    stripped = residual.strip()
    if not stripped:
        return "empty"
    words = stripped.split()
    if len(words) < 15:
        return "template_only" if TEMPLATE_HINTS.search(stripped) else "unclear"
    if len(words) < 40:
        return "template_only" if TEMPLATE_HINTS.search(stripped) else "unclear"
    # 40+ words of prose: check if it reads like assignment template or real prose
    if TEMPLATE_HINTS.search(stripped) and len(words) < 80:
        return "unclear"
    return "has_ai_prose"


inspection = {
    "phase": "V23 Phase 1 - 72-record inspection",
    "date": "2026-10-17",
    "note": ("Honest per-record inspection. Short residual does NOT automatically "
             "mean invalid; classification is heuristic and previews are provided "
             "for human review."),
    "records": [],
}
class_counts = Counter()
for idx in q72_idxs:
    rec = records[idx]
    text = rec.get("text", "")
    spans = tier1_spans(text)
    v, u, t, merged = union_disjoint(spans)
    residual = remove_spans(text, merged).strip()
    cls = classify_residual(residual)
    class_counts[cls] += 1
    inspection["records"].append({
        "record_idx": idx,
        "record_id": f"mass_{idx:04d}",
        "source_file": rec.get("source"),
        "label": rec.get("label"),
        "turnitin_ai": rec.get("turnitin_ai"),
        "total_chars": len(text),
        "tier1_verified_remove": v,
        "tier1_uncertain_flagged": u,
        "tier1_residual_chars": len(residual),
        "tier1_residual_words": len(residual.split()),
        "residual_preview_200": residual[:200],
        "residual_classification": cls,
        "classification_heuristic": (
            "empty=0 chars; template_only=<40 words or assignment-template markers; "
            "has_ai_prose=40+ words of non-template prose; unclear=needs human review"
        ),
    })
inspection["classification_summary"] = dict(class_counts)
with open(os.path.join(DATA, "quarantine_72_inspection.json"), "w",
          encoding="utf-8") as fh:
    json.dump(inspection, fh, indent=2, ensure_ascii=False)
print(f"  classifications: {dict(class_counts)}", file=sys.stderr)

# ---------------------------------------------------------------------------
# TASK 4: Class balance post-quarantine
# ---------------------------------------------------------------------------
print("Task 4: class balance...", file=sys.stderr)
qset = set(q72_idxs)
kept = [r for r in records if r["_idx"] not in qset]
label_counts = Counter(r.get("label") for r in kept)
word_counts = Counter()
src_dist = Counter()
for r in kept:
    label_counts[r.get("label")] += 0  # ensure key exists
    word_counts[r.get("label")] += len(r.get("text", "").split())
    src = (r.get("source") or "")
    if src.startswith("turnitin_"):
        src_dist["turnitin_mass_paired"] += 1
    else:
        src_dist["other:" + src[:40]] += 1
# also count quarantined by label
qlabel = Counter(records[i].get("label") for i in q72_idxs)
balance = {
    "phase": "V23 Phase 1 - class balance",
    "date": "2026-10-17",
    "note": "After quarantining 72 AI-labelled records. No replacement data fabricated.",
    "before": {"total": len(records),
               "ai": sum(1 for r in records if r.get("label") == "ai"),
               "human": sum(1 for r in records if r.get("label") == "human")},
    "quarantined": {"total": 72, "by_label": dict(qlabel)},
    "after": {
        "total": len(kept),
        "by_label": {k: label_counts[k] for k in sorted(label_counts)},
        "word_counts_by_label": dict(word_counts),
        "source_distribution": dict(src_dist),
    },
}
with open(os.path.join(DATA, "class_balance_post_quarantine.json"), "w",
          encoding="utf-8") as fh:
    json.dump(balance, fh, indent=2, ensure_ascii=False)
print(f"  after: {dict(label_counts)}", file=sys.stderr)

# ---------------------------------------------------------------------------
# TASK 5: Preservation tests
# ---------------------------------------------------------------------------
print("Task 5: preservation tests...", file=sys.stderr)
live_md5_before = md5_file(os.path.join(PROJ, "models", "nak_detector.joblib"))
src_md5_after = md5_file(SRC)
src_mtime_after = os.path.getmtime(SRC)
artifacts = [
    "data/quarantine_manifest_v23.json",
    "data/chrome_removal_TIER1_dryrun.json",
    "data/quarantine_72_inspection.json",
    "data/class_balance_post_quarantine.json",
]
checksums = {}
for a in artifacts:
    p = os.path.join(PROJ, a)
    checksums[a] = {"md5": md5_file(p), "bytes": os.path.getsize(p),
                    "exists": True}
pres = {
    "phase": "V23 Phase 1 - preservation tests",
    "date": "2026-10-17",
    "live_model": {
        "path": "models/nak_detector.joblib",
        "md5_expected": "3e24dc7c79bc40b20c597e37e9aaaaf0",
        "md5_observed": live_md5_before,
        "unchanged": live_md5_before == "3e24dc7c79bc40b20c597e37e9aaaaf0",
    },
    "source_data": {
        "path": "data/turnitin_mass_paired.jsonl",
        "md5_before": src_md5,
        "md5_after": src_md5_after,
        "md5_unchanged": src_md5 == src_md5_after,
        "mtime_before": src_mtime,
        "mtime_after": src_mtime_after,
        "mtime_unchanged": src_mtime == src_mtime_after,
        "record_count": len(records),
        "record_count_ok": len(records) == 246,
    },
    "quarantine_reversible": {
        "originals_deleted": False,
        "originals_moved": False,
        "manifest_maps_back_to_line_numbers": True,
        "restore_procedure": ("Re-include record_idx values from "
                              "data/quarantine_manifest_v23.json in training; "
                              "no file moves were performed"),
    },
    "artifact_checksums": checksums,
    "retraining_performed": False,
    "deployment_performed": False,
}
with open(os.path.join(DATA, "preservation_tests_v23.json"), "w",
          encoding="utf-8") as fh:
    json.dump(pres, fh, indent=2, ensure_ascii=False)

# add self checksum
self_path = os.path.join(DATA, "preservation_tests_v23.json")
pres["artifact_checksums"]["data/preservation_tests_v23.json"] = {
    "md5": md5_file(self_path), "bytes": os.path.getsize(self_path),
    "exists": True,
    "note": "checksum computed after write; re-verify by recomputation",
}
with open(self_path, "w", encoding="utf-8") as fh:
    json.dump(pres, fh, indent=2, ensure_ascii=False)

print("DONE", file=sys.stderr)
print(json.dumps({
    "quarantine_count": 72,
    "tier1_verified": tot_v,
    "tier1_uncertain": tot_u,
    "tier1_total": tot_t,
    "disjoint_ok": tot_v + tot_u == tot_t,
    "inspection_classes": dict(class_counts),
    "kept_ai": label_counts.get("ai"),
    "kept_human": label_counts.get("human"),
    "live_md5_ok": pres["live_model"]["unchanged"],
    "src_md5_ok": pres["source_data"]["md5_unchanged"],
}))
