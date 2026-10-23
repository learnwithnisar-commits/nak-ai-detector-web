#!/usr/bin/env python3
"""V23 Phase 1: Tier-1 dry-run RERUN with fixed ai_disclaimer regex + per-record diff.

DRY-RUN ONLY. Reads data/turnitin_mass_paired.jsonl, writes:
- data/chrome_removal_TIER1_dryrun_v2.json  (rerun totals with fixed regex)
- data/tier1_per_record_diff.json           (per-record before/after diff)

Verifies source file md5/mtime unchanged. Never modifies source data.
"""
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from chrome_patterns import (
    TIER1_PATTERNS,
    tier1_clean_dryrun,
    union_intervals,
)

PROJ = os.path.expanduser("~/workspace/ai-detector")
DATA = os.path.join(PROJ, "data")
SRC = os.path.join(DATA, "turnitin_mass_paired.jsonl")
OUT_DRYRUN = os.path.join(DATA, "chrome_removal_TIER1_dryrun_v2.json")
OUT_DIFF = os.path.join(DATA, "tier1_per_record_diff.json")


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    md5_before = md5_of(SRC)
    mtime_before = os.path.getmtime(SRC)

    records = []
    with open(SRC, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if line:
                records.append(json.loads(line))

    per_record = []
    per_record_diff = []
    pat_hits = Counter()
    tot_v = tot_u = tot_all = 0
    affected = 0

    for i, rec in enumerate(records):
        text = rec.get("text", "")
        tot_all += len(text)
        res = tier1_clean_dryrun(text)
        tot_v += res["verified_chars"]
        tot_u += res["uncertain_chars"]
        if res["verified_chars"] or res["uncertain_chars"]:
            affected += 1
        for _s, _e, name in res["verified_spans"]:
            pat_hits[name] += 1

        # disjoint sanity per record
        assert res["verified_chars"] + res["uncertain_chars"] == res["total_union_chars"]

        cleaned = res["cleaned_text"]
        per_record.append({
            "record_idx": i,
            "source": rec.get("source"),
            "label": rec.get("label"),
            "total_chars": len(text),
            "tier1_verified_remove": res["verified_chars"],
            "tier1_uncertain_flagged": res["uncertain_chars"],
            "tier1_total_union": res["total_union_chars"],
            "tier1_residual_chars": res["residual_chars"],
            "tier1_pct_remove": round(res["verified_chars"] / max(len(text), 1), 4),
        })

        # before/after snippets for human review
        spans = sorted(res["verified_spans"])
        removed_previews = [
            {"pattern": n, "start": s, "end": e,
             "removed_text": text[s:e][:300]}
            for s, e, n in spans
        ]
        per_record_diff.append({
            "record_idx": i,
            "source": rec.get("source"),
            "label": rec.get("label"),
            "total_chars": len(text),
            "removed_spans": removed_previews,
            "num_removed_spans": len(removed_previews),
            "chars_removed": res["verified_chars"],
            "residual_chars": res["residual_chars"],
            "before_head": text[:300],
            "before_tail": text[-300:] if len(text) > 300 else "",
            "after_head": cleaned[:300],
            "after_tail": cleaned[-300:] if len(cleaned) > 300 else "",
            "after_full": cleaned,  # full cleaned text for audit (dry-run artifact only)
        })

    dryrun = {
        "phase": "V23 Phase 1 Tier-1 DRY-RUN v2 (fixed ai_disclaimer regex, no files modified)",
        "date": "2026-10-17",
        "dry_run": True,
        "source_modified": False,
        "source_md5": md5_before,
        "tier1_patterns": [n for n, _r, _d in TIER1_PATTERNS],
        "fix": "ai_disclaimer bounded to canonical two-sentence disclaimer (was overbroad DOTALL swallowing whole records)",
        "overlap_resolution_rule": "any verified span touching an uncertain span is demoted to uncertain; verified/uncertain disjoint by construction",
        "totals": {
            "records": len(records),
            "total_chars_all": tot_all,
            "tier1_verified_chars_unique": tot_v,
            "tier1_uncertain_chars_unique": tot_u,
            "tier1_total_union": tot_v + tot_u,
            "disjoint_check": f"{tot_v} + {tot_u} == {tot_v + tot_u}",
            "flagged_fraction": round((tot_v + tot_u) / max(tot_all, 1), 4),
            "records_affected": affected,
        },
        "per_pattern_verified_span_counts": dict(pat_hits),
        "per_record": per_record,
    }

    with open(OUT_DRYRUN, "w", encoding="utf-8") as fh:
        json.dump(dryrun, fh, indent=1, ensure_ascii=False)
    with open(OUT_DIFF, "w", encoding="utf-8") as fh:
        json.dump({
            "phase": "V23 Phase 1 per-record Tier-1 before/after diff (DRY-RUN, no source modified)",
            "date": "2026-10-17",
            "source_md5": md5_before,
            "records": per_record_diff,
        }, fh, indent=1, ensure_ascii=False)

    # verify source untouched
    assert md5_of(SRC) == md5_before, "SOURCE MODIFIED!"
    assert os.path.getmtime(SRC) == mtime_before, "SOURCE MTIME CHANGED!"
    print(f"records={len(records)} affected={affected}")
    print(f"verified={tot_v} uncertain={tot_u} total_union={tot_v + tot_u} frac={dryrun['totals']['flagged_fraction']}")
    print(f"pattern hits: {dict(pat_hits)}")
    print(f"source md5 unchanged: {md5_before}")
    print(f"wrote {OUT_DRYRUN}")
    print(f"wrote {OUT_DIFF}")


if __name__ == "__main__":
    main()
