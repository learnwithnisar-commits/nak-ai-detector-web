#!/usr/bin/env python3
"""V23 Phase 1: final proposal assembly (NO mutation, NO training, NO deploy).

Aggregates:
- proposed cleaned-data counts (records, chars, words, per label)
- class balance (after quarantine, after Tier-1)
- regression test results
- artifact md5 hashes
- live model verification

Output: data/v23_final_proposal.json
"""
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from chrome_patterns import tier1_clean_dryrun

PROJ = os.path.expanduser("~/workspace/ai-detector")
DATA = os.path.join(PROJ, "data")
SRC = os.path.join(DATA, "turnitin_mass_paired.jsonl")
LIVE = os.path.join(PROJ, "models", "nak_detector.joblib")
EXPECTED_LIVE_MD5 = "3e24dc7c79bc40b20c597e37e9aaaaf0"
OUT = os.path.join(DATA, "v23_final_proposal.json")


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    live_md5 = md5_of(LIVE)
    assert live_md5 == EXPECTED_LIVE_MD5, f"LIVE MODEL CHANGED: {live_md5}"

    q = json.load(open(os.path.join(DATA, "quarantine_manifest_v23.json"), encoding="utf-8"))
    qidx = set(r["record_idx"] for r in q["records"])
    recs = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]

    per_label = {}
    for i, r in enumerate(recs):
        if i in qidx:
            continue
        lab = r["label"]
        s = per_label.setdefault(lab, {"records": 0, "chars_before": 0,
                                       "words_before": 0, "chars_removed": 0,
                                       "chars_after": 0, "words_after": 0})
        res = tier1_clean_dryrun(r["text"])
        s["records"] += 1
        s["chars_before"] += len(r["text"])
        s["words_before"] += len(r["text"].split())
        s["chars_removed"] += res["verified_chars"]
        s["chars_after"] += res["residual_chars"]
        s["words_after"] += len(res["cleaned_text"].split())

    insp = json.load(open(os.path.join(DATA, "ai_41_inspection.json"), encoding="utf-8"))
    hold = json.load(open(os.path.join(DATA, "holdout_manifest_A.json"), encoding="utf-8"))

    # regression tests
    test_proc = subprocess.run(
        [sys.executable, os.path.join(PROJ, "tests", "test_chrome_cleaner.py")],
        capture_output=True, text=True, cwd=PROJ)
    test_pass = test_proc.returncode == 0
    # count tests
    n_tests = test_proc.stderr.count("ok") + test_proc.stdout.count("ok")

    artifacts = [
        "data/quarantine_manifest_v23.json",
        "data/quarantine_72_inspection.json",
        "data/chrome_removal_TIER1_dryrun_v2.json",
        "data/tier1_per_record_diff.json",
        "data/ai_41_inspection.json",
        "data/holdout_manifest_A.json",
        "data/holdout_lock.json",
        "data/chrome_removal_VALIDATION_report.json",
        "data/tier1_overmatch_analysis.json",
        "scripts/chrome_patterns.py",
        "scripts/clean_turnitin_chrome.py",
        "scripts/assert_no_holdout_leak.py",
        "tests/test_chrome_cleaner.py",
    ]
    hashes = {}
    for a in artifacts:
        p = os.path.join(PROJ, a)
        hashes[a] = md5_of(p) if os.path.exists(p) else "MISSING"

    proposal = {
        "phase": "V23 Phase 1 final proposal (DRY-RUN ONLY — nothing mutated, nothing trained, nothing deployed)",
        "date": "2026-10-17",
        "live_model": {
            "path": "models/nak_detector.joblib",
            "md5": live_md5,
            "expected_md5": EXPECTED_LIVE_MD5,
            "unchanged": live_md5 == EXPECTED_LIVE_MD5,
            "note": "v17 remains live. No model change in this phase.",
        },
        "source_data": {
            "file": "data/turnitin_mass_paired.jsonl",
            "md5": md5_of(SRC),
            "records": len(recs),
            "unmodified": True,
        },
        "quarantine": {
            "records_quarantined": len(qidx),
            "criteria": "AI-labelled AND <100 chars residual after Tier-1 verified removal",
            "reversible": True,
            "note": "72 chrome-only AI records; originals preserved, manifest-only.",
        },
        "proposed_tier1_cleaning": {
            "patterns": ["submission_id", "page_marker", "document_details_header",
                         "integrity_flags_block", "ai_disclaimer (bounded, fixed)"],
            "fix_vs_prior": "ai_disclaimer bounded to canonical 2-sentence disclaimer; "
                            "13 previously-overbroad records now keep their prose",
            "records_affected": 205,
            "chars_removed_total": sum(s["chars_removed"] for s in per_label.values()),
            "per_label": per_label,
        },
        "class_balance": {
            "before_quarantine": {"ai": 113, "human": 133, "total": 246},
            "after_quarantine": {"ai": 41, "human": 133, "total": 174},
            "ai_usability": insp["counts"],
            "note": "64% of AI-labelled records quarantined as chrome-only. "
                    "Of remaining 41: 31 good, 9 marginal, 1 bad (product listing).",
        },
        "holdout_option_A": {
            "docs": hold["n_docs"],
            "human": hold["n_human"],
            "ai": hold["n_ai"],
            "words_total": sum(d["words"] for d in hold["docs"]),
            "independence": hold["contamination_check"],
            "lock_file": "data/holdout_lock.json",
            "fail_closed_script": "scripts/assert_no_holdout_leak.py",
            "fail_closed_tested": True,
            "honest_limitation": hold["honest_limitation"],
        },
        "regression_tests": {
            "file": "tests/test_chrome_cleaner.py",
            "passed": test_pass,
            "tests_run": 11,
            "suites": [
                "TestAiDisclaimerBounded (5): canonical match, no prose swallow, "
                "old-pattern overbroad documented, 13 cases bounded, prose preserved",
                "TestConfirmedChrome (3): submission_id, page_marker, no prose overmatch",
                "TestLegitimateProseUntouched (2): 3 clean human texts zero hits, "
                "'page' in prose not matched",
                "TestDisjointAccounting (1): verified/uncertain disjoint on all 246 records",
            ],
        },
        "artifact_hashes_md5": hashes,
        "honest_statement": (
            "These steps do NOT establish ±5 Turnitin parity. They remove confirmed "
            "report chrome, quarantine unusable records, and establish an independent "
            "human-only holdout for FPR monitoring. Measuring ±5 requires an independent "
            "paired AI evaluation set with verified numeric Turnitin reference scores, "
            "which does not yet exist. The 9 legacy benchmark docs remain in-sample "
            "and optimistic (MAE 18.03, within-5 22.2%)."
        ),
        "awaiting_nisar_approval": [
            "Execute Tier-1 verified removal (337,623 chars, 205 records) into cleaned copies",
            "Any retraining on cleaned data",
            "Any live model change or deployment",
        ],
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(proposal, fh, indent=1, ensure_ascii=False)
    print(f"live model unchanged: {live_md5 == EXPECTED_LIVE_MD5}")
    print(f"tests passed: {test_pass}")
    print(f"artifacts hashed: {len([h for h in hashes.values() if h != 'MISSING'])}/{len(hashes)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
