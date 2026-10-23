#!/usr/bin/env python3
"""Regression tests for V23 Phase 1 Turnitin chrome cleaner.

Covers:
1. The 13 known ai_disclaimer overmatch cases (fixed 2026-10-17): the
   bounded regex must match ONLY the canonical disclaimer, never swallow
   following prose.
2. Confirmed chrome still matches (submission_id, page_marker, disclaimer).
3. Legitimate manuscript text is NEVER matched.

Run:  cd ~/workspace/ai-detector && python3 -m pytest tests/test_chrome_cleaner.py -v
      (or) python3 tests/test_chrome_cleaner.py
"""
import json
import os
import re
import sys
import unittest

PROJ = os.path.expanduser("~/workspace/ai-detector")
sys.path.insert(0, os.path.join(PROJ, "scripts"))

from chrome_patterns import (
    AI_DISCLAIMER_RE,
    AI_DISCLAIMER_RE_OLD_OVERBROAD,
    TIER1_PATTERNS,
    tier1_clean_dryrun,
    union_intervals,
)

SRC = os.path.join(PROJ, "data", "turnitin_mass_paired.jsonl")

# The 13 record idxs whose entire text was swallowed by the old overbroad
# ai_disclaimer regex (no paragraph breaks in extracted text).
OVERMATCH_IDXS = [12, 17, 114, 118, 129, 138, 146, 179, 181, 183, 185, 197, 214]

CANONICAL_DISCLAIMER = (
    "It is essential to understand the limitations of AI detection before "
    "making decisions about a student’s work. We encourage you to learn "
    "more about Turnitin’s AI detection"
)


def load_records():
    recs = []
    with open(SRC, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


class TestAiDisclaimerBounded(unittest.TestCase):
    def test_canonical_disclaimer_matches_exactly(self):
        m = AI_DISCLAIMER_RE.search(CANONICAL_DISCLAIMER)
        self.assertIsNotNone(m, "canonical disclaimer must match")
        self.assertEqual(m.group(0), CANONICAL_DISCLAIMER,
                         "match must be exactly the disclaimer, no tail")

    def test_disclaimer_does_not_swallow_following_prose(self):
        text = CANONICAL_DISCLAIMER + " Made fashion affordable for way more folks, no denying that."
        m = AI_DISCLAIMER_RE.search(text)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(0), CANONICAL_DISCLAIMER)
        self.assertNotIn("Made fashion", m.group(0))

    def test_old_pattern_was_overbroad_on_13_cases(self):
        """Documents the bug: old pattern swallowed whole records."""
        recs = load_records()
        overbroad = 0
        for i in OVERMATCH_IDXS:
            m = AI_DISCLAIMER_RE_OLD_OVERBROAD.search(recs[i]["text"])
            self.assertIsNotNone(m)
            if len(m.group(0)) > 1500:
                overbroad += 1
        self.assertGreater(overbroad, 0,
                           "old pattern must reproduce the overmatch on known cases")

    def test_new_pattern_bounded_on_13_cases(self):
        """The fix: new pattern matches only the disclaimer in all 13."""
        recs = load_records()
        for i in OVERMATCH_IDXS:
            m = AI_DISCLAIMER_RE.search(recs[i]["text"])
            self.assertIsNotNone(m, f"record {i}: disclaimer must still be found")
            self.assertLessEqual(len(m.group(0)), 600,
                                 f"record {i}: match {len(m.group(0))} chars exceeds bound")
            self.assertEqual(m.group(0), CANONICAL_DISCLAIMER,
                             f"record {i}: match must equal canonical disclaimer")

    def test_13_cases_keep_prose_after_tier1(self):
        """After Tier-1 dry-run cleaning, the 13 records keep their prose."""
        recs = load_records()
        for i in OVERMATCH_IDXS:
            res = tier1_clean_dryrun(recs[i]["text"])
            self.assertGreater(res["residual_chars"], 1000,
                               f"record {i}: residual {res['residual_chars']} too small, prose lost?")
            # the disclaimer itself must be gone from cleaned text
            self.assertNotIn("We encourage you to learn more about Turnitin",
                             res["cleaned_text"])


class TestConfirmedChrome(unittest.TestCase):
    def test_submission_id(self):
        rx = dict((n, r) for n, r, _ in TIER1_PATTERNS)["submission_id"]
        m = rx.search("Document Details Submission ID trn:oid:::3618:145517147 Downloaded")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(0), "trn:oid:::3618:145517147")

    def test_page_marker(self):
        rx = dict((n, r) for n, r, _ in TIER1_PATTERNS)["page_marker"]
        self.assertIsNotNone(rx.search("Page 1 of 11 - Cover Page"))
        self.assertIsNotNone(rx.search("page 3 of 8"))

    def test_page_marker_no_overmatch_on_prose(self):
        rx = dict((n, r) for n, r, _ in TIER1_PATTERNS)["page_marker"]
        # "page" in normal prose without the N-of-M shape must not match
        self.assertIsNone(rx.search("turn the page and continue reading the chapter"))


class TestLegitimateProseUntouched(unittest.TestCase):
    """Known-clean human texts must produce ZERO Tier-1 verified spans."""

    CLEAN_FILES = [
        "data/verified/nisar_001_waste.txt",
        "data/verified/human_004_listeria.txt",
        "data/verified/human_006_shakespeare.txt",
    ]

    def test_clean_texts_have_no_tier1_hits(self):
        for rel in self.CLEAN_FILES:
            path = os.path.join(PROJ, rel)
            if not os.path.exists(path):
                self.skipTest(f"missing {rel}")
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            spans = []
            for name, rx, _r in TIER1_PATTERNS:
                spans += [(m.start(), m.end(), name) for m in rx.finditer(text)]
            self.assertEqual(spans, [], f"{rel}: Tier-1 matched legitimate prose: {spans[:2]}")

    def test_prose_with_word_page_not_matched(self):
        text = ("The results are shown on the following page. Turn to page two "
                "for the appendix. Submission of the final draft is due Friday.")
        res = tier1_clean_dryrun(text)
        self.assertEqual(res["verified_chars"], 0)
        self.assertEqual(res["cleaned_text"], text)


class TestDisjointAccounting(unittest.TestCase):
    def test_verified_uncertain_disjoint(self):
        recs = load_records()
        for i, rec in enumerate(recs):
            res = tier1_clean_dryrun(rec["text"])
            v = union_intervals(res["verified_spans"])
            u = union_intervals(res["uncertain_spans"])
            for vs, ve in v:
                for us, ue in u:
                    self.assertFalse(vs < ue and us < ve,
                                     f"record {i}: verified/uncertain overlap")
            self.assertEqual(res["verified_chars"] + res["uncertain_chars"],
                             res["total_union_chars"],
                             f"record {i}: disjoint sum mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
