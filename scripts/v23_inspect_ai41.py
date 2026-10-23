#!/usr/bin/env python3
"""V23 Phase 1: inspect the 41 remaining AI-labelled records (72 quarantined).

For each: apply Tier-1 dry-run cleaning (fixed regex), report residual length
and preview, classify as good/marginal/bad.

- good:     residual >= 500 chars of coherent prose (usable AI training signal)
- marginal: 100-500 chars, or fragments of genuine prose (weak signal)
- bad:      < 100 chars residual, or no genuine prose (templates/listings only)

DRY-RUN. No source modification. Output: data/ai_41_inspection.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from chrome_patterns import tier1_clean_dryrun

PROJ = os.path.expanduser("~/workspace/ai-detector")
DATA = os.path.join(PROJ, "data")
SRC = os.path.join(DATA, "turnitin_mass_paired.jsonl")
QMAN = os.path.join(DATA, "quarantine_manifest_v23.json")
OUT = os.path.join(DATA, "ai_41_inspection.json")

# Manual review notes for the 10 short-residual records (from preview inspection)
MANUAL_NOTES = {
    0: "fragmented technical phrases (5G transceiver); outline-like, weak signal",
    2: "outline fragments (PLC report headings); not coherent prose",
    10: "TOC/figure fragments (DFD); not prose",
    19: "fragments + DOI links; weak signal",
    21: "product listing text (motor specs); not prose at all",
    143: "'Student papers' chrome remnants + equation fragments; weak",
    164: "coherent FIFA World Cup essay prose, short but genuine",
    189: "DFD fragments + bibliography entries; weak",
    191: "fragment of genuine mud-brick prose; short",
    227: "genuine filter-circuit prose fragments; short",
}


def main():
    q = json.load(open(QMAN, encoding="utf-8"))
    qidx = set(r["record_idx"] for r in q["records"])
    recs = []
    with open(SRC, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    remain = sorted(i for i, r in enumerate(recs)
                    if r.get("label") == "ai" and i not in qidx)
    assert len(remain) == 41, f"expected 41, got {len(remain)}"

    inspections = []
    counts = {"good": 0, "marginal": 0, "bad": 0}
    for i in remain:
        rec = recs[i]
        res = tier1_clean_dryrun(rec["text"])
        rc = res["residual_chars"]
        preview = res["cleaned_text"][:400]
        if rc >= 500:
            # check for template-only: mostly headings/URLs?
            cls = "good"
        elif rc >= 100:
            cls = "marginal"
        else:
            cls = "bad"
        # idx 21 is product-listing text, not prose -> bad regardless
        if i == 21:
            cls = "bad"
        counts[cls] += 1
        inspections.append({
            "record_idx": i,
            "record_id": f"mass_{i:04d}",
            "source": rec.get("source"),
            "turnitin_ai": rec.get("turnitin_ai"),
            "total_chars": len(rec["text"]),
            "tier1_removed_chars": res["verified_chars"],
            "residual_chars": rc,
            "classification": cls,
            "manual_note": MANUAL_NOTES.get(i, ""),
            "residual_preview": preview,
        })

    out = {
        "phase": "V23 Phase 1: 41 remaining AI-record inspection (DRY-RUN)",
        "date": "2026-10-17",
        "note": "72 chrome-only AI records quarantined separately; these 41 are the remainder. "
                "Records are Turnitin AI-highlighted SPANS (fragments), not full documents.",
        "counts": counts,
        "usable_for_training": {
            "good": counts["good"],
            "marginal": counts["marginal"],
            "bad": counts["bad"],
        },
        "records": inspections,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"41 inspected: good={counts['good']} marginal={counts['marginal']} bad={counts['bad']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
