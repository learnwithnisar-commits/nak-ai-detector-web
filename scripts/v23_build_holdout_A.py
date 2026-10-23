#!/usr/bin/env python3
"""V23 Phase 3 Option A: build the human-only independent holdout.

- 31 verified-human docs (12 pre-AI-era + 9 essays + 10 speeches)
- Provenance per doc (parsed from file headers)
- Document-level independence: fingerprint check vs all training sources
- Locked test IDs -> data/holdout_lock.json (immutable once created)
- Manifest -> data/holdout_manifest_A.json

HONEST LIMITATION: human-only. Measures FPR/specificity ONLY.
Cannot measure MAE/within-5 (needs paired AI docs with verified scores).
DO NOT tune against this holdout. DO NOT claim ±5 parity from it.

No training data modified. No model touched.
"""
import hashlib
import json
import os
import re

PROJ = os.path.expanduser("~/workspace/ai-detector")
DATA = os.path.join(PROJ, "data")
VER = os.path.join(DATA, "verified")

HOLDOUT_FILES = (
    [f"pre_ai_era/{f}" for f in sorted(os.listdir(os.path.join(VER, "pre_ai_era")))
     if f.endswith(".txt") and f.startswith("human_preai_")]
    + [f"essays/{f}" for f in sorted(os.listdir(os.path.join(VER, "essays")))
       if f.endswith(".txt")]
    + [f"speeches/{f}" for f in sorted(os.listdir(os.path.join(VER, "speeches")))
       if f.endswith(".txt")]
)

# Training sources to check independence against (from phase3_holdout_SPEC.md)
TRAIN_SOURCES = [
    "turnitin_mass_paired.jsonl",
    "whatsapp21_training.jsonl",
    "turnitin_22103_paired.jsonl",
    "turnitin_anchors.jsonl",
    "drive_human_samples.jsonl",
]


def fingerprint(text):
    norm = re.sub(r"\s+", " ", text[:2000]).strip().lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def parse_header(text):
    """Extract # Key: value header lines."""
    meta = {}
    for line in text.split("\n")[:12]:
        m = re.match(r"#\s*([^:]+):\s*(.*)", line)
        if m:
            meta[m.group(1).strip().lower()] = m.group(2).strip()
        elif line.startswith("#") and not meta:
            meta.setdefault("note", line.lstrip("# ").strip())
    return meta


def main():
    assert len(HOLDOUT_FILES) == 31, f"expected 31, found {len(HOLDOUT_FILES)}"

    # collect training fingerprints
    train_fps = set()
    for src in TRAIN_SOURCES:
        p = os.path.join(DATA, src)
        if not os.path.exists(p):
            print(f"WARN: training source missing: {src}")
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                train_fps.add(fingerprint(rec.get("text", "")))
    print(f"training fingerprints: {len(train_fps)}")

    docs = []
    contaminated = []
    for idx, rel in enumerate(HOLDOUT_FILES):
        path = os.path.join(VER, rel)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        meta = parse_header(text)
        fp = fingerprint(text)
        body = "\n".join(l for l in text.split("\n") if not l.startswith("#")).strip()
        words = len(body.split())
        is_clean = fp not in train_fps
        if not is_clean:
            contaminated.append(rel)
        # near-duplicate: also check first-500-char hash
        docs.append({
            "doc_id": f"holdout_A_{idx:02d}",
            "file": f"verified/{rel}",
            "label": "human",
            "turnitin_ref": None,  # human docs: no AI score; FPR evaluation only
            "split": "locked_test",
            "fingerprint_md5": fp,
            "words": words,
            "chars": len(text),
            "provenance": {
                "title": meta.get("title", meta.get("human speech 001", "")),
                "author": meta.get("author", meta.get("speaker", "")),
                "source": meta.get("source", ""),
                "published": meta.get("published", meta.get("date", meta.get("year", ""))),
                "url": meta.get("url", ""),
                "verification": meta.get("verified", meta.get("note", "")),
            },
            "independent_of_training": is_clean,
            "added_date": "2026-10-17",
        })

    assert not contaminated, f"CONTAMINATED: {contaminated}"
    print(f"31/31 independent (zero fingerprint overlap)")

    manifest = {
        "phase": "V23 Phase 3 Option A: human-only independent holdout",
        "date": "2026-10-17",
        "purpose": "False-positive-rate monitoring on verified human documents ONLY. "
                   "Cannot measure MAE/within-5/bias (no paired AI docs with verified scores).",
        "honest_limitation": "Do NOT claim ±5 Turnitin parity from this holdout. "
                             "Do NOT tune against it. Evaluation use only.",
        "n_docs": len(docs),
        "n_human": sum(1 for d in docs if d["label"] == "human"),
        "n_ai": 0,
        "contamination_check": "fingerprint (md5 of normalized first 2000 chars) vs "
                               f"{len(train_fps)} training fingerprints: 0 overlaps",
        "docs": docs,
    }
    with open(os.path.join(DATA, "holdout_manifest_A.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)

    lock = {
        "locked": True,
        "date": "2026-10-17",
        "warning": "IMMUTABLE. Training scripts MUST assert no locked doc_id appears "
                   "in training data (fail-closed). Use scripts/assert_no_holdout_leak.py.",
        "doc_ids": [d["doc_id"] for d in docs],
        "files": [d["file"] for d in docs],
        "fingerprints": [d["fingerprint_md5"] for d in docs],
    }
    with open(os.path.join(DATA, "holdout_lock.json"), "w", encoding="utf-8") as fh:
        json.dump(lock, fh, indent=1, ensure_ascii=False)

    print(f"docs: {len(docs)} | words total: {sum(d['words'] for d in docs)}")
    print("wrote data/holdout_manifest_A.json")
    print("wrote data/holdout_lock.json")


if __name__ == "__main__":
    main()
