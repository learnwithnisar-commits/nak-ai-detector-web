#!/usr/bin/env python3
"""FAIL-CLOSED training guard: assert no locked holdout doc appears in training data.

Usage (from any training script, BEFORE training):
    from assert_no_holdout_leak import assert_no_leak
    assert_no_leak(list_of_training_texts)   # raises SystemExit on leak

Or standalone:
    python3 scripts/assert_no_holdout_leak.py --check data/some_train.jsonl

Reads data/holdout_lock.json (fingerprints + files). Any match -> exit(1)
with a loud error. Never silently pass.
"""
import hashlib
import json
import os
import re
import sys

PROJ = os.path.expanduser("~/workspace/ai-detector")
LOCK = os.path.join(PROJ, "data", "holdout_lock.json")


def fingerprint(text):
    norm = re.sub(r"\s+", " ", text[:2000]).strip().lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def load_lock():
    with open(LOCK, encoding="utf-8") as fh:
        return json.load(fh)


def find_leaks(training_texts):
    """Return list of (train_idx, matched_doc_id) for any holdout fingerprint
    found in the training texts."""
    lock = load_lock()
    locked_fps = dict(zip(lock["fingerprints"], lock["doc_ids"]))
    locked_files = set(lock["files"])
    leaks = []
    for i, t in enumerate(training_texts):
        fp = fingerprint(t)
        if fp in locked_fps:
            leaks.append((i, locked_fps[fp], "fingerprint"))
    return leaks


def assert_no_leak(training_texts, context=""):
    leaks = find_leaks(training_texts)
    if leaks:
        msg = (f"FATAL: holdout leak detected {context}! "
               f"{len(leaks)} training text(s) match locked holdout docs: "
               f"{leaks[:5]}. Training ABORTED. Remove leaked docs or "
               f"unlock/rebuild the holdout with Nisar's approval.")
        print(msg, file=sys.stderr)
        raise SystemExit(1)
    print(f"holdout leak check PASSED {context}: {len(training_texts)} texts, 0 leaks.")


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "--check":
        print(__doc__)
        sys.exit(2)
    path = sys.argv[2]
    texts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                texts.append(rec.get("text", ""))
            except json.JSONDecodeError:
                continue
    assert_no_leak(texts, context=f"file={path}")


if __name__ == "__main__":
    main()
