#!/usr/bin/env python3
"""Fetch additional Wikipedia intros (40-150 words) as SHORT real-human
training samples (label 0). Teaches the model that short formulaic human
text (biographical/event stubs) is human. Appends to wiki_lm.txt (LM)."""
import json
import os
import random
import re
import subprocess

API = ("https://en.wikipedia.org/w/api.php?action=query&generator=random"
       "&grnnamespace=0&grnlimit=20&prop=extracts&explaintext&exintro&format=json")
N_CALLS = 15
OUT_TRAIN = os.path.expanduser("~/workspace/ai-detector/data/wiki_short_train.jsonl")
OUT_LM = os.path.expanduser("~/workspace/ai-detector/data/wiki_lm.txt")
SEED = 20261013


def fetch_once():
    r = subprocess.run(["curl", "-sS", "--max-time", "60", API],
                       capture_output=True, text=True)
    r.check_returncode()
    return json.loads(r.stdout)


def main():
    rng = random.Random(SEED)
    seen, shorts = set(), []
    for i in range(N_CALLS):
        try:
            data = fetch_once()
        except Exception as e:
            print(f"call {i}: FAILED ({e})", flush=True)
            continue
        pages = (data.get("query") or {}).get("pages") or {}
        got = 0
        for pid, pg in pages.items():
            if pid in seen:
                continue
            seen.add(pid)
            ext = re.sub(r"\s+", " ", (pg.get("extract") or "")).strip()
            w = len(ext.split())
            if 40 <= w <= 150:
                shorts.append(ext)
                got += 1
        print(f"call {i}: {got} (total {len(shorts)})", flush=True)
    rng.shuffle(shorts)
    train_s = shorts[:220]
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for t in train_s:
            f.write(json.dumps({"text": t, "label": 0,
                                "genre": "wikipedia_short"}) + "\n")
    with open(OUT_LM, "a", encoding="utf-8") as f:
        for t in train_s:
            f.write(t.replace("\n", " ") + "\n")
    print(f"TRAIN {len(train_s)} -> {OUT_TRAIN} (+ appended to wiki_lm.txt)")


if __name__ == "__main__":
    main()
