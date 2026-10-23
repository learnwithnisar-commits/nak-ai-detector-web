#!/usr/bin/env python3
"""Fetch random Wikipedia intros (exintro batches 20/call) for the FPR gate.

Two outputs:
  data/wiki_lm.txt        - ALL intro texts, one per line (LM corpus material)
  data/wiki_human_eval.jsonl - 150 held-out intros >=40 words (FPR gate only)
"""
import json
import os
import random
import re
import subprocess

API = ("https://en.wikipedia.org/w/api.php?action=query&generator=random"
       "&grnnamespace=0&grnlimit=20&prop=extracts&explaintext&exintro&format=json")
N_CALLS = 12
OUT_LM = os.path.expanduser("~/workspace/ai-detector/data/wiki_lm.txt")
OUT_EVAL = os.path.expanduser("~/workspace/ai-detector/data/wiki_human_eval.jsonl")
SEED = 20261012


def fetch_once():
    r = subprocess.run(["curl", "-sS", "--max-time", "60", API],
                       capture_output=True, text=True)
    r.check_returncode()
    return json.loads(r.stdout)


def main():
    rng = random.Random(SEED)
    seen, intros = set(), []
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
            if len(ext.split()) >= 30:
                intros.append(ext)
                got += 1
        print(f"call {i}: {got} intros (total {len(intros)})", flush=True)
    with open(OUT_LM, "w", encoding="utf-8") as f:
        for t in intros:
            f.write(t.replace("\n", " ") + "\n")
    rng.shuffle(intros)
    eval_s = [t for t in intros if len(t.split()) >= 40][:150]
    with open(OUT_EVAL, "w", encoding="utf-8") as f:
        for t in eval_s:
            f.write(json.dumps({"text": t, "label": 0,
                                "genre": "wikipedia"}) + "\n")
    toks = sum(len(t.split()) for t in intros)
    print(f"LM intros: {len(intros)} ({toks} tokens) -> {OUT_LM}")
    print(f"EVAL intros: {len(eval_s)} -> {OUT_EVAL}")


if __name__ == "__main__":
    main()
