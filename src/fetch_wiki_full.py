#!/usr/bin/env python3
"""Fetch ~150 full Wikipedia article extracts (1 call/article - API limits
whole-article extracts to 1/request). 200-400 word samples for classifier
augmentation. Split: 100 train / 50 held-out eval.
"""
import json
import os
import random
import re
import subprocess

ID_API = ("https://en.wikipedia.org/w/api.php?action=query&generator=random"
          "&grnnamespace=0&grnlimit=50&prop=info&format=json")
EXT_API = ("https://en.wikipedia.org/w/api.php?action=query&pageids={pid}"
           "&prop=extracts&explaintext&exsectionformat=plain&format=json")
OUT_TRAIN = os.path.expanduser("~/workspace/ai-detector/data/wiki_full_train.jsonl")
OUT_EVAL = os.path.expanduser("~/workspace/ai-detector/data/wiki_full_eval.jsonl")
SEED = 20261012
TARGET = 150

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def get(url):
    r = subprocess.run(["curl", "-sS", "--max-time", "60", url],
                       capture_output=True, text=True)
    r.check_returncode()
    return json.loads(r.stdout)


def chunk_200_400(text):
    text = re.sub(r"\n+", " ", text).strip()
    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    words, out = 0, []
    for s in sents:
        w = len(s.split())
        if w < 4:
            continue
        if words + w > 400 and words >= 200:
            break
        out.append(s)
        words += w
    t = " ".join(out)
    return t if 200 <= len(t.split()) <= 400 else ""


def main():
    rng = random.Random(SEED)
    pageids = []
    for i in range(4):
        d = get(ID_API)
        for p in ((d.get("query") or {}).get("pages") or {}).values():
            pageids.append(p["pageid"])
    print(f"collected {len(pageids)} pageids", flush=True)
    rng.shuffle(pageids)
    samples, seen = [], set()
    for j, pid in enumerate(pageids):
        if len(samples) >= TARGET:
            break
        try:
            d = get(EXT_API.format(pid=pid))
            pg = list(((d.get("query") or {}).get("pages") or {}).values())[0]
            ext = (pg.get("extract") or "").strip()
        except Exception as e:
            print(f"  [{j}] pid {pid} failed: {e}", flush=True)
            continue
        if len(ext.split()) < 220 or ext in seen:
            continue
        seen.add(ext)
        t = chunk_200_400(ext)
        if t:
            samples.append(t)
        if (j + 1) % 25 == 0:
            print(f"  [{j+1}] samples={len(samples)}", flush=True)
    print(f"total samples: {len(samples)}", flush=True)
    rng.shuffle(samples)
    train_s, eval_s = samples[:100], samples[100:150]
    for path, subset in ((OUT_TRAIN, train_s), (OUT_EVAL, eval_s)):
        with open(path, "w", encoding="utf-8") as f:
            for t in subset:
                f.write(json.dumps({"text": t, "label": 0,
                                    "genre": "wikipedia"}) + "\n")
    print(f"TRAIN {len(train_s)} -> {OUT_TRAIN}")
    print(f"EVAL {len(eval_s)} -> {OUT_EVAL}")


if __name__ == "__main__":
    main()
