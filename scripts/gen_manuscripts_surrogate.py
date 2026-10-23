#!/usr/bin/env python3
"""Generate full AI manuscripts via DeepSeek using the skill's surrogate auth.

Why this exists: scripts/gen_deepseek.py --mode manuscripts reads a raw
DEEPSEEK_API_KEY env var, but the deepseek skill forbids raw keys in env vars
("Never ... set a secret environment variable"). This wrapper reuses
gen_deepseek's MANUSCRIPT_TOPICS / gen_manuscript logic while routing every
API call through ~/workspace/skills/deepseek/bin/deepseek_chat.py's
surrogate mechanism (custom.deepseek credential; raw key never touched).

Resumable: skips topics already present in data/family_deepseek_train.jsonl
(matched by exact topic title). Appends in the canonical schema:
text/family/register/manuscript-{field}/label/topic/words.

Usage: python3 scripts/gen_manuscripts_surrogate.py [N]   (default 16)
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                      # gen_deepseek
sys.path.insert(0, "/home/hatch/workspace/skills/deepseek/bin")  # deepseek_chat

import gen_deepseek
from deepseek_chat import chat as ds_chat


def _api_call(client_key, system, user, max_tokens, temperature=0.9):
    # client_key ignored on purpose: auth comes from the stored credential.
    return ds_chat(system, user, temperature=temperature,
                   max_tokens=max_tokens, timeout=180)


gen_deepseek._api_call = _api_call

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "family_deepseek_train.jsonl")


def existing_topics():
    have = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("topic"):
                have.add(o["topic"])
    return have


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    have = existing_topics()
    targets = [t for t in gen_deepseek.MANUSCRIPT_TOPICS if t[1] not in have][:n]
    print(f"{len(have)} topics already present; generating {len(targets)} new",
          flush=True)
    done, failed = 0, []
    with open(OUT, "a", encoding="utf-8") as f:
        for i, (field, title, genre) in enumerate(targets):
            print(f"[ms {i+1}/{len(targets)}] {field} | {title}", flush=True)
            try:
                full = gen_deepseek.gen_manuscript(None, title, genre, field)
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
                failed.append(title)
                continue
            words = len(full.split())
            f.write(json.dumps({
                "text": full, "family": "deepseek",
                "register": f"manuscript-{field}", "label": 1,
                "topic": title, "words": words,
            }) + "\n")
            f.flush()
            done += 1
            print(f"    done ({words} words)", flush=True)
    print(f"wrote {done} manuscripts to {OUT}"
          + (f"; failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
