#!/usr/bin/env python3
"""Generate academic-register AI samples via Kimi/Moonshot API (Nisar's key).

Needs: KIMI_API_KEY env (from platform.moonshot.ai or moonshot.cn, Nisar's account).
Model: kimi-k2 (or moonshot-v1-8k). Cost: a few cents per 100 samples.

Usage: KIMI_API_KEY=sk-... .venv/bin/python scripts/gen_kimi.py --n 100
Output: data/family_kimi_train.jsonl  (schema: text/family/register/label)
"""
import argparse, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TOPICS = [
    ("academic-ml", "contrastive learning for job-resume matching"),
    ("academic-ml", "Bayesian networks for personnel selection"),
    ("academic-ml", "multilingual sentence embeddings in hiring"),
    ("academic-ml", "uncertainty quantification in neural rankers"),
    ("academic-bio", "antimicrobial resistance surveillance in Salmonella"),
    ("academic-bio", "CRISPR diagnostics for field pathogen detection"),
    ("academic-business", "supply chain resilience modeling"),
    ("academic-business", "CBDC adoption in developing economies"),
    ("academic-humanities", "digital public sphere and affective publics"),
    ("academic-humanities", "oral history methodology in Partition studies"),
]

SYSTEM = ("You are writing a single academic paper paragraph (180-280 words). "
          "Formal scholarly tone, no headings, no bullet points, one paragraph only.")

def gen(client_key, topic):
    req = urllib.request.Request(
        "https://api.moonshot.ai/v1/chat/completions",
        data=json.dumps({
            "model": "kimi-k2",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Write the paragraph about: {topic}."},
            ],
            "temperature": 0.9, "max_tokens": 500,
        }).encode(),
        headers={"Authorization": f"Bearer {client_key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    key = os.environ.get("KIMI_API_KEY")
    if not key:
        sys.exit("KIMI_API_KEY not set")
    out = os.path.join(ROOT, "data", "family_kimi_train.jsonl")
    done = sum(1 for _ in open(out)) if os.path.exists(out) else 0
    with open(out, "a", encoding="utf-8") as f:
        for i in range(done, a.n):
            reg, topic = TOPICS[i % len(TOPICS)]
            try:
                t = gen(key, topic)
            except Exception as e:
                print(f"[{i}] error: {e}; retry in 20s", flush=True)
                time.sleep(20)
                continue
            f.write(json.dumps({"text": t, "family": "kimi",
                                "register": reg, "label": 1}) + "\n")
            f.flush()
            print(f"[{i+1}/{a.n}] {reg}", flush=True)
            time.sleep(2)
    print("wrote", out)

if __name__ == "__main__":
    main()
