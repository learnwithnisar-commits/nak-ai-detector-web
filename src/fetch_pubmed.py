#!/usr/bin/env python3
"""Fetch PubMed abstracts (human-written academic biomedical prose) via
E-utilities (no key needed). Domain: public health / lifestyle, close to the
demo proposal's register.

Output: data/pubmed_train.jsonl (250, label 0, classifier+LM training),
        data/pubmed_eval.jsonl  (50 held-out, extra academic FPR probe).
Samples are 120-350 words.
"""
import json
import os
import random
import re
import subprocess
import xml.etree.ElementTree as ET

ESEARCH = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
           "?db=pubmed&term={q}&retmax=150&retmode=json&sort=relevance")
EFETCH = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
          "?db=pubmed&id={ids}&retmode=xml")
QUERIES = ["public+health+lifestyle+risk+behaviour",
           "epidemiology+chronic+disease+prevention",
           "biomedical+research+methods+cohort+study"]
OUT_TRAIN = os.path.expanduser("~/workspace/ai-detector/data/pubmed_train.jsonl")
OUT_EVAL = os.path.expanduser("~/workspace/ai-detector/data/pubmed_eval.jsonl")
SEED = 20261012


def get(url):
    r = subprocess.run(["curl", "-sS", "--max-time", "90", url],
                       capture_output=True, text=True)
    r.check_returncode()
    return r.stdout


def main():
    rng = random.Random(SEED)
    ids = []
    for q in QUERIES:
        d = json.loads(get(ESEARCH.format(q=q)))
        ids += d["esearchresult"]["idlist"]
    ids = list(dict.fromkeys(ids))
    print(f"pubmed ids: {len(ids)}", flush=True)
    rng.shuffle(ids)

    samples, seen = [], set()
    for i in range(0, len(ids), 60):
        batch = ids[i:i + 60]
        try:
            xml = get(EFETCH.format(ids=",".join(batch)))
            root = ET.fromstring(xml)
        except Exception as e:
            print(f"  batch {i}: FAILED {e}", flush=True)
            continue
        for art in root.iter("PubmedArticle"):
            parts = []
            for ab in art.iter("AbstractText"):
                label = ab.get("Label")
                t = " ".join(ab.itertext()).strip()
                parts.append(f"{label}: {t}" if label else t)
            text = re.sub(r"\s+", " ", " ".join(parts)).strip()
            w = len(text.split())
            if 120 <= w <= 350 and text not in seen:
                seen.add(text)
                samples.append(text)
        print(f"  batch {i}: samples={len(samples)}", flush=True)
        if len(samples) >= 320:
            break
    print(f"total: {len(samples)}", flush=True)
    rng.shuffle(samples)
    train_s, eval_s = samples[:250], samples[250:300]
    for path, subset in ((OUT_TRAIN, train_s), (OUT_EVAL, eval_s)):
        with open(path, "w", encoding="utf-8") as f:
            for t in subset:
                f.write(json.dumps({"text": t, "label": 0,
                                    "genre": "academic_abstract"}) + "\n")
    print(f"TRAIN {len(train_s)} -> {OUT_TRAIN}")
    print(f"EVAL {len(eval_s)} -> {OUT_EVAL}")


if __name__ == "__main__":
    main()
