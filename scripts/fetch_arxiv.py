#!/usr/bin/env python3
"""Fetch real human academic abstracts from the arXiv API (legal, open).

Uses curl via subprocess (Python urllib gets 400s through this egress
proxy; curl works fine). Polite: 3s delay between requests per arXiv terms.
Stores: data/real_human/arxiv_raw.jsonl (one record per paper).
"""
import json
import os
import re
import subprocess
import time
import urllib.parse
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "data", "real_human")
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "arxiv_raw.jsonl")

NS = {"a": "http://www.w3.org/2005/Atom"}

QUERIES = [
    (["cs.CL", "cs.AI", "cs.LG"], 700),
    (["stat.ML", "math.ST", "math.PR"], 500),
    (["astro-ph", "quant-ph"], 400),
    (["q-bio.QM", "q-bio.GN"], 400),
    (["econ.EM", "cs.CY"], 400),
    (["cs.CV", "cs.RO"], 400),
]

def clean(t):
    t = t.replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def curl_get(url):
    out = subprocess.run(
        ["curl", "-s", "-m", "60", "-A", "NAK-AI-Detector research use", url],
        capture_output=True, text=False, timeout=70,
    )
    if out.returncode != 0:
        raise RuntimeError("curl rc=%d" % out.returncode)
    return out.stdout

def fetch(cats, want):
    q = " OR ".join("cat:%s" % c for c in cats)
    params = {
        "search_query": q, "start": 0, "max_results": want,
        "sortBy": "submittedDate", "sortOrder": "descending",
    }
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    data = curl_get(url)
    if b"<entry>" not in data:
        raise RuntimeError("no entries in response (%d bytes)" % len(data))
    root = ET.fromstring(data)
    recs = []
    for e in root.findall("a:entry", NS):
        id_el = e.find("a:id", NS)
        aid = id_el.text.strip().split("/abs/")[-1] if id_el is not None else ""
        title_el = e.find("a:title", NS)
        sum_el = e.find("a:summary", NS)
        pub_el = e.find("a:published", NS)
        title = clean(title_el.text or "") if title_el is not None else ""
        abstract = clean(sum_el.text or "") if sum_el is not None else ""
        pub = (pub_el.text or "")[:10] if pub_el is not None else ""
        words = len(abstract.split())
        if aid and 80 <= words <= 400 and title:
            recs.append({
                "text": abstract, "label": 0, "genre": "academic_abstract",
                "source": "arxiv", "arxiv_id": aid, "title": title,
                "published": pub, "words": words,
            })
    return recs

def main():
    seen = set()
    total = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for qi, (cats, want) in enumerate(QUERIES):
            print("[%d/%d] %s ..." % (qi + 1, len(QUERIES), "+".join(cats)), flush=True)
            try:
                recs = fetch(cats, want)
            except Exception as ex:
                print("  FAILED: %s" % ex, flush=True)
                continue
            new = 0
            for r in recs:
                if r["arxiv_id"] in seen:
                    continue
                seen.add(r["arxiv_id"])
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                new += 1
            total += new
            print("  got %d, new %d (total %d)" % (len(recs), new, total), flush=True)
            if qi < len(QUERIES) - 1:
                time.sleep(3.2)
    print("DONE: %d arXiv abstracts -> %s" % (total, OUT))

if __name__ == "__main__":
    main()
