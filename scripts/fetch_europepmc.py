#!/usr/bin/env python3
"""Fetch open-access biomedical abstracts from Europe PMC (legal, open).

Uses the public REST search API with OPEN_ACCESS:Y filter.
Polite: 1s delay between pages. No scraping, no paywalls.
Stores: data/real_human/europepmc_raw.jsonl
"""
import json
import os
import re
import time
import urllib.request
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "data", "real_human")
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "europepmc_raw.jsonl")

QUERY = 'OPEN_ACCESS:Y AND PUB_YEAR:[2023 TO 2026] AND ABSTRACT:"*"'
PAGE_SIZE = 1000
PAGES = 2  # ~2000 candidates

def clean(t):
    t = (t or "").replace("\n", " ").strip()
    return re.sub(r"\s+", " ", t)

def fetch_page(cursor):
    params = {
        "query": QUERY,
        "format": "json",
        "pageSize": PAGE_SIZE,
        "cursorMark": cursor,
        "resultType": "core",
    }
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
           + urllib.parse.urlencode(params))
    req = urllib.request.Request(url, headers={"User-Agent": "NAK-AI-Detector research data (contact: research use)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))

def main():
    seen = set()
    total = 0
    cursor = "*"
    with open(OUT, "w", encoding="utf-8") as f:
        for p in range(PAGES):
            print(f"[page {p+1}/{PAGES}] ...", flush=True)
            try:
                data = fetch_page(cursor)
            except Exception as ex:
                print(f"  FAILED: {ex}", flush=True)
                break
            results = data.get("resultList", {}).get("result", [])
            if not results:
                print("  no results, stopping", flush=True)
                break
            new = 0
            for r in results:
                pmid = r.get("pmid") or r.get("id")
                abstract = clean(r.get("abstractText"))
                title = clean(r.get("title"))
                words = len(abstract.split())
                if not pmid or pmid in seen or not (80 <= words <= 400) or not title:
                    continue
                seen.add(pmid)
                f.write(json.dumps({
                    "text": abstract, "label": 0, "genre": "academic_abstract",
                    "source": "europepmc", "pmid": pmid, "title": title,
                    "journal": clean(r.get("journalInfo", {}).get("journal", {}).get("title", "")),
                    "pubYear": r.get("pubYear"), "words": words,
                }, ensure_ascii=False) + "\n")
                new += 1
            total += new
            print(f"  new {new} (total {total})", flush=True)
            cursor = data.get("nextCursorMark")
            if not cursor:
                break
            time.sleep(1.2)
    print(f"DONE: {total} Europe PMC abstracts -> {OUT}")

if __name__ == "__main__":
    main()
