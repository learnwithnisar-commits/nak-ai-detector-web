#!/usr/bin/env python3
"""Harvest open-access journal full texts via OpenAlex (free, no key needed).
Targets: all OA works with PDF/fulltext links. Writes to data/oa_corpus/openalex/.
Resumable via cursor pagination. Run in background.
"""
import json
import os
import random
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

PROJ = Path("/home/hatch/workspace/ai-detector")
OUT = PROJ / "data" / "oa_corpus" / "famous"
OUT.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT / "progress.json"
TEXTS = OUT / "texts.jsonl"

MAILTO = "engr.nisarahmedkatyar@gmail.com"  # OpenAlex polite pool

def api(url, params=None):
    # OpenAlex now requires a free API key for cursor paging past page 1.
    # Set OPENALEX_API_KEY env var (get one at https://openalex.org/settings/api).
    params = dict(params or {})
    _key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if _key:
        params["api_key"] = _key
    q = urllib.parse.urlencode(params)
    full = f"{url}?{q}" if q else url
    req = urllib.request.Request(full, headers={"User-Agent": f"NAKDetector/1.0 (mailto:{MAILTO})"})
    return json.load(urllib.request.urlopen(req, timeout=30))

def load_progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {"cursor": "*", "done": 0, "errors": 0}

def save_progress(p):
    PROGRESS.write_text(json.dumps(p))

def main():
    p = load_progress()
    cursor = p["cursor"]
    out_f = open(TEXTS, "a", encoding="utf-8")
    seen_ids = set()
    if TEXTS.exists():
        # quick resume: count lines
        pass

    consec_fail = 0
    while True:
        try:
            data = api("https://api.openalex.org/works", {
                "filter": "is_oa:true,has_fulltext:true",
                # FAMOUS FIRST: most-cited papers = most copied by students
                "sort": "cited_by_count:desc",
                # per-page=50: smaller responses survive the egress proxy;
                # 200 caused persistent IncompleteRead truncation storms.
                "per-page": 50,
                "cursor": cursor,
                "select": "id,title,doi,primary_location,open_access,best_oa_location",
                "mailto": MAILTO,
            })
        except Exception as e:
            body = ""
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
            # OpenAlex signals end-of-results with 400 "Cursor is null" — that's DONE, not an error
            if "Cursor is null" in body or "reached end of results" in body:
                print(f"DONE: {p['done']} OA works catalogued (end of results)", flush=True)
                break
            consec_fail += 1
            wait = min(10 * (2 ** min(consec_fail - 1, 5)), 600)
            print(f"API error ({consec_fail} in a row): {e} {body} — retry in {wait}s", flush=True)
            p["errors"] += 1
            save_progress(p)
            time.sleep(wait + random.uniform(0, 5))
            continue
        consec_fail = 0

        results = data.get("results", [])
        if not results:
            print("No more results, done", flush=True)
            break

        for w in results:
            # get best OA PDF URL
            pdf_url = None
            bol = w.get("best_oa_location") or {}
            pdf_url = bol.get("pdf_url")
            if not pdf_url:
                continue
            rec = {
                "id": w["id"],
                "title": w.get("title", ""),
                "doi": w.get("doi", ""),
                "pdf_url": pdf_url,
            }
            out_f.write(json.dumps(rec) + "\n")
            p["done"] += 1

        cursor = data["meta"]["next_cursor"]
        p["cursor"] = cursor
        if p["done"] % 1000 == 0:
            save_progress(p)
            print(f"progress: {p['done']} works, cursor={cursor[:20]}...", flush=True)
        # polite rate limit (jittered, single-stream)
        time.sleep(random.uniform(0.8, 1.6))

    out_f.close()
    save_progress(p)
    print(f"DONE: {p['done']} OA works catalogued", flush=True)

if __name__ == "__main__":
    main()
