#!/usr/bin/env python3
"""Harvest OA papers from ALL universities worldwide via OpenAlex.
Each record tagged with its university name for source attribution.
Works through top universities by works_count. Resumable. Background.
"""
import json
import os
import time
import urllib.request
import urllib.parse
from pathlib import Path

PROJ = Path("/home/hatch/workspace/ai-detector")
OUT = PROJ / "data" / "oa_corpus" / "universities"
OUT.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT / "progress.json"
RECORDS = OUT / "records.jsonl"
UNIV_LIST = OUT / "universities.json"
MAILTO = "engr.nisarahmedkatyar@gmail.com"

def api(base, params):
    # OpenAlex now requires a free API key for cursor paging past page 1.
    # Set OPENALEX_API_KEY env var (get one at https://openalex.org/settings/api).
    params = dict(params)
    _key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if _key:
        params["api_key"] = _key
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{base}?{q}",
        headers={"User-Agent": f"NAKDetector/1.0 (mailto:{MAILTO})"})
    return json.load(urllib.request.urlopen(req, timeout=60))

def get_universities(limit=500):
    """Get top universities by works count."""
    if UNIV_LIST.exists():
        return json.loads(UNIV_LIST.read_text())
    univs = []
    cursor = "*"
    while len(univs) < limit:
        data = api("https://api.openalex.org/institutions", {
            "per-page": 200, "cursor": cursor, "sort": "works_count:desc",
            "select": "id,display_name,works_count,country_code,type",
            "mailto": MAILTO,
        })
        for inst in data.get("results", []):
            # Only universities (not labs, companies, etc.)
            itype = inst.get("type", "")
            if itype in ("education", "university") or "niversity" in inst.get("display_name", ""):
                univs.append({
                    "id": inst["id"],
                    "name": inst["display_name"],
                    "works": inst.get("works_count", 0),
                    "country": inst.get("country_code", ""),
                })
                if len(univs) >= limit:
                    break
        cursor = data["meta"].get("next_cursor")
        if not cursor or not data.get("results"):
            break
        time.sleep(0.3)
    UNIV_LIST.write_text(json.dumps(univs, indent=1))
    print(f"Got {len(univs)} universities", flush=True)
    return univs

def load_progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {"univ_idx": 0, "cursor": "*", "done": 0, "univs_done": []}

def save_progress(p):
    PROGRESS.write_text(json.dumps(p))

def main():
    univs = get_universities(500)
    p = load_progress()
    seen = set()
    if RECORDS.exists():
        with open(RECORDS, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["id"])
                except:
                    pass
        print(f"Resuming: {len(seen)} existing, univ_idx={p['univ_idx']}", flush=True)

    out_f = open(RECORDS, "a", encoding="utf-8")
    
    while p["univ_idx"] < len(univs):
        univ = univs[p["univ_idx"]]
        uid = univ["id"]
        uname = univ["name"]
        cursor = p["cursor"]
        
        print(f"[{p['univ_idx']+1}/{len(univs)}] {uname}...", flush=True)
        
        while True:
            try:
                data = api("https://api.openalex.org/works", {
                    "filter": f"authorships.institutions.id:{uid},is_oa:true,has_fulltext:true",
                    # per-page=50: smaller responses survive the egress proxy;
                    # 200 caused persistent IncompleteRead truncation storms.
                    "per-page": 50, "cursor": cursor,
                    "select": "id,title,doi,best_oa_location",
                    "mailto": MAILTO,
                })
            except Exception as e:
                print(f"  API error: {e}, retry 30s", flush=True)
                time.sleep(30)
                continue
            
            results = data.get("results", [])
            if not results:
                break
                
            for w in results:
                if w["id"] in seen:
                    continue
                bol = w.get("best_oa_location") or {}
                pdf_url = bol.get("pdf_url")
                if not pdf_url:
                    continue
                seen.add(w["id"])
                rec = {
                    "id": w["id"],
                    "title": w.get("title", ""),
                    "doi": w.get("doi", ""),
                    "pdf_url": pdf_url,
                    "institution": uname,
                    "country": univ.get("country", ""),
                }
                out_f.write(json.dumps(rec) + "\n")
                p["done"] += 1
            
            cursor = data["meta"].get("next_cursor")
            p["cursor"] = cursor if cursor else "*"
            if p["done"] % 1000 == 0:
                save_progress(p)
            if not cursor:
                break
            time.sleep(0.2)
        
        # University complete
        p["univs_done"].append(uname)
        p["univ_idx"] += 1
        p["cursor"] = "*"
        save_progress(p)
        print(f"  DONE {uname}: total {p['done']:,} records", flush=True)
    
    out_f.close()
    save_progress(p)
    print(f"ALL DONE: {p['done']:,} records from {len(p['univs_done'])} universities", flush=True)

if __name__ == "__main__":
    main()
