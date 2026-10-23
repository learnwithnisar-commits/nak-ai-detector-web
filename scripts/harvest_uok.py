#!/usr/bin/env python3
"""Harvest University of Karachi open-access papers via OpenAlex.
Source labeled 'university' with 'University of Karachi' in title.
Resumable. Run in background.
"""
import json
import random
import time
import urllib.request
import urllib.parse
from pathlib import Path

PROJ = Path("/home/hatch/workspace/ai-detector")
OUT = PROJ / "data" / "oa_corpus" / "uok"
OUT.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT / "progress.json"
RECORDS = OUT / "records.jsonl"
UOK_ID = "https://openalex.org/I193670377"
MAILTO = "engr.nisarahmedkatyar@gmail.com"

def api(params):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"https://api.openalex.org/works?{q}",
        headers={"User-Agent": f"NAKDetector/1.0 (mailto:{MAILTO})"})
    return json.load(urllib.request.urlopen(req, timeout=30))

def load_progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {"cursor": "*", "done": 0}

def save_progress(p):
    PROGRESS.write_text(json.dumps(p))

def main():
    p = load_progress()
    cursor = p["cursor"]
    # Load existing IDs to avoid duplicates on restart
    seen = set()
    if RECORDS.exists():
        with open(RECORDS, encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["id"])
                except:
                    pass
        print(f"Resuming with {len(seen)} existing records", flush=True)
    out_f = open(RECORDS, "a", encoding="utf-8")
    consec_fail = 0
    while True:
        try:
            data = api({
                "filter": f"authorships.institutions.id:{UOK_ID},is_oa:true",
                "per-page": 200, "cursor": cursor,
                "select": "id,title,doi,best_oa_location,authorships",
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
                print(f"DONE: {p['done']} UoK papers (end of results)", flush=True)
                break
            consec_fail += 1
            wait = min(10 * (2 ** min(consec_fail - 1, 5)), 600)
            body = ""
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
            print(f"API error ({consec_fail} in a row): {e} {body} — retry in {wait}s", flush=True)
            time.sleep(wait + random.uniform(0, 5))
            continue
        consec_fail = 0
        results = data.get("results", [])
        if not results:
            print("done", flush=True)
            break
        for w in results:
            bol = w.get("best_oa_location") or {}
            pdf_url = bol.get("pdf_url")
            if not pdf_url:
                continue
            # confirm UoK affiliation
            uok = False
            for auth in w.get("authorships", []):
                for inst in auth.get("institutions", []):
                    if inst.get("id") == UOK_ID:
                        uok = True
                        break
            if not uok:
                continue
            if w["id"] in seen:
                continue
            seen.add(w["id"])
            rec = {"id": w["id"], "title": w.get("title", ""),
                   "doi": w.get("doi", ""), "pdf_url": pdf_url,
                   "institution": "University of Karachi"}
            out_f.write(json.dumps(rec) + "\n")
            p["done"] += 1
        cursor = data["meta"]["next_cursor"]
        p["cursor"] = cursor
        if p["done"] % 500 == 0:
            save_progress(p)
            print(f"UoK: {p['done']} OA papers...", flush=True)
        time.sleep(random.uniform(0.8, 1.6))
    out_f.close()
    save_progress(p)
    print(f"DONE: {p['done']} UoK papers", flush=True)

if __name__ == "__main__":
    main()
