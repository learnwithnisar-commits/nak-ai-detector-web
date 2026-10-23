#!/usr/bin/env python3
"""Harvest Pakistani university theses/papers via PASTIC DSpace OAI-PMH.
Source labeled as 'university' with institution name in title.
Resumable. Run in background.
"""
import json
import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

PROJ = Path("/home/hatch/workspace/ai-detector")
OUT = PROJ / "data" / "oa_corpus" / "pastic"
OUT.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT / "progress.json"
RECORDS = OUT / "records.jsonl"
OAI = "https://repository.pastic.gov.pk/oai/request"

NS = {"oai": "http://www.openarchives.org/OAI/2.0/",
      "dc": "http://purl.org/dc/elements/1.1/"}

def fetch(params):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{OAI}?{q}",
        headers={"User-Agent": "NAKDetector/1.0 (research harvest)"})
    return ET.fromstring(urllib.request.urlopen(req, timeout=60).read())

def load_progress():
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {"token": None, "done": 0, "first": True}

def save_progress(p):
    PROGRESS.write_text(json.dumps(p))

def main():
    p = load_progress()
    token = p.get("token")
    out_f = open(RECORDS, "a", encoding="utf-8")
    while True:
        try:
            if token:
                root = fetch({"verb": "ListRecords", "resumptionToken": token})
            else:
                root = fetch({"verb": "ListRecords", "metadataPrefix": "oai_dc"})
        except Exception as e:
            print(f"OAI error: {e}, retry 60s", flush=True)
            time.sleep(60)
            continue
        recs = root.findall(".//oai:record", NS)
        if not recs:
            print("no records, done", flush=True)
            break
        for r in recs:
            header = r.find("oai:header", NS)
            if header is not None and header.get("status") == "deleted":
                continue
            md = r.find("oai:metadata", NS)
            if md is None:
                continue
            def gettext(tag):
                el = md.find(f".//dc:{tag}", NS)
                return el.text.strip() if el is not None and el.text else ""
            title = gettext("title")
            creator = gettext("creator")
            # PDF link
            pdf_url = ""
            for ident in md.findall(".//dc:identifier", NS):
                t = (ident.text or "").strip()
                if t.lower().endswith(".pdf") or "/bitstream/" in t:
                    pdf_url = t
                    break
            # institution from publisher or source
            publisher = gettext("publisher") or gettext("source")
            rec = {"title": title, "creator": creator, "publisher": publisher,
                   "pdf_url": pdf_url,
                   "date": gettext("date")}
            out_f.write(json.dumps(rec) + "\n")
            p["done"] += 1
        rt = root.find(".//oai:resumptionToken", NS)
        token = rt.text.strip() if rt is not None and rt.text else None
        p["token"] = token
        save_progress(p)
        print(f"harvested {p['done']} records...", flush=True)
        if not token:
            break
        time.sleep(1)
    out_f.close()
    save_progress(p)
    print(f"DONE: {p['done']} PASTIC records", flush=True)

if __name__ == "__main__":
    main()
