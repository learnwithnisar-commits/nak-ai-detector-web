#!/usr/bin/env python3
"""Top up partial fields with alternative DOAJ queries."""
import json, os, sys, time, re, hashlib
from urllib.request import Request, urlopen
from urllib.parse import quote as urlquote

BASE = os.path.expanduser("~/workspace/ai-detector/data/research_papers_pre2010")
MANIFEST = os.path.join(BASE, "manifest.jsonl")

# field -> list of alternative queries to try
TOPUP_QUERIES = {
    "humanities": ["literature", "history", "linguistics", "arts", "classics"],
    "mathematics": ["algebra", "geometry", "statistics", "applied mathematics"],
    "engineering": ["mechanical engineering", "electrical engineering", "civil engineering"],
    "economics": ["finance", "business", "management"],
}

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def fetch(url, timeout=30):
    req = Request(url, headers={"User-Agent": "NAK-Research-Collector/1.0"})
    try:
        with urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None

def norm_title(t):
    return re.sub(r"\s+", " ", (t or "").lower().strip())

def load_seen():
    seen_dois, seen_titles = set(), set()
    with open(MANIFEST) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("doi"): seen_dois.add(r["doi"].lower())
                seen_titles.add(norm_title(r.get("title")))
            except Exception: pass
    return seen_dois, seen_titles

def count_field(field):
    n = 0
    with open(MANIFEST) as f:
        for line in f:
            try:
                if json.loads(line).get("field") == field: n += 1
            except Exception: pass
    return n

def collect_query(field, query, seen_dois, seen_titles, need):
    recs = []
    for yr in range(2000, 2010):
        if len(recs) >= need: break
        q = urlquote(f'bibjson.year:{yr} AND "{query}"', safe="")
        url = f"https://doaj.org/api/search/articles/{q}?pageSize=50"
        data = fetch(url)
        if not data:
            time.sleep(1); continue
        try: d = json.loads(data)
        except Exception:
            time.sleep(1); continue
        for r in d.get("results", []):
            if len(recs) >= need: break
            try:
                b = r.get("bibjson", {})
                year = int(b.get("year") or 0)
                if year > 2009 or year < 1990: continue
                title = (b.get("title") or "").strip()
                if not title: continue
                ids = b.get("identifier", [])
                doi = next((i["id"] for i in ids if i.get("type") == "doi"), None)
                nt = norm_title(title)
                if doi and doi.lower() in seen_dois: continue
                if nt in seen_titles: continue
                authors = [a.get("name", "") for a in b.get("author", [])]
                links = [l.get("url") for l in b.get("link", []) if l.get("url")]
                url_art = links[0] if links else None
                if not url_art: continue
                journal = (b.get("journal", {}) or {}).get("title", "")
                abstract = (b.get("abstract") or "").strip()
                rec = {
                    "paper_id": f"RP-{field[:3].upper()}-{hashlib.md5((doi or title).encode()).hexdigest()[:8]}",
                    "title": title, "authors": authors, "year": year,
                    "doi": doi, "journal": journal, "publisher": b.get("publisher", ""),
                    "field": field, "source": "doaj",
                    "url": url_art, "pdf_url": url_art,
                    "abstract": abstract, "abstract_words": len(abstract.split()),
                    "classification": "PRE_CHATGPT_HUMAN_CANDIDATE",
                    "date_verified": True, "url_accessible": None,
                    "collected_at": time.strftime("%Y-%m-%d"),
                }
                recs.append(rec)
                if doi: seen_dois.add(doi.lower())
                seen_titles.add(nt)
            except Exception: continue
        time.sleep(1)
    return recs

def main():
    seen_dois, seen_titles = load_seen()
    for field, queries in TOPUP_QUERIES.items():
        have = count_field(field)
        need = 500 - have
        if need <= 0:
            log(f"{field}: already {have}, skipping"); continue
        log(f"{field}: have {have}, need {need}")
        all_recs = []
        for q in queries:
            if len(all_recs) >= need: break
            recs = collect_query(field, q, seen_dois, seen_titles, need - len(all_recs))
            all_recs.extend(recs)
            log(f"  query '{q}': +{len(recs)}")
        if all_recs:
            with open(MANIFEST, "a") as f:
                for r in all_recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log(f"{field}: saved {len(all_recs)}, now have {count_field(field)}")

if __name__ == "__main__":
    main()
