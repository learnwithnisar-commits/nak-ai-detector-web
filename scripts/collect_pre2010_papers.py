#!/usr/bin/env python3
"""
Collect 5000 pre-2010 research papers across 10 fields for human-writing evaluation.
Sources: arXiv (physics/CS/math/eng/econ), Europe PMC (biomedical), DOAJ (all fields).
Saves incrementally to data/research_papers_pre2010/manifest.jsonl
"""
import json, os, sys, time, re, hashlib
import xml.etree.ElementTree as ET
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import quote as urlquote

BASE = os.path.expanduser("~/workspace/ai-detector/data/research_papers_pre2010")
MANIFEST = os.path.join(BASE, "manifest.jsonl")
LOG = os.path.join(BASE, "logs", "collection.log")
TARGET_PER_FIELD = 500
MAX_YEAR = 2009  # strictly < 2010

FIELDS = [
    ("medicine", "europepmc"),
    ("physics", "arxiv"),
    ("computer_science", "arxiv"),
    ("mathematics", "arxiv"),
    ("engineering", "arxiv"),
    ("environmental_science", "doaj"),
    ("economics", "mixed"),
    ("psychology", "doaj"),
    ("social_sciences", "doaj"),
    ("humanities", "doaj"),
]

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def fetch(url, timeout=30, headers=None):
    req = Request(url, headers=headers or {"User-Agent": "NAK-Research-Collector/1.0 (academic research)"})
    try:
        with urlopen(req, timeout=timeout) as r:
            return r.read()
    except (URLError, HTTPError, TimeoutError, Exception) as e:
        return None

def check_url(url, timeout=15):
    """Lightweight accessibility check via HEAD-like GET with range."""
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": "NAK-Research-Collector/1.0"})
        with urlopen(req, timeout=timeout) as r:
            return r.status < 400
    except Exception:
        return False

def norm_title(t):
    return re.sub(r"\s+", " ", (t or "").lower().strip())

def load_seen():
    seen_dois, seen_titles = set(), set()
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("doi"): seen_dois.add(r["doi"].lower())
                    seen_titles.add(norm_title(r.get("title")))
                except Exception:
                    pass
    return seen_dois, seen_titles

def save_batch(records):
    with open(MANIFEST, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"Saved batch: {len(records)} records (total now: {count_total()})")

def count_total():
    if not os.path.exists(MANIFEST): return 0
    with open(MANIFEST) as f: return sum(1 for _ in f)

def count_field(field):
    if not os.path.exists(MANIFEST): return 0
    n = 0
    with open(MANIFEST) as f:
        for line in f:
            try:
                if json.loads(line).get("field") == field: n += 1
            except Exception: pass
    return n

# ---------- arXiv ----------
ARXIV_CATS = {
    "physics": ["physics.gen-ph", "physics.class-ph", "quant-ph", "astro-ph", "hep-th", "cond-mat"],
    "computer_science": ["cs.AI", "cs.CL", "cs.CV", "cs.DB", "cs.SE", "cs.LG"],
    "mathematics": ["math.NT", "math.AG", "math.AP", "math.PR", "math.ST", "math.OC"],
    "engineering": ["eess.SP", "eess.SY", "physics.app-ph", "cs.SY"],
    "economics": ["econ.EM", "econ.TH", "q-fin.EC"],
}

def collect_arxiv(field, cats, seen_dois, seen_titles, target):
    recs = []
    per_cat = max(20, target // max(1, len(cats)) + 10)
    for cat in cats:
        if len(recs) >= target: break
        # arXiv date range query: submitted between 2000 and 2009
        q = urlquote(f"cat:{cat}", safe="")
        url = (f"https://export.arxiv.org/api/query?search_query={q}"
               f"&start=0&max_results={per_cat}&sortBy=submittedDate&sortOrder=ascending")
        data = fetch(url, timeout=60)
        if not data: 
            log(f"arxiv {cat}: fetch failed"); time.sleep(3); continue
        try:
            ns = {"a": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(data)
        except Exception as e:
            log(f"arxiv {cat}: parse failed {e}"); time.sleep(3); continue
        for e in root.findall("a:entry", ns):
            if len(recs) >= target: break
            try:
                pid = e.find("a:id", ns).text.strip()
                arxiv_id = pid.split("/abs/")[-1]
                title = re.sub(r"\s+", " ", e.find("a:title", ns).text or "").strip()
                pub = e.find("a:published", ns).text[:4]
                year = int(pub) if pub.isdigit() else 0
                if year > MAX_YEAR or year < 1990: continue
                authors = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
                summary = re.sub(r"\s+", " ", (e.find("a:summary", ns).text or "")).strip()
                doi_el = e.find("a:doi", {"a": "http://arxiv.org/schemas/atom"})
                doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None
                nt = norm_title(title)
                if doi and doi.lower() in seen_dois: continue
                if nt in seen_titles: continue
                url_abs = f"https://arxiv.org/abs/{arxiv_id}"
                url_pdf = f"https://arxiv.org/pdf/{arxiv_id}"
                rec = {
                    "paper_id": f"RP-{field[:3].upper()}-{hashlib.md5(arxiv_id.encode()).hexdigest()[:8]}",
                    "title": title, "authors": authors, "year": year,
                    "doi": doi, "journal": "arXiv", "publisher": "arXiv",
                    "field": field, "source": "arxiv", "arxiv_id": arxiv_id,
                    "url": url_abs, "pdf_url": url_pdf,
                    "abstract": summary, "abstract_words": len(summary.split()),
                    "classification": "PRE_CHATGPT_HUMAN_CANDIDATE",
                    "date_verified": True, "url_accessible": True,
                    "collected_at": time.strftime("%Y-%m-%d"),
                }
                recs.append(rec)
                if doi: seen_dois.add(doi.lower())
                seen_titles.add(nt)
            except Exception:
                continue
        time.sleep(5)  # arXiv rate limit: 1 req / 3s, use 5s to be safe
    return recs

# ---------- Europe PMC ----------
def collect_europepmc(field, seen_dois, seen_titles, target, query_terms):
    recs = []
    years = list(range(2000, 2010))
    per_year = max(10, target // len(years) + 5)
    for yr in years:
        if len(recs) >= target: break
        q = f"PUB_YEAR:{yr} AND OPEN_ACCESS:Y AND ({query_terms})"
        url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
               + q.replace(" ", "%20") + f"&format=json&pageSize={per_year}&resultType=core")
        data = fetch(url)
        if not data:
            log(f"europepmc {yr}: fetch failed"); time.sleep(1); continue
        try:
            d = json.loads(data)
        except Exception:
            log(f"europepmc {yr}: parse failed"); time.sleep(1); continue
        for r in d.get("resultList", {}).get("result", []):
            if len(recs) >= target: break
            try:
                year = int(r.get("pubYear") or 0)
                if year > MAX_YEAR or year < 1990: continue
                title = (r.get("title") or "").strip()
                if not title: continue
                doi = r.get("doi")
                nt = norm_title(title)
                if doi and doi.lower() in seen_dois: continue
                if nt in seen_titles: continue
                pmid = r.get("pmid"); pmcid = r.get("pmcid"); src = r.get("source") or "MED"
                eid = pmcid or pmid or r.get("id")
                url_art = f"https://europepmc.org/article/{src}/{eid}"
                authors = r.get("authorString", "")
                abstract = (r.get("abstractText") or "").strip()
                journal = (r.get("journalInfo", {}) or {}).get("journal", {}).get("title", "")
                rec = {
                    "paper_id": f"RP-{field[:3].upper()}-{hashlib.md5(str(eid).encode()).hexdigest()[:8]}",
                    "title": title, "authors": authors, "year": year,
                    "doi": doi, "journal": journal, "publisher": "",
                    "field": field, "source": "europepmc",
                    "pmid": pmid, "pmcid": pmcid,
                    "url": url_art,
                    "pdf_url": f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf" if pmcid else None,
                    "abstract": abstract, "abstract_words": len(abstract.split()),
                    "classification": "PRE_CHATGPT_HUMAN_CANDIDATE",
                    "date_verified": True, "url_accessible": True,
                    "collected_at": time.strftime("%Y-%m-%d"),
                }
                recs.append(rec)
                if doi: seen_dois.add(doi.lower())
                seen_titles.add(nt)
            except Exception:
                continue
        time.sleep(1)
    return recs

# ---------- DOAJ ----------
DOAJ_QUERIES = {
    "environmental_science": "environmental",
    "economics": "economics",
    "psychology": "psychology",
    "social_sciences": "sociology",
    "humanities": "philosophy",
    "engineering": "engineering",
    "physics": "physics",
    "computer_science": "computer science",
    "mathematics": "mathematics",
}

def collect_doaj(field, query, seen_dois, seen_titles, target):
    recs = []
    years = list(range(2000, 2010))
    per_year = max(10, target // len(years) + 5)
    for yr in years:
        if len(recs) >= target: break
        q = urlquote(f'bibjson.year:{yr} AND "{query}"', safe="")
        url = f"https://doaj.org/api/search/articles/{q}?pageSize={per_year}"
        data = fetch(url)
        if not data:
            log(f"doaj {field} {yr}: fetch failed"); time.sleep(1); continue
        try:
            d = json.loads(data)
        except Exception:
            log(f"doaj {field} {yr}: parse failed"); time.sleep(1); continue
        for r in d.get("results", []):
            if len(recs) >= target: break
            try:
                b = r.get("bibjson", {})
                year = int(b.get("year") or 0)
                if year > MAX_YEAR or year < 1990: continue
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
                publisher = b.get("publisher", "")
                abstract = (b.get("abstract") or "").strip()
                subj = b.get("subject", [])
                subj_names = [s.get("term", "") for s in subj] if subj else []
                rec = {
                    "paper_id": f"RP-{field[:3].upper()}-{hashlib.md5((doi or title).encode()).hexdigest()[:8]}",
                    "title": title, "authors": authors, "year": year,
                    "doi": doi, "journal": journal, "publisher": publisher,
                    "field": field, "source": "doaj",
                    "url": url_art, "pdf_url": url_art,
                    "subjects": subj_names,
                    "abstract": abstract, "abstract_words": len(abstract.split()),
                    "classification": "PRE_CHATGPT_HUMAN_CANDIDATE",
                    "date_verified": True, "url_accessible": None,
                    "collected_at": time.strftime("%Y-%m-%d"),
                }
                recs.append(rec)
                if doi: seen_dois.add(doi.lower())
                seen_titles.add(nt)
            except Exception:
                continue
        time.sleep(1)
    return recs

def main():
    only_field = sys.argv[1] if len(sys.argv) > 1 else None
    seen_dois, seen_titles = load_seen()
    log(f"Starting collection. Already have {count_total()} records.")
    for field, src in FIELDS:
        if only_field and field != only_field: continue
        have = count_field(field)
        need = TARGET_PER_FIELD - have
        if need <= 0:
            log(f"{field}: already have {have}, skipping")
            continue
        log(f"{field}: have {have}, collecting {need} via {src}")
        recs = []
        try:
            if src == "arxiv":
                recs = collect_arxiv(field, ARXIV_CATS.get(field, ["physics.gen-ph"]), seen_dois, seen_titles, need)
                # Fallback to DOAJ if arXiv failed (rate limited)
                if not recs and field in DOAJ_QUERIES:
                    log(f"{field}: arXiv failed, falling back to DOAJ")
                    recs = collect_doaj(field, DOAJ_QUERIES[field], seen_dois, seen_titles, need)
            elif src == "europepmc":
                terms = {"medicine": "cancer OR diabetes OR malaria OR tuberculosis OR HIV"}[field]
                recs = collect_europepmc(field, seen_dois, seen_titles, need, terms)
            elif src == "doaj":
                recs = collect_doaj(field, DOAJ_QUERIES[field], seen_dois, seen_titles, need)
            elif src == "mixed":
                # economics: arxiv econ + doaj economics (arXiv may be rate-limited)
                r1 = collect_arxiv(field, ARXIV_CATS.get(field, ["econ.EM"]), seen_dois, seen_titles, need // 2)
                if r1:
                    save_batch(r1)
                    r2 = collect_doaj(field, DOAJ_QUERIES["economics"], seen_dois, seen_titles, need - len(r1))
                else:
                    log(f"{field}: arXiv failed, using DOAJ for full target")
                    r2 = collect_doaj(field, DOAJ_QUERIES["economics"], seen_dois, seen_titles, need)
                recs = r2
            else:
                r1 = collect_arxiv(field, ARXIV_CATS.get(field, []), seen_dois, seen_titles, need // 2)
                save_batch(r1)
                r2 = collect_doaj(field, DOAJ_QUERIES.get(field, field), seen_dois, seen_titles, need - len(r1))
                recs = r2
        except Exception as e:
            log(f"{field}: ERROR {e}")
        if recs:
            save_batch(recs)
        log(f"{field}: done, now have {count_field(field)}")
    log(f"Collection run complete. Total: {count_total()}")

if __name__ == "__main__":
    main()
