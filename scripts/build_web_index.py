"""Web academic plagiarism index builder for NAK AI Detector.

Collects article HTML from target domains via Common Crawl (respectful:
no origin-server hits), extracts text, winnows fingerprints (K=5, W=4,
identical to scripts/build_plag_index.py), writes fp JSONL shards.

Usage:
  python build_web_index.py --stage urls       # collect CDX records
  python build_web_index.py --stage fetch      # download+extract+fingerprint
  python build_web_index.py --stage db         # build SQLite DB

Fingerprint params MUST stay in sync with app/plag.py (K, W, norm, md5->int64).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "web_academic_index"
URLS_JSONL = OUT_DIR / "cdx_records.jsonl"
FP_JSONL = OUT_DIR / "fp_web.jsonl"
FINAL_DB = OUT_DIR / "plag_web.db"

CC_INDEX = "https://index.commoncrawl.org"
CC_DATA = "https://data.commoncrawl.org"
CRAWLS = ["CC-MAIN-2026-39", "CC-MAIN-2026-34"]

K = 5
W = 4
_PUNCT = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)
UA = {"User-Agent": "NAK-academic-index/1.0 (research plagiarism index; contact: admin)"}

# domain -> (url path regex for article pages, strip suffixes)
DOMAIN_RULES = {
    "mdpi.com": (re.compile(r"^/\d{4}-\d{3}[\dxX]/\d+/\d+/\d+$"),
                 ("/reprints", "/notes", "/xml", "/pdf", "/htm")),
    "frontiersin.org": (re.compile(r"^/articles/10\.3389/"), ()),
    "nature.com": (re.compile(r"^/articles/"), ("/metrics", "/references", "/figures")),
    "biomedcentral.com": (re.compile(r"^/articles/"), ("/tables", "/figures")),
    "sportrxiv.org": (re.compile(r"^/article/view/\d+"), ()),
    "vu.edu.au": (re.compile(r"/(handle|entities/publication)/"), ()),
    "newswise.com": (re.compile(r"^/articles/"), ()),
}


def norm_word(w: str) -> str:
    return _PUNCT.sub("", w.lower())


def _h64(s: str) -> int:
    h = int.from_bytes(hashlib.md5(s.encode("utf-8")).digest()[:8], "big")
    h &= (1 << 64) - 1
    return h - (1 << 64) if h >= (1 << 63) else h


def winnow(tokens: list[str], k: int = K, w: int = W) -> list[tuple[int, int]]:
    n = len(tokens)
    if n < k:
        return []
    hashes = [(_h64(" ".join(tokens[i:i + k])), i) for i in range(n - k + 1)]
    out, last = [], None
    for j in range(len(hashes) - w + 1):
        m = min(hashes[j:j + w], key=lambda t: (t[0], -t[1]))
        if m != last:
            out.append(m)
            last = m
    return out


def normalize_url(url: str, domain: str) -> str | None:
    try:
        p = urllib.parse.urlparse(url)
    except Exception:
        return None
    host = p.hostname or ""
    if domain not in host:
        return None
    path = p.path or "/"
    _, suffixes = DOMAIN_RULES.get(domain, (None, ()))
    for sfx in suffixes:
        if path.endswith(sfx):
            path = path[: -len(sfx)]
    path = re.sub(r"/+$", "", path) or "/"
    return f"https://{host}{path}".lower()


def article_ok(url: str, domain: str) -> bool:
    rule = DOMAIN_RULES.get(domain)
    if not rule:
        return False
    pat, _ = rule
    path = urllib.parse.urlparse(url).path or "/"
    return bool(pat.search(path))


def _cc_query(url: str, tries: int = 3) -> list[str]:
    last_e = None
    for attempt in range(tries):
        req = urllib.request.Request(url, headers=UA)
        try:
            resp = urllib.request.urlopen(req, timeout=120)
            return resp.read().decode("utf-8", "replace").splitlines()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []  # no results for this prefix: definitive, don't retry
            last_e = e
            time.sleep(4 * (attempt + 1))
        except Exception as e:
            last_e = e
            time.sleep(4 * (attempt + 1))
    raise last_e  # type: ignore[misc]


def stage_urls() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen_urls: set[str] = set()
    seen_digests: set[str] = set()
    total = 0
    with open(URLS_JSONL, "w", encoding="utf-8") as out:
        for domain in DOMAIN_RULES:
            for crawl in CRAWLS:
                # partition huge domains by first path char to keep queries cheap
                prefixes = [""]
                if domain in ("mdpi.com",):
                    prefixes = [str(d) for d in range(10)]
                for pfx in prefixes:
                    q = (f"{CC_INDEX}/{crawl}-index?url={urllib.parse.quote(domain + '/' + pfx + '*')}"
                         f"&output=json&filter=status:200&filter=mime:text/html"
                         f"&collapse=urlkey")
                    try:
                        lines = _cc_query(q)
                    except Exception as e:
                        print(f"  !! {domain} {crawl} pfx={pfx!r}: {e}", flush=True)
                        continue
                    n_new = 0
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith("{'") or line.startswith('{"message"'):
                            continue
                        try:
                            d = json.loads(line)
                        except Exception:
                            continue
                        url = d.get("url", "")
                        if not article_ok(url, domain):
                            continue
                        nurl = normalize_url(url, domain)
                        if not nurl or nurl in seen_urls:
                            continue
                        digest = d.get("digest", "")
                        if digest and digest in seen_digests:
                            continue
                        seen_urls.add(nurl)
                        if digest:
                            seen_digests.add(digest)
                        try:
                            length = int(d["length"])
                        except Exception:
                            continue
                        if length < 15000 or length > 3_000_000:
                            continue
                        out.write(json.dumps({
                            "url": nurl, "domain": domain,
                            "filename": d["filename"], "offset": int(d["offset"]),
                            "length": length, "digest": digest,
                        }) + "\n")
                        n_new += 1
                    total += n_new
                    print(f"  {domain} {crawl} pfx={pfx!r}: {len(lines)} recs -> {n_new} new urls "
                          f"(total {total})", flush=True)
                    time.sleep(2)
    print(f"saved {total} url records -> {URLS_JSONL}")



# ---------------------------------------------------------------- fetch stage

def _extract_text(html: bytes, url: str) -> tuple[str, str]:
    """Return (title, text) from article HTML. Uses lxml (installed).

    Publisher-aware: MDPI uses div.html-p, Springer/Nature use
    div.c-article-body, Frontiers uses article divs; generic <p> fallback.
    """
    from lxml import html as lhtml
    try:
        doc = lhtml.fromstring(html)
    except Exception:
        return "", ""
    # title
    title = ""
    for xp in ("//meta[@property='og:title']/@content", "//title/text()", "//h1//text()"):
        try:
            r = doc.xpath(xp)
        except Exception:
            r = []
        if r:
            title = " ".join(t.strip() for t in r if t.strip())
            if title:
                break
    title = re.sub(r"\s+", " ", title)[:300]
    # drop boilerplate
    for el in doc.xpath("//script|//style|//nav|//header|//footer|//aside|//form|"
                         "//noscript|//*[@role='navigation']|//*[@role='banner']|"
                         "//*[@role='contentinfo']"):
        el.drop_tree()

    def _texts(nodes) -> list[str]:
        out = []
        for n in nodes:
            try:
                t = n.text_content().strip()
            except Exception:
                continue
            t = re.sub(r"\s+", " ", t)
            if len(t) > 40:
                out.append(t)
        return out

    paras: list[str] = []
    # publisher-specific paragraph containers, most specific first
    selectors = [
        "//div[contains(@class,'html-p')]",                    # MDPI
        "//div[contains(@class,'c-article-body')]//p",          # Nature/Springer
        "//div[contains(@class,'ArticleBody')]//p",             # Frontiers-ish
        "//div[contains(@class,'article-body')]//p",
        "//article//p",
        "//main//p",
        "//div[contains(@class,'abstract')]//p",
        "//section[contains(@class,'html-abstract')]//div",
    ]
    for xp in selectors:
        try:
            nodes = doc.xpath(xp)
        except Exception:
            nodes = []
        got = _texts(nodes)
        if len(got) >= 8:
            paras = got
            break
    if not paras:
        # generic: all substantial <p>, plus divs that look like paragraphs
        got = _texts(doc.xpath("//p"))
        divs = _texts(doc.xpath("//div[not(*) and string-length(normalize-space(.))>200]"))
        paras = got + divs
    # headings add useful matchable text
    heads = []
    for n in doc.xpath("//h1|//h2|//h3"):
        try:
            t = re.sub(r"\s+", " ", n.text_content()).strip()
        except Exception:
            continue
        if 10 < len(t) < 300 and "menu" not in t.lower():
            heads.append(t)
    text = "\n".join(heads + paras)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return title, text


def _fetch_record(rec: dict) -> bytes | None:
    url = f"{CC_DATA}/{rec['filename']}"
    req = urllib.request.Request(
        url, headers={**UA, "Range": f"bytes={rec['offset']}-{rec['offset'] + rec['length'] - 1}"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = r.read()
        if len(data) != rec["length"]:
            return None
        return gzip.decompress(data)
    except Exception:
        return None


def _warc_html(warc: bytes) -> bytes | None:
    # split WARC headers / HTTP response / payload
    try:
        head_end = warc.index(b"\r\n\r\n")
    except ValueError:
        return None
    rest = warc[head_end + 4:]
    # HTTP response headers
    try:
        h_end = rest.index(b"\r\n\r\n")
    except ValueError:
        return None
    http_head = rest[:h_end].decode("latin1", "replace")
    body = rest[h_end + 4:]
    m = re.search(r"Content-Length:\s*(\d+)", http_head, re.I)
    if m:
        body = body[: int(m.group(1))]
    # Decompress only if the payload is actually gzip-encoded (magic bytes).
    # NOTE: Common Crawl adds "X-Crawler-content-encoding: gzip" informationally
    # after it already decompressed; do NOT trust a substring header match.
    if body[:2] == b"\x1f\x8b":
        try:
            body = gzip.decompress(body)
        except Exception:
            return None
    if len(body) < 100 or b"<html" not in body[:2000].lower():
        # tolerate non-html wrappers but require some markup
        if b"<!doctype" not in body[:2000].lower():
            return None
    return body


_thread_local = threading.local()


def _process_one(rec: dict) -> dict | None:
    warc = _fetch_record(rec)
    if not warc:
        return None
    html = _warc_html(warc)
    if not html or len(html) < 5000:
        return None
    title, text = _extract_text(html, rec["url"])
    words = text.split()
    if len(words) < 200:
        return None
    tokens = [t for t in (norm_word(w) for w in words) if t]
    if len(tokens) < 200:
        return None
    fps = winnow(tokens)
    if not fps:
        return None
    return {
        "doc_id": rec["url"],
        "title": title or rec["url"],
        "source": rec["domain"],
        "n_words": len(tokens),
        "fps": fps,
    }


def stage_fetch(max_docs: int = 0, workers: int = 12) -> None:
    recs = [json.loads(l) for l in open(URLS_JSONL, encoding="utf-8") if l.strip()]
    print(f"{len(recs)} url records to process", flush=True)
    # group by WARC file for locality, then interleave so workers hit many files
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_file[r["filename"]].append(r)
    for lst in by_file.values():
        lst.sort(key=lambda r: r["offset"])
    files = sorted(by_file)
    order: list[dict] = []
    i = 0
    while True:
        added = False
        for f in files:
            if i < len(by_file[f]):
                order.append(by_file[f][i])
                added = True
        i += 1
        if not added:
            break
    if max_docs:
        order = order[:max_docs]
    done = 0
    kept = 0
    skipped = 0
    t0 = time.time()
    lock = threading.Lock()
    with open(FP_JSONL, "w", encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_process_one, r): r for r in order}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    res = fut.result()
                except Exception:
                    res = None
                with lock:
                    done += 1
                    if res:
                        out.write(json.dumps(res) + "\n")
                        kept += 1
                    else:
                        skipped += 1
                    if done % 500 == 0:
                        dt = time.time() - t0
                        print(f"  {done}/{len(order)} done, kept={kept} skipped={skipped} "
                              f"{done/dt:.1f} docs/s", flush=True)
    dt = time.time() - t0
    print(f"fetch done: {kept} docs kept, {skipped} skipped in {dt:.0f}s "
          f"({done/dt:.1f}/s) -> {FP_JSONL}")


# ---------------------------------------------------------------- db stage

SCHEMA = """
CREATE TABLE docs(
  id      INTEGER PRIMARY KEY,
  doc_id  TEXT UNIQUE NOT NULL,
  title   TEXT NOT NULL DEFAULT '',
  source  TEXT NOT NULL DEFAULT ''
);
CREATE TABLE fps(
  fp  INTEGER NOT NULL,
  doc INTEGER NOT NULL REFERENCES docs(id),
  PRIMARY KEY (fp, doc)
);
"""


def stage_db() -> None:
    import subprocess
    if FINAL_DB.exists():
        FINAL_DB.unlink()
    pairs_tsv = OUT_DIR / "fps_pairs.tsv"
    if pairs_tsv.exists():
        pairs_tsv.unlink()
    conn = __import__("sqlite3").connect(str(FINAL_DB))
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-1000000")
    conn.executescript(SCHEMA)
    n_docs = n_pairs = 0
    # Pass 1: docs + dump (fp, doc) pairs to TSV (dedup per doc)
    with open(FP_JSONL, encoding="utf-8") as f, open(pairs_tsv, "w") as pf:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cur = conn.execute(
                "INSERT OR IGNORE INTO docs (doc_id, title, source) VALUES (?,?,?)",
                (d["doc_id"], d.get("title", "") or d["doc_id"], d.get("source", "")))
            rowid = cur.lastrowid or conn.execute(
                "SELECT id FROM docs WHERE doc_id=?", (d["doc_id"],)).fetchone()[0]
            seen: set[int] = set()
            for h, _pos in d["fps"]:
                h = int(h)
                if h not in seen:
                    seen.add(h)
                    pf.write(f"{h}\t{rowid}\n")
                    n_pairs += 1
            n_docs += 1
            if n_docs % 2000 == 0:
                conn.commit()
                print(f"  pass1 ...{n_docs} docs, {n_pairs} pairs", flush=True)
    conn.commit()
    print(f"  pass1 done: {n_docs} docs, {n_pairs} pairs; sorting...", flush=True)
    # Pass 2: external sort by fp for sequential B-tree fill
    sorted_tsv = OUT_DIR / "fps_pairs.sorted.tsv"
    if sorted_tsv.exists():
        sorted_tsv.unlink()
    subprocess.run(["sort", "-n", "-t", "\t", "-k1,1", "-T", "/tmp",
                    "-o", str(sorted_tsv), str(pairs_tsv)], check=True)
    pairs_tsv.unlink()
    print("  sorted; bulk inserting...", flush=True)
    # Pass 3: bulk insert in fp order (CREATE TABLE without PK, then index)
    conn.execute("DROP TABLE IF EXISTS fps")
    conn.execute("CREATE TABLE fps(fp INTEGER NOT NULL, doc INTEGER NOT NULL)")
    batch = []
    ins = 0
    with open(sorted_tsv, encoding="utf-8") as f:
        for line in f:
            fp_s, doc_s = line.split("\t")
            batch.append((int(fp_s), int(doc_s)))
            if len(batch) >= 200000:
                conn.executemany("INSERT INTO fps (fp, doc) VALUES (?,?)", batch)
                ins += len(batch)
                batch = []
                if ins % 2000000 == 0:
                    print(f"  ...{ins} pairs", flush=True)
        if batch:
            conn.executemany("INSERT INTO fps (fp, doc) VALUES (?,?)", batch)
            ins += len(batch)
    conn.commit()
    sorted_tsv.unlink()
    print(f"  inserted {ins} pairs; dedup + index...", flush=True)
    # dedup (fp, doc) and add PK-equivalent index
    conn.execute("CREATE TABLE fps_dedup(fp INTEGER NOT NULL, doc INTEGER NOT NULL, "
                 "PRIMARY KEY (fp, doc)) WITHOUT ROWID")
    conn.execute("INSERT OR IGNORE INTO fps_dedup SELECT fp, doc FROM fps")
    conn.execute("DROP TABLE fps")
    conn.execute("ALTER TABLE fps_dedup RENAME TO fps")
    conn.commit()
    print("  analyzing...", flush=True)
    conn.execute("ANALYZE")
    conn.commit()
    # quick integrity + counts
    qc = conn.execute("PRAGMA quick_check").fetchone()[0]
    dc = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    fc = conn.execute("SELECT COUNT(*) FROM fps").fetchone()[0]
    conn.close()
    size_mb = FINAL_DB.stat().st_size / 1024 / 1024
    print(f"built {FINAL_DB}: {dc} docs, {fc} pairs, {size_mb:.1f} MB, quick_check={qc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["urls", "fetch", "db", "all"], required=True)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.stage in ("urls", "all"):
        print("== stage: urls ==", flush=True)
        stage_urls()
    if a.stage in ("fetch", "all"):
        print("== stage: fetch ==", flush=True)
        stage_fetch(max_docs=a.max_docs, workers=a.workers)
    if a.stage in ("db", "all"):
        print("== stage: db ==", flush=True)
        stage_db()


if __name__ == "__main__":
    main()
