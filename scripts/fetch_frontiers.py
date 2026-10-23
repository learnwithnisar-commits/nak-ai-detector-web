"""Frontiers supplement: fetch recent JATS XML articles directly (robots.txt allows).

Polite: 4 workers, ~0.5s delay between requests per worker.
Appends fingerprints to data/web_academic_index/fp_web.jsonl.
"""
from __future__ import annotations

import concurrent.futures
import json
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location("bwi", str(PROJECT_ROOT / "scripts" / "build_web_index.py"))
bwi = importlib.util.module_from_spec(_spec)
sys.modules["bwi"] = bwi
_spec.loader.exec_module(bwi)

OUT_DIR = PROJECT_ROOT / "data" / "web_academic_index"
FP_JSONL = OUT_DIR / "fp_web.jsonl"
UA = {"User-Agent": "NAK-academic-index/1.0 (research plagiarism index; contact: admin)"}
SITEMAPS = [
    "https://www.frontiersin.org/articles/xml-sitemap_13.xml",
    "https://www.frontiersin.org/articles/xml-sitemap_14.xml",
    "https://www.frontiersin.org/articles/xml-sitemap_15.xml",
]
MAX_ARTICLES = 800


def get_urls() -> list[str]:
    urls = []
    for sm in SITEMAPS:
        req = urllib.request.Request(sm, headers=UA)
        data = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
        urls += re.findall(r"<loc>(https://www\.frontiersin\.org/[^<]+/xml)</loc>", data)
        time.sleep(1)
    # dedup, prefer recent (later sitemaps first)
    seen, out = set(), []
    for u in reversed(urls):
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:MAX_ARTICLES]


def fetch_xml(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            data = r.read()
        time.sleep(0.4)
        return data if len(data) > 3000 else None
    except Exception:
        return None


def parse_jats(xml: bytes) -> tuple[str, str] | None:
    from lxml import etree
    try:
        # strip doctype to avoid DTD fetch attempts
        s = re.sub(rb"<!DOCTYPE[^>]*>", b"", xml, count=1)
        root = etree.fromstring(s)
    except Exception:
        return None
    ns = {"x": "http://www.w3.org/1999/xlink"}
    def txt(el) -> str:
        return re.sub(r"\s+", " ", "".join(el.itertext())).strip()
    title_els = root.xpath("//article-title")
    title = txt(title_els[0])[:300] if title_els else ""
    parts: list[str] = []
    for xp in ("//abstract//p", "//body//p", "//body//title"):
        for el in root.xpath(xp):
            t = txt(el)
            if len(t) > 40:
                parts.append(t)
    text = "\n".join(parts)
    if len(text.split()) < 200:
        return None
    return title, text


def process(url: str) -> dict | None:
    xml = fetch_xml(url)
    if not xml:
        return None
    parsed = parse_jats(xml)
    if not parsed:
        return None
    title, text = parsed
    # canonical article url (strip /xml)
    doc_url = re.sub(r"/xml$", "/full", url)
    tokens = [t for t in (bwi.norm_word(w) for w in text.split()) if t]
    if len(tokens) < 200:
        return None
    fps = bwi.winnow(tokens)
    if not fps:
        return None
    return {"doc_id": doc_url, "title": title or doc_url,
            "source": "frontiersin.org", "n_words": len(tokens), "fps": fps}


def main() -> None:
    urls = get_urls()
    print(f"frontiers: {len(urls)} article urls", flush=True)
    done = kept = 0
    lock = threading.Lock()
    t0 = time.time()
    with open(FP_JSONL, "a", encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(process, u): u for u in urls}
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
                    if done % 100 == 0:
                        print(f"  {done}/{len(urls)} kept={kept} "
                              f"{done/(time.time()-t0):.1f}/s", flush=True)
    print(f"frontiers done: kept={kept}/{done}", flush=True)


if __name__ == "__main__":
    main()
