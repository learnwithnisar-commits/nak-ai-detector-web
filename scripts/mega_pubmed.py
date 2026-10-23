"""MEGA plagiarism index: PubMed baseline abstracts (fingerprint-only).

Downloads N PubMed baseline XML files (https://ftp.ncbi.nlm.nih.gov/pubmed/
baseline/ — open bulk data, no key), stream-parses MedlineCitation records,
and appends winnowing fingerprints (k=5/w=4, same as app/plag.py) to
data/oa_corpus/fp_mega_pubmed.jsonl. NO full text stored.

Dedupe: doc_id (pubmed:PMID) + title-hash against the existing index
(fp_winnow.jsonl + fp_mega*.jsonl snapshot at start).

Checkpoint: data/oa_corpus/mega_pubmed_progress.json — rerunning resumes
from the next unprocessed file. Downloaded .xml.gz files are deleted after
parsing to save disk.

Usage:
    python scripts/mega_pubmed.py [n_files]   (default 8, ~100k records)
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT_ROOT / "data" / "oa_corpus"
OUT_JSONL = CORPUS / "fp_mega_pubmed.jsonl"
PROGRESS = CORPUS / "mega_pubmed_progress.json"
TMP = Path("/tmp/pubmed_mega")
BASE = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from build_plag_index import winnow, norm_word  # noqa: E402  (same K/W/hash)

MIN_TOKENS = 30


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def title_hash(t: str) -> str:
    return hashlib.sha1(norm_title(t).encode()).hexdigest()[:16]


def load_seen() -> tuple[set[str], set[str]]:
    doc_ids: set[str] = set()
    thashes: set[str] = set()
    for path in [CORPUS / "fp_winnow.jsonl"] + sorted(CORPUS.glob("fp_mega*.jsonl")):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                doc_ids.add(d.get("doc_id", ""))
                th = d.get("title") or ""
                if th:
                    thashes.add(title_hash(th))
    return doc_ids, thashes


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(
        url, headers={"User-Agent": "NAK-AI-Detector/1.0 (research index)"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def parse_file(path: Path, seen_ids: set[str], seen_th: set[str], out) -> tuple[int, int, int]:
    n_new = n_dup = n_noabs = 0
    with gzip.open(path, "rb") as gz:
        ctx = ET.iterparse(gz, events=("end",))
        for _ev, elem in ctx:
            if elem.tag != "MedlineCitation":
                continue
            pmid = (elem.findtext("PMID") or "").strip()
            title = (elem.findtext("./Article/ArticleTitle") or "").strip()
            abst = " ".join(
                (t.text or "") for t in elem.findall("./Article/Abstract/AbstractText"))
            abst = re.sub(r"\s+", " ", abst).strip()
            elem.clear()
            if not pmid or not title:
                continue
            doc_id = f"pubmed:{pmid}"
            if doc_id in seen_ids or title_hash(title) in seen_th:
                n_dup += 1
                continue
            if not abst:
                n_noabs += 1
                continue
            tokens = [t for t in (norm_word(w) for w in abst.split()) if t]
            if len(tokens) < MIN_TOKENS:
                continue
            fps = winnow(tokens)
            if not fps:
                continue
            out.write(json.dumps({
                "doc_id": doc_id,
                "title": title[:300],
                "source": "pubmed",
                "n_words": len(tokens),
                "fps": fps,
            }) + "\n")
            seen_ids.add(doc_id)
            seen_th.add(title_hash(title))
            n_new += 1
    return n_new, n_dup, n_noabs


def main() -> None:
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    TMP.mkdir(parents=True, exist_ok=True)
    prog: dict = {}
    if PROGRESS.exists():
        try:
            prog = json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            prog = {}
    done_files: list[str] = prog.get("done_files", [])
    totals = prog.get("totals", {"new": 0, "dup": 0, "noabs": 0})

    print("loading seen doc_ids + title hashes ...", flush=True)
    seen_ids, seen_th = load_seen()
    print(f"  {len(seen_ids)} doc_ids, {len(seen_th)} title hashes", flush=True)

    # file list: deterministic order; start offset allows targeting newer files
    # (higher numbers = newer PMIDs, better abstract coverage)
    files = [f"pubmed26n{i:04d}.xml.gz" for i in range(start, start + n_files)]
    out = open(OUT_JSONL, "a", encoding="utf-8")
    try:
        for fname in files:
            if fname in done_files:
                continue
            dest = TMP / fname
            print(f"downloading {fname} ...", flush=True)
            t0 = time.time()
            try:
                download(BASE + fname, dest)
            except Exception as e:
                print(f"  download failed: {str(e)[:80]}, skipping", flush=True)
                continue
            print(f"  {(dest.stat().st_size / 1e6):.0f} MB in {time.time() - t0:.0f}s; parsing ...",
                  flush=True)
            n_new, n_dup, n_noabs = parse_file(dest, seen_ids, seen_th, out)
            dest.unlink(missing_ok=True)
            totals["new"] += n_new
            totals["dup"] += n_dup
            totals["noabs"] += n_noabs
            done_files.append(fname)
            prog.update({"done_files": done_files, "totals": totals})
            PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
            print(f"  {fname}: new={n_new} dup={n_dup} noabs={n_noabs} "
                  f"(total new={totals['new']})", flush=True)
    finally:
        out.close()
    print(f"PUBMED DONE: {totals}", flush=True)


if __name__ == "__main__":
    main()
