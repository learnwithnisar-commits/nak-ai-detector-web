"""Round-3: PubMed baseline abstracts, 281k -> ~1M (fingerprint-only).

Continues the round-2 file sequence (pubmed26n0017.xml.gz onward) and
appends to data/oa_corpus/fp_r3_pubmed.jsonl. Downloads each .xml.gz,
stream-parses, deletes after parsing. NO full text stored.

Dedupe: r3_seen.db (round-1/2) + own in-memory set. Checkpoint:
data/oa_corpus/r3_pubmed_progress.json (done_files list).

Usage: python scripts/r3_pubmed.py [n_files] [start]  (default 44, 17)
"""

from __future__ import annotations

import gzip
import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_common import (  # noqa: E402
    CORPUS, title_hash, Seen, fingerprint_text)
from build_plag_index import norm_word  # noqa: E402

OUT_JSONL = CORPUS / "fp_r3_pubmed.jsonl"
PROGRESS = CORPUS / "r3_pubmed_progress.json"
TMP = CORPUS / "tmp_r3_pubmed"  # big disk, not /tmp (tmpfs is only 512MB)
BASE = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(
        url, headers={"User-Agent": "NAK-AI-Detector/1.0 (research index)"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def parse_file(path: Path, seen: Seen, own_ids: set, own_th: set, out):
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
            th = title_hash(title)
            if doc_id in own_ids or th in own_th or seen.has_doc(doc_id) or seen.has_th(th):
                n_dup += 1
                continue
            if not abst:
                n_noabs += 1
                continue
            fps, n_tok = fingerprint_text(abst)
            if not fps:
                continue
            out.write(json.dumps({
                "doc_id": doc_id,
                "title": title[:300],
                "source": "pubmed",
                "n_words": n_tok,
                "fps": fps,
            }) + "\n")
            own_ids.add(doc_id)
            own_th.add(th)
            n_new += 1
    return n_new, n_dup, n_noabs


def main() -> None:
    n_files = int(sys.argv[1]) if len(sys.argv) > 1 else 44
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 17
    TMP.mkdir(parents=True, exist_ok=True)
    prog: dict = {}
    if PROGRESS.exists():
        try:
            prog = json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            prog = {}
    done_files: list[str] = prog.get("done_files", [])
    totals = prog.get("totals", {"new": 0, "dup": 0, "noabs": 0})

    seen = Seen()
    own_ids: set[str] = set()
    own_th: set[str] = set()
    if OUT_JSONL.exists():
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                own_ids.add(d.get("doc_id", ""))
                t = d.get("title") or ""
                if t:
                    own_th.add(title_hash(t))
    print(f"r3_pubmed: resume, own={len(own_ids)}", flush=True)

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
            n_new, n_dup, n_noabs = parse_file(dest, seen, own_ids, own_th, out)
            dest.unlink(missing_ok=True)
            totals["new"] += n_new
            totals["dup"] += n_dup
            totals["noabs"] += n_noabs
            done_files.append(fname)
            prog.update({"done_files": done_files, "totals": totals})
            PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
            print(f"  {fname}: new={n_new} dup={n_dup} (total new={totals['new']})",
                  flush=True)
    finally:
        out.close()
        seen.close()
    prog.update({"done_files": done_files, "totals": totals, "done": True})
    PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
    print(f"R3_PUBMED DONE: {totals}", flush=True)


if __name__ == "__main__":
    main()
