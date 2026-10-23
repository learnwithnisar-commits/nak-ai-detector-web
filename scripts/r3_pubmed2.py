"""Round-3 big: PubMed baseline abstracts, parallel workers (3-5M new).

Worker <tag> processes baseline files pubmed26n{start}..{start+n_files}
(disjoint ranges across workers), stream-parses MedlineCitation records,
and appends compact fingerprints to rotating shards
data/oa_corpus/fp_pubmed2_{tag}_p{n}.jsonl (rotated every SHARD_DOCS).
Active shard published in data/oa_corpus/r3_pubmed2_{tag}.current.
Each .xml.gz is deleted after parsing. Fingerprint-only.

Dedupe: KEEPS r3_seen.db checks (doc_id 'pubmed:{pmid}' and title-hash
collide with the 281k round-1/2 pubmed docs). No in-memory own sets:
baseline files are disjoint, so a PMID can't repeat within a worker;
the build dedupes stragglers.

Usage: python scripts/r3_pubmed2.py <tag> <start> <n_files>
"""

from __future__ import annotations

import gzip
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3_common import CORPUS, title_hash, Seen, fingerprint_text  # noqa: E402
from r3_pubmed import download  # noqa: E402

TMP = CORPUS / "tmp_r3_pubmed"
BASE = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
SHARD_DOCS = 250_000
HEAD_CHARS = 60_000


def parse_file(path: Path, seen: Seen, out):
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
            if seen.has_doc(doc_id) or seen.has_th(title_hash(title)):
                n_dup += 1
                continue
            if not abst:
                n_noabs += 1
                continue
            if len(abst) > HEAD_CHARS:
                abst = abst[:HEAD_CHARS]
            fps, n_tok = fingerprint_text(abst)
            if not fps:
                continue
            out.write(json.dumps({
                "doc_id": doc_id,
                "title": title[:300],
                "source": "pubmed",
                "n_words": n_tok,
                "fps": [int(h) for h, _pos in fps],
            }) + "\n")
            n_new += 1
    return n_new, n_dup, n_noabs


def main() -> None:
    tag = sys.argv[1]
    start = int(sys.argv[2])
    n_files = int(sys.argv[3])
    TMP.mkdir(parents=True, exist_ok=True)
    PROGRESS = CORPUS / f"r3_pubmed2_{tag}_progress.json"
    CURRENT = CORPUS / f"r3_pubmed2_{tag}.current"

    prog: dict = {}
    if PROGRESS.exists():
        try:
            prog = json.loads(PROGRESS.read_text(encoding="utf-8"))
        except Exception:
            prog = {}
    done_files: list[str] = prog.get("done_files", [])
    done_set = set(done_files)
    totals = prog.get("totals", {"new": 0, "dup": 0, "noabs": 0})
    shard_no = prog.get("shard_no", 0)
    shard_docs = prog.get("shard_docs", 0)

    seen = Seen()
    print(f"r3_pubmed2_{tag}: resume new={totals['new']}, "
          f"files {start}..{start + n_files - 1}", flush=True)

    def open_shard():
        p = CORPUS / f"fp_pubmed2_{tag}_p{shard_no}.jsonl"
        CURRENT.write_text(p.name, encoding="utf-8")
        return open(p, "a", encoding="utf-8")

    out = open_shard()
    try:
        for i in range(start, start + n_files):
            fname = f"pubmed26n{i:04d}.xml.gz"
            if fname in done_set:
                continue
            dest = TMP / f"pm2_{tag}_{fname}"
            try:
                download(BASE + fname, dest)
            except Exception as e:
                print(f"  pm2_{tag} download failed {fname}: {str(e)[:60]}",
                      flush=True)
                continue
            n_new, n_dup, n_noabs = parse_file(dest, seen, out)
            dest.unlink(missing_ok=True)
            totals["new"] += n_new
            totals["dup"] += n_dup
            totals["noabs"] += n_noabs
            shard_docs += n_new
            done_files.append(fname)
            done_set.add(fname)
            if shard_docs >= SHARD_DOCS:
                out.close()
                shard_no += 1
                shard_docs = 0
                out = open_shard()
                print(f"  pm2_{tag} rotated to shard {shard_no} "
                      f"(total {totals['new']})", flush=True)
            prog.update({"done_files": done_files, "totals": totals,
                         "shard_no": shard_no, "shard_docs": shard_docs})
            PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
            print(f"  pm2_{tag} {fname}: new={n_new} dup={n_dup} "
                  f"(total {totals['new']})", flush=True)
    finally:
        out.close()
        seen.close()
    prog.update({"done_files": done_files, "totals": totals,
                 "shard_no": shard_no, "shard_docs": shard_docs,
                 "done": len(done_files) >= n_files})
    PROGRESS.write_text(json.dumps(prog), encoding="utf-8")
    print(f"R3_PUBMED2_{tag} DONE: {totals}", flush=True)


if __name__ == "__main__":
    main()
