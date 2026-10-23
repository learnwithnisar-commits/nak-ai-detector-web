#!/usr/bin/env python3
"""Phase 4 v2 — re-derived locked split on the rebuilt (chrome-stripped) dataset.

Reads:  data/phase2_dataset_v2.jsonl   (NEVER modified)
Writes: data/phase4_splits_v2/
  - splits.json, train.json / val.json / test.json, star_band_pool.json,
    split_meta.json, phase4_v2_build.log

Same seed (20261017) and grouping rules as build_phase4_split.py.
The v1 locked test set (data/phase4_splits/test.json) is NOT touched.

NO model fitting. NO model files touched.
"""
import hashlib
import json
import os
import random
import sys
from collections import Counter

SEED = 20261017
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "phase4_splits_v2")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "phase4_v2_build.log"), "w")

def log(msg):
    LOG.write(msg + "\n"); LOG.flush(); print(msg)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    ds_path = os.path.join(DATA, "phase2_dataset_v2.jsonl")
    recs = [json.loads(l) for l in open(ds_path, encoding="utf-8")]
    log(f"dataset v2 records: {len(recs)}")

    # ---- v1 locked-test quarantine: any v2 record whose source PDFs intersect
    # the v1 locked test set is EXCLUDED from train/val/test entirely, so a
    # candidate trained on v2 splits can later be scored on the v1 locked test
    # without contamination. ----
    v1_test_ids = json.load(open(os.path.join(DATA, "phase4_splits", "test.json")))
    v1_recs = {json.loads(l)["doc_id"]: json.loads(l)
               for l in open(os.path.join(DATA, "phase2_dataset.jsonl"), encoding="utf-8")}
    v1_test_pdfs = set()
    for d in v1_test_ids:
        v1_test_pdfs.update(v1_recs[d].get("all_source_pdfs", []))
    log(f"v1 locked-test source PDFs quarantined: {len(v1_test_pdfs)}")
    quarantined = []
    kept = []
    for r in recs:
        if set(r.get("all_source_pdfs", [])) & v1_test_pdfs:
            quarantined.append(r["doc_id"])
        else:
            kept.append(r)
    log(f"quarantined (in v1 locked test): {len(quarantined)} -> {sorted(quarantined)}")
    recs = kept

    def usable(r):
        return bool(r.get("text")) and (r.get("text_words") or 0) >= 50
    elig = [r for r in recs
            if isinstance(r.get("turnitin_ai_score"), (int, float)) and usable(r)]
    log(f"eligible numeric+text records: {len(elig)} "
        f"(zero={sum(1 for r in elig if r['turnitin_ai_score']==0)}, "
        f"nonzero={sum(1 for r in elig if r['turnitin_ai_score']>0)})")

    ids = {r["doc_id"] for r in elig}
    parent = {r["doc_id"]: r["doc_id"] for r in elig}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    for r in elig:
        for o in r.get("related_doc_ids") or []:
            if o in ids:
                union(r["doc_id"], o)
    groups = {}
    for r in elig:
        groups.setdefault(find(r["doc_id"]), []).append(r)
    log(f"document groups: {len(groups)} "
        f"(multi-doc: {sum(1 for g in groups.values() if len(g)>1)})")
    for g, members in groups.items():
        if len(members) > 1:
            log(f"  multi-doc group {g}: "
                + ", ".join(f"{m['doc_id']}({m['turnitin_ai_score']})" for m in members))

    # verify the mandated 23626 revision pair is grouped (locate by source pdf)
    by_src = {}
    for r in elig:
        for p in r.get("all_source_pdfs", [r["source_pdf"]]):
            by_src[p] = r["doc_id"]
    d1 = by_src.get("23626.pdf"); d2 = by_src.get("23626__1___1.pdf")
    if d1 and d2:
        assert find(d1) == find(d2), "23626 pair NOT grouped!"
        log(f"verified: 23626 pair ({d1} / {d2}) in the SAME group")
    else:
        log(f"WARNING: 23626 pair pdfs not both in eligible pool ({d1}, {d2})")

    def stratum(members):
        return "nonzero" if any(m["turnitin_ai_score"] > 0 for m in members) else "zero"
    strata = {}
    for g, members in groups.items():
        strata.setdefault(stratum(members), []).append(g)
    log(f"strata: { {k: len(v) for k, v in strata.items()} }")

    rng = random.Random(SEED)
    assignment = {}
    for sname in sorted(strata):
        glist = sorted(strata[sname])
        rng.shuffle(glist)
        n = len(glist)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        n_test = n - n_train - n_val
        log(f"stratum {sname}: n={n} -> train={n_train} val={n_val} test={n_test}")
        for g in glist[:n_train]:
            assignment[g] = "train"
        for g in glist[n_train:n_train+n_val]:
            assignment[g] = "val"
        for g in glist[n_train+n_val:]:
            assignment[g] = "test"

    doc_split = {}
    for g, members in groups.items():
        for m in members:
            doc_split[m["doc_id"]] = assignment[g]
    splits = {"train": [], "val": [], "test": []}
    for doc_id, sp in sorted(doc_split.items()):
        splits[sp].append(doc_id)
    for sp, lst in splits.items():
        p = os.path.join(OUT, f"{sp}.json")
        json.dump(lst, open(p, "w"), indent=1)
        log(f"wrote {p}: {len(lst)} docs (sha256 {sha256(p)[:16]}...)")
    p = os.path.join(OUT, "splits.json")
    json.dump(doc_split, open(p, "w"), indent=1, sort_keys=True)

    star = [r for r in recs if r.get("turnitin_ai_band") == "*%"]
    star_ids = sorted(r["doc_id"] for r in star)
    p = os.path.join(OUT, "star_band_pool.json")
    json.dump({"band": "1-19% (Turnitin '*' display; exact value unknown)",
               "n": len(star_ids), "doc_ids": star_ids}, open(p, "w"), indent=1)
    log(f"wrote {p}: {len(star_ids)} interval-censored records")

    assigned = set(doc_split) | set(star_ids)
    excluded = sorted(set(r["doc_id"] for r in recs) - assigned)
    log(f"excluded from AI-score eval (similarity-only/old-format/unscored): {len(excluded)}")

    meta = {
        "seed": SEED,
        "source_dataset": "data/phase2_dataset_v2.jsonl (chrome-stripped rebuild)",
        "grouping_rules": [
            "doc_group from Phase 2 v2, merged with connected components of related_doc_ids",
            "23626 / 23626__1___1 revision pair forced into one group (verified by source pdf)",
            "stratified by group stratum: 'nonzero' (any member AI>0) vs 'zero'",
            "within stratum: sorted group ids, seeded shuffle (random.Random(20261017)), "
            "70/15/15 split by group count with rounding",
            "*% records kept as separate interval-censored pool, never split-assigned",
            "v1 locked test set (data/phase4_splits/test.json) untouched",
        ],
        "eligible_numeric_records": len(elig),
        "groups": len(groups),
        "quarantined_v1_locked_test_docs": sorted(quarantined),
        "quarantined_n": len(quarantined),
        "split_group_counts": dict(Counter(assignment.values())),
        "split_doc_counts": {k: len(v) for k, v in splits.items()},
        "star_band_pool_n": len(star_ids),
        "excluded_n": len(excluded),
        "checksums_sha256": {f: sha256(os.path.join(OUT, f))
                             for f in ["train.json", "val.json", "test.json",
                                       "splits.json", "star_band_pool.json"]},
        "note": "Turnitin scores are reference scores, not verified authorship ground truth. "
                "1-19% AI scores labeled '*%' per Nisar's rule; raw values in turnitin_ai_raw.",
    }
    mp = os.path.join(OUT, "split_meta.json")
    json.dump(meta, open(mp, "w"), indent=2)
    log(f"wrote {mp}")
    log("DONE. No training performed. No model files touched.")
    LOG.close()

if __name__ == "__main__":
    sys.exit(main())
