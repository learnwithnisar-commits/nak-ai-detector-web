#!/usr/bin/env python3
"""Phase 4 — build a locked, leakage-resistant train/validation/test split.

Reads:  data/phase2_dataset.jsonl   (NEVER modified)
Writes: data/phase4_splits/
  - splits.json            {doc_id: split} for the numeric AI-score pool
  - train.json / val.json / test.json   doc_id lists
  - star_band_pool.json    68 interval-censored *% doc_ids (NOT split-assigned)
  - split_meta.json        seed, rules, counts, sha256 checksums
  - phase4_build.log       build log

NO model fitting. NO model files touched. Re-runnable: same seed -> same split.
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
OUT = os.path.join(DATA, "phase4_splits")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "phase4_build.log"), "w")

def log(msg):
    LOG.write(msg + "\n"); LOG.flush(); print(msg)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    ds_path = os.path.join(DATA, "phase2_dataset.jsonl")
    recs = [json.loads(l) for l in open(ds_path, encoding="utf-8")]
    log(f"dataset records: {len(recs)}")

    # ---- eligibility: exact numeric AI score AND usable text ----
    def usable(r):
        return bool(r.get("text")) and (r.get("text_words") or 0) >= 50
    elig = [r for r in recs
            if isinstance(r.get("turnitin_ai_score"), (int, float)) and usable(r)]
    log(f"eligible numeric+text records: {len(elig)} "
        f"(zero={sum(1 for r in elig if r['turnitin_ai_score']==0)}, "
        f"nonzero={sum(1 for r in elig if r['turnitin_ai_score']>0)})")

    # ---- grouping: doc_group merged with connected components of related_doc_ids ----
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
    # verify the mandated revision pair is grouped
    pair = [m for g in groups.values() for m in g if m["doc_id"] in ("23626","23626__2")]
    assert len(pair) == 2 and find("23626") == find("23626__2"), "23626 pair NOT grouped!"
    log("verified: 23626 / 23626__2 revision pair in the SAME group")

    # ---- stratified split: stratum = group contains any AI>0 ----
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

    # ---- emit manifests ----
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

    # ---- *% interval-censored pool (NOT split-assigned, no invented scores) ----
    star = [r for r in recs if r.get("turnitin_ai_band") == "*%"]
    star_ids = sorted(r["doc_id"] for r in star)
    p = os.path.join(OUT, "star_band_pool.json")
    json.dump({"band": "1-19% (Turnitin '*' display; exact value unknown)",
               "n": len(star_ids), "doc_ids": star_ids}, open(p, "w"), indent=1)
    log(f"wrote {p}: {len(star_ids)} interval-censored records")

    # ---- excluded records record ----
    assigned = set(doc_split) | set(star_ids)
    excluded = sorted(set(r["doc_id"] for r in recs) - assigned)
    log(f"excluded from AI-score eval (similarity-only/old-format/unscored): {len(excluded)}")

    meta = {
        "seed": SEED,
        "grouping_rules": [
            "doc_group from Phase 2, merged with connected components of related_doc_ids",
            "23626 / 23626__2 revision pair forced into one group (verified)",
            "stratified by group stratum: 'nonzero' (any member AI>0) vs 'zero'",
            "within stratum: sorted group ids, seeded shuffle (random.Random(20261017)), "
            "70/15/15 split by group count with rounding",
            "*% records kept as separate interval-censored pool, never split-assigned",
            "test split is LOCKED: no feature/model/threshold selection or calibration on it",
        ],
        "eligible_numeric_records": len(elig),
        "groups": len(groups),
        "split_group_counts": dict(Counter(assignment.values())),
        "split_doc_counts": {k: len(v) for k, v in splits.items()},
        "star_band_pool_n": len(star_ids),
        "excluded_n": len(excluded),
        "checksums_sha256": {f: sha256(os.path.join(OUT, f))
                             for f in ["train.json", "val.json", "test.json",
                                       "splits.json", "star_band_pool.json"]},
        "note": "Turnitin scores are reference scores, not verified authorship ground truth.",
    }
    mp = os.path.join(OUT, "split_meta.json")
    json.dump(meta, open(mp, "w"), indent=2)
    log(f"wrote {mp}")
    log("DONE. No training performed. No model files touched.")
    LOG.close()

if __name__ == "__main__":
    sys.exit(main())
