#!/usr/bin/env python3
"""Phase 2 finalize: holdout fingerprint check, report pairing, final statuses."""
import os, json, csv, hashlib, re

PROJ = "/home/hatch/workspace/ai-detector"
FILES = "/home/hatch/workspace/user/files"
DATA = f"{PROJ}/data"

def fp_holdout(text):
    norm = re.sub(r"\s+", " ", text[:2000]).strip().lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()

def read_docx(p):
    try:
        from docx import Document
        d = Document(p)
        return '\n'.join([para.text for para in d.paragraphs])
    except Exception as e:
        return ""

# holdout fingerprints
hm = json.load(open(f"{DATA}/holdout_manifest_A.json"))
holdout_fps = {d.get('fingerprint_md5'): d.get('file') for d in hm['docs']}
print(f"holdout fps: {len(holdout_fps)}")

# inventory index by doc_group
inv = {}
for r in csv.DictReader(open(f"{DATA}/inventory_472.csv", encoding='utf-8')):
    inv.setdefault(r['doc_group'], []).append(r)

# read crosscheck
rows = list(csv.DictReader(open(f"{DATA}/candidate_crosscheck.csv", encoding='utf-8')))

final = []
for r in rows:
    group = r['doc_group']
    fn = r['filename']
    text = read_docx(f"{FILES}/{fn}") if os.path.exists(f"{FILES}/{fn}") else ""
    hfp = fp_holdout(text) if text else ""

    # holdout check
    holdout_hit = holdout_fps.get(hfp)

    # report pairing from inventory
    ai_pct, sim_pct, ai_file, sim_file, sub = "", "", "", "", ""
    for ir in inv.get(group, []):
        role = ir['role']
        if role == 'ai_report':
            ai_file = ir['filename']
            ai_pct = ir['ai_pct']
            sub = ir['submission_id'] or sub
        elif role == 'similarity_report':
            sim_file = ir['filename']
            sim_pct = ir['sim_pct']
            sub = ir['submission_id'] or sub

    status = r['status']
    rationale = r['rationale']

    # fix _28804: 0.481 near-dup of anchor version -> duplicate
    if group == '_28804_-_CVL398_Su26_Project_PD2-1_3_nw9f':
        status = "duplicate_other_dataset"
        rationale = f"jaccard 0.481 vs anchors_new_oct16/28804_cvl398_su26_project_pd2_1.txt — revised version of same CVL398 PD2 document; leakage risk"

    # holdout overlap overrides
    if holdout_hit:
        status = "duplicate_locked_test"
        rationale = f"holdout fingerprint match: {holdout_hit} — LOCKED TEST OVERLAP, must stay out of training"

    # refine unique statuses by label availability
    if status == "unique_unverified_pair":
        if ai_pct and ai_pct not in ("", "band_1_19"):
            # numeric verified
            try:
                float(ai_pct)
                status = "unique_verified_numeric_label"
                rationale += f"; AI report {ai_file} shows verified numeric {ai_pct}% (submission {sub})"
            except ValueError:
                pass
        if status == "unique_unverified_pair" and ai_pct == "band_1_19":
            status = "unique_band_only"
            rationale += f"; AI report {ai_file} shows categorical *% band only (numeric=null per V22 rule)"
        if status == "unique_unverified_pair" and not ai_file:
            rationale += "; no AI report paired in inventory — label unverified"

    final.append({
        "doc_group": group,
        "filename": fn,
        "md5": r['md5'],
        "word_count": r['word_count'],
        "closest_match": r['closest_match'],
        "match_source": r['match_source'],
        "similarity": r['similarity'],
        "holdout_fingerprint_hit": holdout_hit or "",
        "ai_report": ai_file,
        "verified_ai_pct": ai_pct,
        "similarity_report": sim_file,
        "similarity_pct": sim_pct,
        "submission_id": sub,
        "status": status,
        "rationale": rationale,
    })

with open(f"{DATA}/candidate_crosscheck.csv", 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(final[0].keys()))
    w.writeheader()
    w.writerows(final)

from collections import Counter
cnt = Counter(x['status'] for x in final)
print("FINAL status counts:", dict(cnt))
print(f"\nWrote {DATA}/candidate_crosscheck.csv ({len(final)} rows)")

# print uniques for the report
print("\n--- unique_verified_numeric_label ---")
for x in final:
    if x['status'] == 'unique_verified_numeric_label':
        print(f"  {x['doc_group']}: AI={x['verified_ai_pct']}% sim={x['similarity_pct']}% words={x['word_count']}")
print("--- unique_band_only ---")
for x in final:
    if x['status'] == 'unique_band_only':
        print(f"  {x['doc_group']}: AI=*band sim={x['similarity_pct']}% words={x['word_count']}")
print("--- unique_unverified_pair ---")
for x in final:
    if x['status'] == 'unique_unverified_pair':
        print(f"  {x['doc_group']}: words={x['word_count']} rationale={x['rationale'][:100]}")
