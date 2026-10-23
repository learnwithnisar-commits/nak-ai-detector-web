#!/usr/bin/env python3
"""Phase 2: Cross-check 34 candidates against ALL data sources."""
import os, json, csv, hashlib, re
from collections import defaultdict

PROJ = "/home/hatch/workspace/ai-detector"
FILES = "/home/hatch/workspace/user/files"
DATA = f"{PROJ}/data"

def norm(t):
    t = t.lower()
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def shingles(t, k=5):
    words = norm(t).split()
    if len(words) < k:
        return set()
    return {' '.join(words[i:i+k]) for i in range(len(words)-k+1)}

def jaccard(a, b):
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def fp_norm(t):
    return hashlib.md5(norm(t)[:2000].encode()).hexdigest()

def read_txt(p):
    try:
        with open(p, encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception:
        return ""

def read_docx(p):
    try:
        from docx import Document
        d = Document(p)
        return '\n'.join([para.text for para in d.paragraphs])
    except Exception as e:
        return f"__UNREADABLE__: {e}"

# ---- Build reference DB: name -> (text, source_label) ----
refs = {}  # key -> (text, source)

def add_ref(key, text, source):
    if text and len(text.strip()) > 50:
        refs[key] = (text, source)

# 1. anchors_new_oct16
ad = f"{DATA}/anchors_new_oct16"
for fn in os.listdir(ad):
    if fn.endswith('.txt'):
        add_ref(f"anchors_new_oct16/{fn}", read_txt(f"{ad}/{fn}"), "anchors_new_oct16")

# 2. anchors_oct16r2
ad2 = f"{DATA}/anchors_oct16r2"
for fn in os.listdir(ad2):
    if fn.endswith('.txt'):
        add_ref(f"anchors_oct16r2/{fn}", read_txt(f"{ad2}/{fn}"), "anchors_oct16r2")

# 3. mass_paired (both ai and human texts)
n_mass = 0
for line in open(f"{DATA}/turnitin_mass_paired.jsonl", encoding='utf-8'):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    rid = r.get('record_id', f'mass_{n_mass}')
    add_ref(f"mass_paired/{rid}", r.get('text', ''), f"mass_paired({r.get('label','?')})")
    n_mass += 1

# 4. Legacy eval set docs (extract from user/files if present)
legacy_files = [
    ("Full-Paper-sincere-conf.docx", "legacy_eval/FullPaper"),
    ("22103.docx", "legacy_eval/22103"),
    ("Personal_statement_PhD.docx", "legacy_eval/PhD"),
    ("Listeria_goat_MANUSCRIPT_IMPROVED_CLEAN.docx", "legacy_eval/Listeria"),
    ("XDR_Typhoid_Pakistan_PJMR_Format.docx", "legacy_eval/XDR_Typhoid"),
    ("aigc_download_2027795315876175872_aigc_rate.docx", "legacy_eval/AIGC"),
    ("upload.docx", "legacy_eval/Competition"),
    ("Essay-_Junlan_Li-25041323.docx", "legacy_eval/Junlan_Li"),
    ("Liu_Yueru_Vicky__FA1_Yueru__Vicky__LIU.docx", "legacy_eval/Liu_Yueru"),
    ("ES1002_Essay.docx", "legacy_eval/ES1002"),
]
for fn, label in legacy_files:
    p = f"{FILES}/{fn}"
    if os.path.exists(p):
        add_ref(label, read_docx(p), label)
# ash_coal txt
ap = f"{DATA}/real_ai/ash_coal_essay_chatgpt.txt"
if os.path.exists(ap):
    add_ref("legacy_eval/ash_coal", read_txt(ap), "legacy_eval/ash_coal")

# 5. WhatsApp 21 docs (from whatsapp21_training.jsonl + comparison md names)
wa_names = []
try:
    for line in open(f"{DATA}/whatsapp21_training.jsonl", encoding='utf-8'):
        line = line.strip()
        if line:
            r = json.loads(line)
            rid = r.get('record_id', r.get('id', 'wa_?'))
            add_ref(f"whatsapp21/{rid}", r.get('text', ''), "whatsapp21_training")
except FileNotFoundError:
    pass
# also check comparison md for doc names (best effort)
try:
    md = open(f"{DATA}/whatsapp_21_comparison_20261017.md", encoding='utf-8').read()
    # doc names appear like **Name** — capture a few known ones
    for nm in ["BDO_Lebanon", "Personal_statement_PhD", "Arab_Design", "77701", "Nikita", "nikita"]:
        pass
except FileNotFoundError:
    pass

# 6. Holdout 31 (fingerprints only — check name overlap via manifest)
holdout_names = set()
try:
    hm = json.load(open(f"{DATA}/holdout_manifest_A.json"))
    for d in hm.get('docs', []):
        holdout_names.add(d.get('filename', d.get('name', '')))
        # also add text if present
        if d.get('text'):
            add_ref(f"holdout31/{d.get('filename','?')}", d['text'], "holdout_31")
except Exception as e:
    print(f"holdout manifest read issue: {e}")

# 7. new_001..new_012 already ingested (from manifest)
ingested = {}
try:
    with open(f"{DATA}/new_data_manifest.csv", encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ingested[row['doc_id']] = row
except FileNotFoundError:
    pass

print(f"Reference DB built: {len(refs)} texts, holdout names: {len(holdout_names)}, ingested: {len(ingested)}")

# ---- Load candidates ----
cands = []
with open(f"{DATA}/eligible_unique_documents.csv", encoding='utf-8') as f:
    for row in csv.DictReader(f):
        cands.append(row)
print(f"Candidates: {len(cands)}")

# ---- Cross-check each candidate ----
results = []
for c in cands:
    group = c['doc_group']
    fn = c['filename']
    fpath = f"{FILES}/{fn}"
    text = read_docx(fpath) if os.path.exists(fpath) else ""
    unreadable = text.startswith("__UNREADABLE__")
    wc = c.get('word_count', '')

    best = None  # (score, refkey, source)
    if text and not unreadable:
        cfp = fp_norm(text)
        # exact fingerprint pass first
        for rk, (rt, src) in refs.items():
            if fp_norm(rt) == cfp:
                best = (1.0, rk, src)
                break
        if not best:
            # jaccard pass (sample: compare against all refs; refs ~ 246+219+11+wa — fine)
            bscore, bkey, bsrc = 0.0, None, None
            for rk, (rt, src) in refs.items():
                s = jaccard(text, rt)
                if s > bscore:
                    bscore, bkey, bsrc = s, rk, src
            best = (round(bscore, 3), bkey, bsrc)

    score, refkey, source = best if best else (0.0, None, None)

    # Report pairing: find AI report PDF(s) for this group in user/files
    ai_report = None
    sim_report = None
    ai_pct = None
    sim_pct = None
    # check inventory_472.csv for this group's reports
    # Decision logic
    fn_lower = fn.lower()
    status = ""
    rationale = ""
    if unreadable:
        status = "unique_unverified_pair"
        rationale = f"docx unreadable ({text[:80]}); cannot verify content"
    elif score >= 0.9:
        # near-exact duplicate — classify by source
        if source and source.startswith("legacy_eval"):
            status = "duplicate_other_dataset"
            rationale = f"jaccard {score} vs {refkey} (legacy eval set — in-sample anchor, NOT independent)"
        elif source and "mass_paired" in source:
            status = "duplicate_existing_training"
            rationale = f"jaccard {score} vs {refkey}"
        elif source and "holdout_31" in source:
            status = "duplicate_locked_test"
            rationale = f"jaccard {score} vs {refkey} — LOCKED TEST OVERLAP"
        elif source and "anchors_" in source:
            status = "duplicate_other_dataset"
            rationale = f"jaccard {score} vs {refkey} (anchor dir)"
        elif source and "whatsapp21" in source:
            status = "duplicate_other_dataset"
            rationale = f"jaccard {score} vs {refkey} (WhatsApp 21 set)"
        else:
            status = "duplicate_other_dataset"
            rationale = f"jaccard {score} vs {refkey}"
    elif score >= 0.5:
        status = "duplicate_other_dataset"
        rationale = f"near-duplicate jaccard {score} vs {refkey} ({source}); likely revised version — leakage risk"
    else:
        # check name-based known anchors even with low jaccard
        known_anchors = {
            "full-paper-sincere-conf": "legacy_eval/FullPaper (Turnitin 80)",
            "personal_statement_phd": "legacy_eval/PhD (Turnitin 100)",
            "xdr_typhoid": "legacy_eval/XDR_Typhoid (Turnitin 64)",
            "check_nikita": "known anchor Nikita (Turnitin 48)",
            "aigc_download": "legacy_eval/AIGC (Turnitin 43)",
            "reintroducing_advisory_services_at_bdo_lebanon": "WhatsApp anchor BDO_Lebanon (Turnitin 22)",
            "ai_automation_and_employment_thesis": "family-labeled Claude training data",
            "upload": "legacy_eval/Competition (Turnitin *)",
            "77701": "WhatsApp doc 77701 (Turnitin 0, NAK FP case)",
        }
        matched_anchor = None
        for k, v in known_anchors.items():
            if k in fn_lower:
                matched_anchor = v
                break
        if matched_anchor:
            status = "duplicate_other_dataset"
            rationale = f"name-matched known anchor: {matched_anchor}; content jaccard {score} (low — may be different version, verify manually)"
        else:
            status = "unique_unverified_pair"
            rationale = f"max jaccard {score} vs {refkey or 'none'}; no report pairing verified yet"

    results.append({
        "doc_group": group,
        "filename": fn,
        "md5": c.get('md5', ''),
        "word_count": wc,
        "closest_match": refkey or "",
        "match_source": source or "",
        "similarity": score,
        "ai_report_paired": ai_report or "",
        "verified_ai_pct": ai_pct if ai_pct is not None else "",
        "similarity_pct": sim_pct if sim_pct is not None else "",
        "status": status,
        "rationale": rationale,
    })
    print(f"{group}: score={score} src={source} -> {status}")

# ---- new_008 special investigation ----
print("\n=== new_008 investigation ===")
n8 = ingested.get('new_008', {})
print(json.dumps(n8, indent=1)[:1500])

# Write output CSV
outp = f"{DATA}/candidate_crosscheck.csv"
with open(outp, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
print(f"\nWrote {outp} ({len(results)} rows)")

# Summary counts
from collections import Counter
cnt = Counter(r['status'] for r in results)
print("\nStatus counts:", dict(cnt))
