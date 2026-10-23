#!/usr/bin/env python3
"""Phase 2 (audit): build a structured, re-runnable dataset from the 330 Turnitin PDFs.

Reads:  ~/workspace/user/files/*.pdf
Writes: ~/workspace/ai-detector/data/phase2_dataset.jsonl  (one record per UNIQUE document)
Logs:   ~/workspace/ai-detector/data/calibration/phase2_build.log

Rules (Nisar's hard constraints):
  - No model files are touched; no training of any kind.
  - Missing values stay null. Never invent a score, never zero-fill, never infer
    authorship from a Turnitin score.
  - AI-score prediction and authorship classification are separate fields.
  - Turnitin scores are REFERENCE scores, not ground truth.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import collections
from datetime import datetime, timezone

FILES_DIR = os.path.expanduser("~/workspace/user/files")
OUT_JSONL = os.path.expanduser("~/workspace/ai-detector/data/phase2_dataset.jsonl")
LOG_PATH = os.path.expanduser("~/workspace/ai-detector/data/calibration/phase2_build.log")

os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logf = open(LOG_PATH, "w", encoding="utf-8")

def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    logf.write(line + "\n")
    logf.flush()

# ---------------------------------------------------------------- pdf helpers
def pdf_page_count(path):
    try:
        out = subprocess.run(["pdfinfo", path], capture_output=True, text=True, timeout=60)
        m = re.search(r"Pages:\s+(\d+)", out.stdout)
        return int(m.group(1)) if m else None
    except Exception:
        return None

def pdf_text_pages(path):
    """Return list of per-page text via pdftotext -layout, or None on failure."""
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", path, "-"],
            capture_output=True, text=True, timeout=300)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return out.stdout.split("\f")
    except Exception:
        return None

# ---------------------------------------------------------------- patterns
AI_SCORE_RE = re.compile(r"([0-9]+|\*)%\s+detected as AI")
SIM_NEW_RE = re.compile(r"(\d+)%\s+Overall Similarity")
SIM_OLD_RE = re.compile(r"(\d+)\s*%\s*\n\s*SIMILARITY INDEX")
SIM_OLD2_RE = re.compile(r"(?m)^(\d+)\s*\n\s*SIMILARITY INDEX")  # number on line before, no % sign
SUBID_NEW_RE = re.compile(r"Submission ID\s*\n\s*(trn:oid:::\S+)")
SUBID_OLD_RE = re.compile(r"Submission ID:\s*(\d+)")
DATE_NEW_RE = re.compile(r"([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}\s+[AP]M)")
DATE_OLD_RE = re.compile(r"Submission date:\s*([^\n]+)")
PAGE_HDR_RE = re.compile(r"Page\s+\d+\s+of\s+\d+")
SKIP_PAGE_RE = re.compile(r"-\s*(Cover Page|AI Writing Overview|Integrity Overview)\s*$")

CHROME_OLD_RES = [
    re.compile(r"Submission date:.*"),
    re.compile(r"Submission ID:.*"),
    re.compile(r"File name:.*"),
    re.compile(r"Word count:.*"),
    re.compile(r"Character count:.*"),
    re.compile(r"^by\s+\S+.*"),
]

def classify(full_text):
    if AI_SCORE_RE.search(full_text):
        return "ai_report"
    if SIM_NEW_RE.search(full_text):
        return "similarity_only"
    if "ORIGINALITY REPORT" in full_text or "SIMILARITY INDEX" in full_text:
        return "old_format_similarity_only"
    return "unscored_unusable"

def is_junk_line(s):
    """Margin line numbers / extraction artifacts: line with digits only."""
    t = s.replace(" ", "").replace("\t", "")
    return bool(t) and all(ch.isdigit() for ch in t)

def clean_new_format(pages):
    kept = []
    for pg in pages:
        lines = pg.split("\n")
        hdr = lines[0].strip() if lines else ""
        if SKIP_PAGE_RE.search(hdr):
            continue
        body = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if PAGE_HDR_RE.search(s):
                continue
            if "Submission ID trn:oid" in s:
                continue
            if re.fullmatch(r"\d{1,4}", s):  # margin line numbers
                continue
            if is_junk_line(s):  # digit-only artifact lines ("1  1  1")
                continue
            body.append(ln.rstrip())
        kept.append("\n".join(body))
    return "\n\n".join(t for t in kept if t.strip())

def clean_generic(pages):
    """Best-effort text for unscored files: drop page headers, keep the rest."""
    kept = []
    for pg in pages:
        body = []
        for ln in pg.split("\n"):
            s = ln.strip()
            if not s or PAGE_HDR_RE.search(s) or is_junk_line(s):
                continue
            body.append(ln.rstrip())
        t = "\n".join(body).strip()
        if t:
            kept.append(t)
    return "\n\n".join(kept)

def clean_old_format(pages):
    kept = []
    for pg in pages:
        if "ORIGINALITY REPORT" in pg or "GRADEMARK REPORT" in pg:
            break
        body = []
        for ln in pg.split("\n"):
            s = ln.strip()
            if not s or PAGE_HDR_RE.search(s):
                continue
            if any(r.match(s) for r in CHROME_OLD_RES):
                continue
            body.append(ln.rstrip())
        t = "\n".join(body).strip()
        if t:
            kept.append(t)
    return "\n\n".join(kept)

def norm_text(t):
    return re.sub(r"\s+", " ", t.lower()).strip()

def text_hash(t):
    return hashlib.sha1(norm_text(t).encode("utf-8")).hexdigest()

def stem_of(fname):
    s = fname[:-4] if fname.lower().endswith(".pdf") else fname
    s = re.sub(r"\.docx$", "", s, flags=re.I)
    s = re.sub(r"\s*\(\d+\)\s*$", "", s)          # "23626 (1)"
    s = re.sub(r"(_+\d+)+$", "", s)                # "__1___1", "__1", "_2"
    s = re.sub(r"-\d+$", "", s)                   # "-2"
    s = re.sub(r"\s*-\s*Copie\s*$", "", s, flags=re.I)
    return s.strip().lower()

# Verified near-duplicate filename pairs (overlap measured on full text):
#  - 18798.pdf + MIS485_Project_Deliverable-2_AI_Study_Planner.pdf: ratio 0.9545,
#    same AI score 0 -> MERGE into one record.
#  - 23626.pdf + 23626__1___1.pdf: ratio 0.881, DIFFERENT scores (35 vs 41) and
#    different submission IDs (revised submission) -> keep separate, link only.
VERIFIED_MERGE_PAIRS = [
    {"MIS485_Project_Deliverable-2_AI_Study_Planner.pdf", "18798.pdf"},
]
VERIFIED_RELATED_PAIRS = [
    {"23626.pdf", "23626__1___1.pdf"},
]

# ---------------------------------------------------------------- main
def main():
    pdfs = sorted(f for f in os.listdir(FILES_DIR) if f.lower().endswith(".pdf"))
    log(f"found {len(pdfs)} PDFs in {FILES_DIR}")

    raw = []
    for fname in pdfs:
        path = os.path.join(FILES_DIR, fname)
        pages = pdf_text_pages(path)
        rec = {"source_pdf": fname}
        if pages is None:
            rec.update({"extraction_status": "failed", "quality_flags": ["pdftotext_failed"],
                        "class": "unscored_unusable"})
            raw.append(rec)
            log(f"FAILED extraction: {fname}")
            continue
        full = "\n".join(pages)
        cls = classify(full)

        ai_m = AI_SCORE_RE.search(full)
        if ai_m:
            ai_score = int(ai_m.group(1)) if ai_m.group(1) != "*" else None
            ai_band = "*%" if ai_m.group(1) == "*" else None
        else:
            ai_score, ai_band = None, None

        sim = None
        m = SIM_NEW_RE.search(full)
        if m:
            sim = int(m.group(1))
        else:
            m = SIM_OLD_RE.search(full) or SIM_OLD2_RE.search(full)
            if m:
                sim = int(m.group(1))

        subid = None
        m = SUBID_NEW_RE.search(full) or SUBID_OLD_RE.search(full)
        if m:
            subid = m.group(1).strip()

        rdate = None
        if cls != "old_format_similarity_only":
            m = DATE_NEW_RE.search(pages[0] if pages else "")
            if m:
                rdate = m.group(1).strip()
        else:
            m = DATE_OLD_RE.search(full)
            if m:
                rdate = m.group(1).strip()

        if cls == "old_format_similarity_only":
            text = clean_old_format(pages)
        elif cls == "unscored_unusable":
            text = clean_generic(pages)
        else:
            text = clean_new_format(pages)

        words = len(text.split())
        flags = []
        if subid is None:
            flags.append("no_submission_id")
        if rdate is None:
            flags.append("no_report_date")
        if cls in ("ai_report", "similarity_only", "old_format_similarity_only") and words < 50:
            flags.append("short_text_lt_50_words")
        if cls == "ai_report" and ai_score is None and ai_band is None:
            flags.append("score_regex_missed")
        if cls == "similarity_only" and sim is None:
            flags.append("similarity_regex_missed")
        if cls == "old_format_similarity_only" and sim is None:
            flags.append("similarity_regex_missed")
        if cls == "unscored_unusable":
            flags.append("no_recognized_report_format")

        status = "failed" if pages is None else ("partial" if words < 50 else "ok")

        raw.append({
            "source_pdf": fname, "class": cls,
            "turnitin_ai_score": ai_score, "turnitin_ai_band": ai_band,
            "turnitin_similarity": sim, "submission_id": subid,
            "report_date": rdate, "detector_version": None,
            "text": text, "text_words": words, "text_hash": text_hash(text) if text else None,
            "page_count": pdf_page_count(path),
            "quality_flags": flags, "extraction_status": status,
        })
    log(f"extracted {len(raw)} records")

    # ---- dedupe: exact text hash, then filename-stem groups verified by overlap
    groups = []  # list of dicts: members=[raw...], canonical=raw
    by_hash = {}
    for r in raw:
        if r["text_hash"]:
            by_hash.setdefault(r["text_hash"], []).append(r)
    assigned = set()
    gid = 0
    for h, members in by_hash.items():
        if len(members) > 1:
            gid += 1
            members_sorted = sorted(members, key=lambda r: r["source_pdf"])
            groups.append({"doc_group": f"g{gid:04d}", "members": members_sorted,
                           "reason": "identical_text_hash"})
            assigned.update(id(m) for m in members)
    # stem groups for near-dupes
    related_links = collections.defaultdict(set)  # pdf -> set of related pdfs
    by_stem = {}
    for r in raw:
        if id(r) in assigned or not r["text"]:
            continue
        by_stem.setdefault(stem_of(r["source_pdf"]), []).append(r)
    import difflib
    def fast_ratio(a, b):
        # cheap prefilter before the expensive SequenceMatcher
        qa = difflib.SequenceMatcher(None, a[:4000], b[:4000])
        if qa.real_quick_ratio() < 0.70:
            return 0.0
        return difflib.SequenceMatcher(None, a[:8000], b[:8000]).ratio()
    def scores_of(r):
        return (r["turnitin_ai_score"], r["turnitin_ai_band"], r["turnitin_similarity"])
    for stem, members in by_stem.items():
        if len(members) < 2:
            continue
        members_sorted = sorted(members, key=lambda r: r["source_pdf"])
        base = members_sorted[0]
        dupes, rest = [base], []
        for other in members_sorted[1:]:
            a = norm_text(base["text"])
            b = norm_text(other["text"])
            ratio = fast_ratio(a, b)
            if ratio >= 0.85 and scores_of(other) == scores_of(base):
                dupes.append(other)
            else:
                rest.append(other)
                if ratio >= 0.85:
                    # same text, different scores -> related, NOT merged
                    related_links[base["source_pdf"]].add(other["source_pdf"])
                    related_links[other["source_pdf"]].add(base["source_pdf"])
                    other["quality_flags"].append("near_duplicate_different_score")
        if len(dupes) > 1:
            gid += 1
            groups.append({"doc_group": f"g{gid:04d}", "members": dupes,
                           "reason": "same_stem_high_overlap_same_scores"})
            assigned.update(id(m) for m in dupes)
            for o in rest:
                if "near_duplicate_different_score" not in o["quality_flags"]:
                    o["quality_flags"].append("same_stem_different_text")
    for r in raw:
        if id(r) not in assigned:
            gid += 1
            groups.append({"doc_group": f"g{gid:04d}", "members": [r],
                           "reason": "unique"})
            assigned.add(id(r))

    # ---- verified near-duplicate pairs: merge only when overlap>=0.90 AND scores agree
    import difflib as _difflib
    by_pdf = {r["source_pdf"]: r for r in raw}
    def group_of(r):
        return next(g for g in groups if r in g["members"])
    for pair in VERIFIED_MERGE_PAIRS:
        pdfs = [by_pdf.get(p) for p in pair]
        if not all(pdfs):
            log(f"verified pair {pair}: a file is missing, skipped")
            continue
        a, b = pdfs
        ratio = _difflib.SequenceMatcher(
            None, norm_text(a["text"]), norm_text(b["text"])).ratio() \
            if norm_text(a["text"]) and norm_text(b["text"]) else 0.0
        same_score = (a["turnitin_ai_score"], a["turnitin_ai_band"]) == \
                     (b["turnitin_ai_score"], b["turnitin_ai_band"])
        ga, gb = group_of(a), group_of(b)
        if ratio >= 0.90 and same_score and ga is not gb:
            ga["members"] = sorted(ga["members"] + gb["members"],
                                   key=lambda r: r["source_pdf"])
            ga["reason"] = f"verified_near_duplicate_overlap_{ratio:.3f}"
            groups.remove(gb)
            log(f"merged verified pair {sorted(pair)}: overlap={ratio:.3f}")
        else:
            related_links[a["source_pdf"]].add(b["source_pdf"])
            related_links[b["source_pdf"]].add(a["source_pdf"])
            log(f"linked (not merged) {sorted(pair)}: overlap={ratio:.3f} same_score={same_score}")
    for pair in VERIFIED_RELATED_PAIRS:
        pdfs = [by_pdf.get(p) for p in pair]
        if all(pdfs):
            a, b = pdfs
            related_links[a["source_pdf"]].add(b["source_pdf"])
            related_links[b["source_pdf"]].add(a["source_pdf"])
            log(f"linked related pair {sorted(pair)} (kept as separate records)")

    # ---- write JSONL, one record per unique document
    pdf_to_docid = {}
    tmp_ids = {}
    used_ids = set()
    for g in groups:
        base_id = stem_of(sorted(g["members"], key=lambda r: r["source_pdf"])[0]["source_pdf"]) or "doc"
        did, n = base_id, 2
        while did in used_ids:
            did = f"{base_id}__{n}"
            n += 1
        used_ids.add(did)
        tmp_ids[id(g)] = did
        for m in g["members"]:
            pdf_to_docid[m["source_pdf"]] = did
    n_dup_pdfs = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for g in groups:
            members = g["members"]
            canon = members[0]
            if len(members) > 1:
                n_dup_pdfs += len(members) - 1
                scores = {m["turnitin_ai_score"] for m in members}
                bands = {m["turnitin_ai_band"] for m in members}
                sims = {m["turnitin_similarity"] for m in members}
                if len(scores) > 1 or len(bands) > 1 or len(sims) > 1:
                    canon["quality_flags"] = canon["quality_flags"] + ["dupe_score_mismatch"]
            doc_id = tmp_ids[id(g)]
            subids = sorted({m["submission_id"] for m in members if m["submission_id"]})
            rel_pdfs = {p for m in members for p in related_links.get(m["source_pdf"], set())}
            rel = sorted({pdf_to_docid[p] for p in rel_pdfs if p in pdf_to_docid} - {doc_id})
            out = {
                "doc_id": doc_id,
                "source_pdf": canon["source_pdf"],
                "all_source_pdfs": [m["source_pdf"] for m in members],
                "doc_group": g["doc_group"],
                "dedupe_reason": g["reason"],
                "related_doc_ids": rel,
                "report_class": canon["class"],
                "text": canon["text"],
                "text_words": canon["text_words"],
                "turnitin_ai_score": canon["turnitin_ai_score"],
                "turnitin_ai_band": canon["turnitin_ai_band"],
                "turnitin_similarity": canon["turnitin_similarity"],
                "authorship_verified": None,  # the 2 verified cases are .docx, not in these PDFs
                "submission_id": canon["submission_id"],
                "all_submission_ids": subids,
                "report_date": canon["report_date"],
                "detector_version": None,  # not present in these PDFs
                "quality_flags": canon["quality_flags"],
                "extraction_status": canon["extraction_status"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    log(f"wrote {len(groups)} unique-document records -> {OUT_JSONL}")
    log(f"duplicate PDFs folded: {n_dup_pdfs}; duplicate groups: {sum(1 for g in groups if len(g['members']) > 1)}")
    from collections import Counter
    log("class counts (unique docs): " + str(Counter(
        json.loads(l)["report_class"] for l in open(OUT_JSONL, encoding="utf-8"))))
    logf.close()

if __name__ == "__main__":
    main()
