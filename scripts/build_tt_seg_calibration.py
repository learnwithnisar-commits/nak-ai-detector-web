#!/usr/bin/env python3
"""Build SEGMENT-scale Turnitin-distilled calibration.

Same as build_tt_calibration.py but labels overlapping 8-sentence segments
(stride 4) — the exact scale the app scores at. A segment is labeled 1 if
>=40% of its words are Turnitin-flagged, else 0.

Output: data/calibration/segments_tt.jsonl
"""
import json
import os
import re
import difflib
import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "data", "calibration")
os.makedirs(OUTDIR, exist_ok=True)
FILES = "/home/hatch/workspace/user/files"
HL = (0.32, 0.777, 0.855)
WIN, STRIDE = 8, 4

def norm_words(t):
    return [re.sub(r'[^a-z0-9]', '', w.lower()) for w in t.split()
            if re.sub(r'[^a-z0-9]', '', w.lower())]

def is_hl(col):
    return col and len(col) == 3 and all(abs(col[i] - HL[i]) < 0.05 for i in range(3))

def turnitin_word_flags(pdf_path, skip_pages):
    doc = fitz.open(pdf_path)
    words, flags = [], []
    for pno in range(skip_pages, len(doc)):
        page = doc[pno]
        rects = [d['rect'] for d in page.get_drawings() if is_hl(d.get('fill'))]
        for w in page.get_text('words'):
            nw = norm_words(w[4])
            if not nw:
                continue
            wr = fitz.Rect(w[:4])
            fl = any((wr & r).get_area() > 0.3 * wr.get_area()
                     for r in rects if wr.intersects(r))
            words.append(nw[0])
            flags.append(fl)
    return words, flags

def build_segments(doc_key, sents, pdf_path, skip_pages, tt_doc_pct):
    our_words = [norm_words(s) for s in sents]
    pdf_words, pdf_flag = turnitin_word_flags(pdf_path, skip_pages)
    ours_flat = [w for s in our_words for w in s]
    sm = difflib.SequenceMatcher(None, ours_flat, pdf_words, autojunk=False)
    mapping = {}
    for a0, b0, n in sm.get_matching_blocks():
        for k in range(n):
            mapping[a0 + k] = b0 + k
    # per-sentence flagged fractions
    sent_flag = []
    wi = 0
    for ow in our_words:
        flags = [pdf_flag[mapping[wi + k]] for k in range(len(ow)) if (wi + k) in mapping]
        wi += len(ow)
        sent_flag.append(sum(flags) / len(flags) if flags else 0.0)
    # segments
    recs = []
    for start in range(0, max(1, len(sents) - WIN + 1), STRIDE):
        seg = sents[start:start + WIN]
        if len(" ".join(seg).split()) < 60:
            continue
        frac = sum(sent_flag[start:start + WIN]) / len(seg)
        lab = 1 if frac >= 0.4 else 0
        recs.append({"text": " ".join(seg), "label": lab, "genre": "tt_distilled_seg",
                     "source": "turnitin_distill", "doc": doc_key,
                     "tt_frac": round(frac, 3), "tt_doc_pct": tt_doc_pct})
    # tail
    if len(sents) > WIN:
        seg = sents[-WIN:]
        frac = sum(sent_flag[-WIN:]) / WIN
        lab = 1 if frac >= 0.4 else 0
        recs.append({"text": " ".join(seg), "label": lab, "genre": "tt_distilled_seg",
                     "source": "turnitin_distill", "doc": doc_key,
                     "tt_frac": round(frac, 3), "tt_doc_pct": tt_doc_pct})
    return recs

def main():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "app"))
    from scan import split_sentences
    import docx
    all_recs = []
    docs = [
        ("listeria", "Listeria_goat_MANUSCRIPT_IMPROVED_CLEAN.docx",
         "Listeria_goat_MANUSCRIPT_IMPROVED_CLEAN__1.pdf", 3, 54),
        ("aigc_bicycle", "aigc_download_2027795315876175872_aigc_rate.docx",
         "aigc_download_2027795315876175872_aigc_rate.pdf", 2, 43),
        ("xdr_typhoid", "XDR_Typhoid_Pakistan_PJMR_Format.docx",
         "XDR__1.pdf", 2, 64),
    ]
    for key, docx_name, pdf_name, skip, pct in docs:
        d = docx.Document(os.path.join(FILES, docx_name))
        text = "\n".join(p.text.strip() for p in d.paragraphs if p.text.strip())
        sents = split_sentences(text)
        recs = build_segments(key, sents, os.path.join(FILES, pdf_name), skip, pct)
        npos = sum(r["label"] for r in recs)
        print(f"{key}: {len(recs)} segments, {npos} positive")
        all_recs.extend(recs)
    out = os.path.join(OUTDIR, "segments_tt.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"TOTAL {len(all_recs)} -> {out}")

if __name__ == "__main__":
    main()
