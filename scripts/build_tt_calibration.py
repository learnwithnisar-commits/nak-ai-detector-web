#!/usr/bin/env python3
"""Build Turnitin-distilled calibration sets from the two anchor documents.

For each document:
  - our sentences (app's split_sentences) + our sentence scores
  - Turnitin's sentence labels via word-level highlight alignment
    (label 1 if >=50% of sentence words highlighted by Turnitin)

Output:
  data/calibration/listeria_tt.jsonl
  data/calibration/aigc_tt.jsonl
  {text, label (Turnitin), genre, source, doc}

These are Turnitin's judgments distilled as training signal — NOT ground
truth. Documented limitation: Turnitin may be wrong on some sentences.
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

def build(doc_key, sents, pdf_path, skip_pages, out_name, tt_doc_pct):
    our_words = [norm_words(s) for s in sents]
    pdf_words, pdf_flag = turnitin_word_flags(pdf_path, skip_pages)
    ours_flat = [w for s in our_words for w in s]
    sm = difflib.SequenceMatcher(None, ours_flat, pdf_words, autojunk=False)
    mapping = {}
    for a0, b0, n in sm.get_matching_blocks():
        for k in range(n):
            mapping[a0 + k] = b0 + k
    print(f"{doc_key}: matched {len(mapping)}/{len(ours_flat)} our words")

    recs, wi = [], 0
    n_pos = 0
    for s, ow in zip(sents, our_words):
        n = len(ow)
        flags = [pdf_flag[mapping[wi + k]] for k in range(n) if (wi + k) in mapping]
        wi += n
        if not flags or len(" ".join(s).split()) < 4:
            continue  # skip unmapped / tiny fragments
        lab = 1 if sum(flags) / len(flags) >= 0.5 else 0
        n_pos += lab
        recs.append({"text": s, "label": lab, "genre": "tt_distilled",
                     "source": "turnitin_distill", "doc": doc_key,
                     "tt_doc_pct": tt_doc_pct})
    out = os.path.join(OUTDIR, out_name)
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{doc_key}: {len(recs)} sentences, {n_pos} TT-positive -> {out}")
    return recs

def main():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "app"))
    from scan import split_sentences
    import docx

    # Doc 1: Listeria manuscript (Turnitin 54%)
    d = docx.Document(os.path.join(FILES, "Listeria_goat_MANUSCRIPT_IMPROVED_CLEAN.docx"))
    text = "\n".join(p.text.strip() for p in d.paragraphs if p.text.strip())
    sents = split_sentences(text)
    build("listeria", sents,
          os.path.join(FILES, "Listeria_goat_MANUSCRIPT_IMPROVED_CLEAN__1.pdf"),
          skip_pages=3, out_name="listeria_tt.jsonl", tt_doc_pct=54)

    # Doc 2: Giant Bicycle case study (Turnitin 43%)
    d = docx.Document(os.path.join(FILES, "aigc_download_2027795315876175872_aigc_rate.docx"))
    text = "\n".join(p.text.strip() for p in d.paragraphs if p.text.strip())
    sents = split_sentences(text)
    build("aigc_bicycle", sents,
          os.path.join(FILES, "aigc_download_2027795315876175872_aigc_rate.pdf"),
          skip_pages=2, out_name="aigc_tt.jsonl", tt_doc_pct=43)

    # Doc 3: XDR Typhoid manuscript (Turnitin 64%)
    d = docx.Document(os.path.join(FILES, "XDR_Typhoid_Pakistan_PJMR_Format.docx"))
    text = "\n".join(p.text.strip() for p in d.paragraphs if p.text.strip())
    sents = split_sentences(text)
    build("xdr_typhoid", sents,
          os.path.join(FILES, "XDR__1.pdf"),
          skip_pages=2, out_name="xdr_tt.jsonl", tt_doc_pct=64)

if __name__ == "__main__":
    main()
