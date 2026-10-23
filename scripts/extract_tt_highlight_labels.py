#!/usr/bin/env python3
"""Extract Turnitin sentence-highlight labels from mixed AI-class reports.

Method (verified approach):
- For each Turnitin AI report with 20-79% AI score, open with PyMuPDF.
- On "AI Writing Submission" pages, read vector drawings and keep fills
  with color ~= (0.32, 0.777, 0.855) -- Turnitin's cyan highlight color.
- Map each highlight rect to the words it covers.
- Sentence segmentation from reading-order word stream (split on . ! ?).
- Sentence label: AI if >=50% of its words are covered by highlights,
  else human.

Outputs a JSON list of {pdf, sentence_text, label, turnitin_pct}.
Sanity metrics printed per doc: word->highlight match rate and
flagged-word fraction vs report AI%.

Usage:
    python extract_tt_highlight_labels.py [--limit N] [--out PATH]
"""
import argparse
import json
import os
import re
import sys

import fitz  # PyMuPDF

TARGET = (0.32, 0.777, 0.855)
COLOR_TOL = 0.06

SENT_END = re.compile(r"[.!?]+[\"'\u201d\u2019)\]]*\s*$")
MIN_SENT_WORDS = 5  # skip headers/fragments


def color_close(c1, c2, tol=COLOR_TOL):
    try:
        return all(abs(a - b) <= tol for a, b in zip(c1, c2))
    except TypeError:
        return False


def is_highlight_draw(d):
    """A filled rect in (approx) Turnitin cyan."""
    fill = d.get("fill")
    if fill is None:
        return False
    rect = d.get("rect")
    if rect is None or rect.width < 3 or rect.height < 2:
        return False
    return color_close(fill, TARGET)


def words_covered(word_rect, rects):
    """True if highlight rects cover >=50% of the word's bbox area."""
    wa = word_rect.width * word_rect.height
    if wa <= 0:
        return False
    for r in rects:
        inter = word_rect & r
        if inter.width > 0 and inter.height > 0:
            ia = inter.width * inter.height
            if ia >= 0.5 * wa:
                return True
    return False


def split_sentences(words):
    """Group reading-order word tuples (x0,y0,x1,y1,text,block,line) into sentences."""
    sents, cur = [], []
    for w in words:
        cur.append(w)
        t = w[4]
        if SENT_END.search(t) and len(cur) >= 3:
            sents.append(cur)
            cur = []
    if len(cur) >= MIN_SENT_WORDS:
        sents.append(cur)
    return [s for s in sents if len(s) >= MIN_SENT_WORDS]


def extract_pdf(path):
    """Return per-page lists: {page_no: (highlight_rects, words)}.

    Only real submission pages are used (page header
    "Page N of M - AI Writing Submission"); the "AI Writing Overview"
    page is excluded even though its explanatory text mentions the
    phrase. Highlight rects are matched to words on the SAME page only
    (page coordinates repeat across pages).
    """
    doc = fitz.open(path)
    pages = {}  # pno -> (list[Rect], list[(x0,y0,x1,y1,text)])
    for pno in range(doc.page_count):
        page = doc[pno]
        text = page.get_text()
        # Page header is literally "Page 3 of 8 - AI Writing Submission".
        # The overview page ("AI Writing Overview") also mentions the
        # phrase in body text, so match the header with the hyphen.
        if "- AI Writing Submission" not in text:
            continue
        rects = []
        for d in page.get_drawings():
            if is_highlight_draw(d):
                rects.append(d["rect"])
        words = []
        for w in page.get_text("words"):
            words.append((w[0], w[1], w[2], w[3], w[4]))
        pages[pno] = (rects, words)
    doc.close()
    return pages


def process(path, turnitin_pct):
    pages = extract_pdf(path)
    records = []
    total_words = 0
    total_flagged = 0
    ai_ct = hu_ct = 0
    for pno, (rects, words) in pages.items():
        if not words:
            continue
        flags = []
        for w in words:
            r = fitz.Rect(w[0], w[1], w[2], w[3])
            cov = words_covered(r, rects)
            flags.append(cov)
            total_words += 1
            if cov:
                total_flagged += 1
        sents = split_sentences(words)
        idx = 0
        for s in sents:
            n = len(s)
            nflag = sum(1 for j in range(idx, idx + n) if flags[j])
            label = "ai" if nflag / n >= 0.5 else "human"
            records.append({
                "pdf": os.path.basename(path),
                "sentence_text": " ".join(w[4] for w in s),
                "label": label,
                "turnitin_pct": turnitin_pct,
            })
            if label == "ai":
                ai_ct += 1
            else:
                hu_ct += 1
            idx += n
    flagged_frac = total_flagged / total_words if total_words else 0.0
    return records, flagged_frac, (ai_ct, hu_ct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", required=True,
                    help="JSON file: list of {path, turnitin_pct}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.pdfs) as f:
        items = json.load(f)
    if args.limit:
        items = items[:args.limit]

    all_recs = []
    tot_ai = tot_hu = 0
    print(f"{'pdf':45s} {'TT%':>4s} {'words':>6s} {'flag%':>6s} {'ai_sent':>8s} {'hu_sent':>8s}")
    for it in items:
        path = it["path"]
        pct = it["turnitin_pct"]
        try:
            recs, ffrac, (a, h) = process(path, pct)
        except Exception as e:
            print(f"{os.path.basename(path):45s} ERROR {e}")
            continue
        nwords = sum(len(r["sentence_text"].split()) for r in recs) or 1
        print(f"{os.path.basename(path):45s} {pct:>4d} {nwords:>6d} {ffrac*100:>5.1f}% {a:>8d} {h:>8d}")
        all_recs.extend(recs)
        tot_ai += a
        tot_hu += h

    print(f"\nTOTAL: {tot_ai} AI sentences, {tot_hu} human sentences "
          f"({tot_ai + tot_hu} total)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_recs, f, ensure_ascii=False, indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
