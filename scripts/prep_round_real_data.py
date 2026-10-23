#!/usr/bin/env python3
"""Round prep: build targeted training data from REAL assets only.

1. round_thesis_train.jsonl   label 0 — real Pakistani student thesis prose
   (51.8k words, verified human), chunked ~180 words. Strengthens the human
   class on the manuscript register the anchors live in.

2. round_polish_train.jsonl   label 1 — REAL ChatGPT polish pairs from Nisar.
   Only the *changed* polished sentences (difflib similarity < 0.85 vs the
   best-matching original sentence), grouped into ~120-word chunks, duplicated
   x10 (linear-model sample weighting: 10 real pairs are honest but tiny).
   This is the real AI-polish signal the simulator caricature failed to teach.
   NOTE: the 'original' drafts are also ChatGPT-written, so they are NOT used
   as human negatives — only the polish delta is extracted.

3. round_nonprose_train.jsonl label 0 — non-prose segments (tables, figure/
   table captions, formula-like lines, reference entries) extracted from the
   4 manuscript anchors, duplicated x5. Teaches the model that non-prose
   artifact text is not an AI signal (it currently scores ~100% AI).

Writes JSONL rows {"text": ..., "label": ...}.
Usage: .venv/bin/python scripts/prep_round_real_data.py
"""
import difflib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from app.scan import split_sentences  # noqa: E402

DATA = os.path.join(ROOT, "data")
FILES = "/home/hatch/workspace/user/files"

CHANGED_SIM = 0.85
POLISH_DUP = 10   # sample-weight amplification for the 10 real pairs
NONPROSE_DUP = 5


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"text": r[0], "label": r[1]},
                               ensure_ascii=False) + "\n")
    print(f"wrote {path}: {len(rows)} rows "
          f"(label1={sum(r[1] for r in rows)}, words={sum(len(r[0].split()) for r in rows)})")


def chunk_sentences(sentences, target_words=180):
    chunks, cur, curw = [], [], 0
    for s in sentences:
        w = len(s.split())
        if curw + w > target_words * 1.5 and cur:
            chunks.append(" ".join(cur))
            cur, curw = [], 0
        cur.append(s)
        curw += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ---- 1. thesis prose (human) ----
def build_thesis():
    rows = []
    d = os.path.join(DATA, "real_human", "theses")
    for fn in sorted(os.listdir(d)):
        if not fn.endswith("_prose.txt"):
            continue
        text = open(os.path.join(d, fn), encoding="utf-8").read()
        sents = [s for s in split_sentences(text) if len(s.split()) >= 5]
        for ch in chunk_sentences(sents, 180):
            if len(ch.split()) >= 60:
                rows.append((ch, 0))
    write_jsonl(os.path.join(DATA, "round_thesis_train.jsonl"), rows)


# ---- 2. real polish pairs: changed polished sentences (AI) ----
def changed_polished(original, polished):
    osents = split_sentences(original)
    psents = split_sentences(polished)
    changed = []
    for ps in psents:
        if len(ps.split()) < 5:
            continue
        best = max((difflib.SequenceMatcher(None, ps, o).ratio()
                    for o in osents), default=0.0)
        if best < CHANGED_SIM:
            changed.append(ps)
    return changed


def build_polish():
    rows = []
    pairs = [json.loads(l) for l in
             open(os.path.join(DATA, "real_polish_pairs", "pairs.jsonl"),
                  encoding="utf-8") if l.strip()]
    n_changed = 0
    for p in pairs:
        changed = changed_polished(p["original"], p["polished"])
        n_changed += len(changed)
        for ch in chunk_sentences(changed, 120):
            if len(ch.split()) >= 40:
                for _ in range(POLISH_DUP):
                    rows.append((ch, 1))
    print(f"  changed polished sentences across pairs: {n_changed}")
    write_jsonl(os.path.join(DATA, "round_polish_train.jsonl"), rows)


# ---- 3. non-prose negatives from manuscript anchors ----
ANCHOR_DOCS = [
    "Listeria_goat_MANUSCRIPT_IMPROVED_CLEAN.docx",
    "XDR_Typhoid_Pakistan_PJMR_Format.docx",
    "aigc_download_2027795315876175872_aigc_rate.docx",
]
CAPTION_RE = re.compile(r"^\s*(figure|fig\.?|table|scheme|plate|graph)\b", re.I)
FORMULA_RE = re.compile(r"[=<>±×÷∑∏∫√α-ωΑ-Ω]|\(\d+\)$|\b\d+\s*[+\-*/^]\s*\d+")


def is_nonprose_line(line):
    s = line.strip()
    if not s:
        return False
    if CAPTION_RE.match(s):
        return True
    toks = s.split()
    digits = sum(1 for t in toks if re.search(r"\d", t))
    if digits >= 3 and digits / max(len(toks), 1) > 0.35:
        return True
    if FORMULA_RE.search(s) and len(toks) < 25:
        return True
    return False


def build_nonprose():
    from docx import Document
    rows = []
    for fn in ANCHOR_DOCS:
        path = os.path.join(FILES, fn)
        if not os.path.exists(path):
            print(f"  skip missing {fn}")
            continue
        doc = Document(path)
        segs = []
        # table cells
        for tbl in doc.tables:
            cells = [c.text.strip() for row in tbl.rows for c in row.cells
                     if c.text.strip()]
            if cells:
                segs.append(" | ".join(cells))
        # body lines: captions, formula-like, references
        in_refs = False
        ref_buf = []
        for para in doc.paragraphs:
            s = para.text.strip()
            if not s:
                continue
            if re.match(r"^references?\s*$", s, re.I):
                in_refs = True
                continue
            if in_refs:
                if re.match(r"^\[\d+\]|^\d+\.\s", s) or len(s.split()) > 8:
                    ref_buf.append(s)
                if len(ref_buf) >= 3:
                    segs.append(" ".join(ref_buf))
                    ref_buf = []
                continue
            if is_nonprose_line(s):
                segs.append(s)
        for seg in segs:
            if len(seg.split()) >= 15:
                for _ in range(NONPROSE_DUP):
                    rows.append((seg, 0))
    print(f"  non-prose segments: {len(rows)//NONPROSE_DUP}")
    write_jsonl(os.path.join(DATA, "round_nonprose_train.jsonl"), rows)


if __name__ == "__main__":
    build_thesis()
    build_polish()
    build_nonprose()
