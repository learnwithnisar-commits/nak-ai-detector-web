#!/usr/bin/env python3
"""Extract real formal human prose from system man pages (groff source).

Man pages are human-written, formal, technical English — a zero-budget proxy
for the academic/formal prose Nisar's users submit. Output samples of
120-350 words, prose-filtered (option lists / verbatim blocks removed).

Split: 300 -> data/real_human_train.jsonl (augment human training),
       100 -> data/real_human_eval.jsonl  (HELD-OUT: never trained on,
              used only to measure real-formal false-positive rate).
"""
import gzip
import json
import os
import random
import re

MAN_DIRS = ["/usr/share/man/man1", "/usr/share/man/man7", "/usr/share/man/man8"]
OUT_TRAIN = os.path.expanduser("~/workspace/ai-detector/data/real_human_train.jsonl")
OUT_EVAL = os.path.expanduser("~/workspace/ai-detector/data/real_human_eval.jsonl")
SEED = 20261012

INLINE_ESC = [
    (r"\\f[BIPR]", ""), (r"\\f\(..", ""), (r"\\\*\(..", ""),
    (r"\\\((..)", ""), (r"\\-", "-"), (r"\\&", ""), (r"\\e", "\\\\"),
    (r"\\ ", " "), (r"\\\|", ""), (r"\\~", " "), (r"\\:", ""),
]
INLINE_ESC = [(re.compile(a), b) for a, b in INLINE_ESC]


def clean_inline(s):
    for rx, rep in INLINE_ESC:
        s = rx.sub(rep, s)
    s = s.replace('"', "")
    return s.strip()


def page_text(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    out, skip_verbatim, para = [], False, []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith('.\\"') or s.startswith('\\"'):
            continue
        if s.startswith("."):
            m = re.match(r"^\.([A-Za-z]+)\s?(.*)$", s)
            if not m:
                continue
            mac, arg = m.group(1), m.group(2)
            if mac in ("Vb", "nf"):
                skip_verbatim = True
                continue
            if mac in ("Ve", "fi"):
                skip_verbatim = False
                continue
            if skip_verbatim:
                continue
            if mac in ("TH", "de", "ie", "el", "if", "ds", "nr", "so",
                       "IX", "Sp", "ne", "ft", "in", "ti", "br", "sp"):
                continue
            if mac in ("SH", "SS"):
                # section headers: keep as paragraph break, drop the header
                if para:
                    out.append(" ".join(para))
                    para = []
                continue
            if mac in ("TP", "IP", "PP", "LP", "P", "RS", "RE"):
                if para:
                    out.append(" ".join(para))
                    para = []
                continue
            if mac in ("B", "I"):
                para.append(clean_inline(arg))
            elif mac in ("BR", "BI", "IB", "IR", "RI", "RB"):
                # quoted segments keep their spacing; crude join is fine
                para.append(clean_inline(re.sub(r'"\s+"', " ", arg)))
            # other macros ignored
        else:
            if skip_verbatim:
                continue
            if s.startswith("'"):  # control line
                continue
            para.append(clean_inline(ln))
    if para:
        out.append(" ".join(para))
    return [p for p in out if p]


def prose_ok(p):
    w = p.split()
    if not (20 <= len(w)):
        return False
    # reject option lists / synopsis fragments
    dash_lines = sum(1 for ln in p.split("\n") if ln.strip().startswith("-"))
    if dash_lines > 2:
        return False
    if len(re.findall(r"[A-Z]{4,}", p)) > len(w) * 0.15:
        return False
    if p.count("|") > 3 or p.count("=") > 5:
        return False
    letters = sum(c.isalpha() for c in p)
    if letters < len(p) * 0.5:
        return False
    return True


def main():
    rng = random.Random(SEED)
    paths = []
    for d in MAN_DIRS:
        if os.path.isdir(d):
            paths += [os.path.join(d, f) for f in os.listdir(d)
                      if not re.search(r"\.(pl|da|de|es|fi|fr|hr|hu|id|it|ja|ko|nl|pl|pt|ro|ru|sl|sr|sv|tr|uk|zh)", f)]
    rng.shuffle(paths)
    samples, seen = [], set()
    for path in paths:
        try:
            paras = [p for p in page_text(path) if prose_ok(p)]
        except Exception:
            continue
        # greedily pack paragraphs into 120-350 word samples
        cur, curw = [], 0
        for p in paras:
            w = len(p.split())
            if curw + w > 350 and curw >= 120:
                t = " ".join(cur)
                if t not in seen:
                    seen.add(t)
                    samples.append(t)
                cur, curw = [], 0
            cur.append(p)
            curw += w
        if curw >= 120:
            t = " ".join(cur)
            if t not in seen:
                seen.add(t)
                samples.append(t)
        if len(samples) >= 420:
            break
    print(f"extracted {len(samples)} real-formal samples")
    rng.shuffle(samples)
    # hard cap: truncate to 350 words, drop <120
    capped = []
    for t in samples:
        w = t.split()
        if len(w) >= 120:
            capped.append(" ".join(w[:350]))
    samples = capped
    train_s, eval_s = samples[:300], samples[300:400]
    for path, subset in ((OUT_TRAIN, train_s), (OUT_EVAL, eval_s)):
        with open(path, "w", encoding="utf-8") as f:
            for t in subset:
                f.write(json.dumps({"text": t, "label": 0,
                                    "genre": "formal_prose"}) + "\n")
        print(f"wrote {len(subset)} -> {path}")
    # word-count sanity
    for name, subset in (("train", train_s), ("eval", eval_s)):
        ws = [len(t.split()) for t in subset]
        print(f"{name}: n={len(subset)} words min={min(ws)} max={max(ws)}")


if __name__ == "__main__":
    main()
