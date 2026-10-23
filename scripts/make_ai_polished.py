#!/usr/bin/env python3
"""Rule-based AI-polish simulator (zero-budget ChatGPT-polish proxy).

Reads data/polish_pairs.jsonl [{doc_id, source, title, original}] and writes
data/ai_polished_train.jsonl with BOTH:
  label 0 (human): the original abstract
  label 1 (AI)   : the AI-polished version

Design: reproduce the *statistical signature* of ChatGPT academic polishing
(higher lexical predictability via standard academic collocations, uniform
discourse markers, confident framing, em-dash parentheticals, reduced
burstiness) while staying semantically faithful and moderate — NOT a
caricature. All randomness seeded per doc_id (reproducible).

Validation (run after): polished versions should show lower mean trigram
surprisal than originals on the human LM, and stay within +/-15% word count.
"""
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

# ---- word-level formalizations (human -> ChatGPT-preferred) ----
# (human, ai, probability). Word-boundary, case-preserving.
LEX_TRIPLES = [
    ("use", "utilize", 0.45), ("uses", "utilizes", 0.45),
    ("used", "utilized", 0.45), ("using", "utilizing", 0.25),
    ("help", "facilitate", 0.5), ("helps", "facilitates", 0.5),
    ("helped", "facilitated", 0.5), ("helpful", "instrumental", 0.4),
    ("show", "demonstrate", 0.5), ("shows", "demonstrates", 0.55),
    ("showed", "demonstrated", 0.5), ("shown", "demonstrated", 0.4),
    ("big", "significant", 0.5), ("small", "modest", 0.4),
    ("large", "substantial", 0.4),
    ("get", "obtain", 0.5), ("gets", "obtains", 0.5),
    ("got", "obtained", 0.5), ("getting", "obtaining", 0.4),
    ("find", "identify", 0.45), ("finds", "identifies", 0.45),
    ("found", "identified", 0.4),
    ("deal with", "address", 0.5), ("dealing with", "addressing", 0.5),
    ("deals with", "addresses", 0.5),
    ("very", "highly", 0.5), ("really", "substantially", 0.4),
    ("also", "additionally", 0.35),
    ("try to", "seek to", 0.5), ("tries to", "seeks to", 0.5),
    ("trying to", "seeking to", 0.4),
    ("need", "require", 0.45), ("needs", "requires", 0.45),
    ("needed", "required", 0.45),
    ("important", "crucial", 0.45), ("importantly", "crucially", 0.4),
    ("different", "diverse", 0.4),
    ("many", "numerous", 0.4), ("much", "considerable", 0.35),
    ("new", "novel", 0.4),
    ("good", "robust", 0.35),
    ("improve", "enhance", 0.5), ("improves", "enhances", 0.5),
    ("improved", "enhanced", 0.5), ("improvement", "enhancement", 0.45),
    ("improvements", "enhancements", 0.45),
    ("think", "contend", 0.4),
    ("say", "state", 0.4), ("says", "states", 0.4),
    ("talk about", "discuss", 0.5), ("talks about", "discusses", 0.5),
    ("seem", "appear", 0.4), ("seems", "appears", 0.4),
    ("maybe", "potentially", 0.5), ("perhaps", "potentially", 0.4),
    ("main", "primary", 0.45), ("mainly", "primarily", 0.45),
    ("mostly", "predominantly", 0.4),
    ("often", "frequently", 0.45), ("usually", "typically", 0.45),
    ("sometimes", "occasionally", 0.4),
    ("quickly", "rapidly", 0.45), ("slowly", "gradually", 0.4),
    ("clearly", "evidently", 0.4),
    ("basically", "fundamentally", 0.5), ("actually", "indeed", 0.35),
    ("just", "merely", 0.4), ("quite", "rather", 0.35),
    ("thing", "factor", 0.4), ("things", "factors", 0.4),
    ("part", "component", 0.35), ("parts", "components", 0.35),
    ("whole", "entire", 0.45),
    ("job", "role", 0.4),
    ("buy", "purchase", 0.5),
    ("end", "conclude", 0.3), ("ends", "concludes", 0.3),
    ("start", "commence", 0.25), ("starts", "commences", 0.25),
    ("bad", "suboptimal", 0.4),
    ("a lot of", "a considerable number of", 0.5),
    ("lots of", "numerous", 0.5),
    ("kind of", "rather", 0.4), ("sort of", "rather", 0.4),
]

PHRASE_TRIPLES = [
    ("in order to", "to", 0.7),
    ("due to the fact that", "as", 0.7),
    ("a number of", "several", 0.5),
    ("plays an important role in", "is pivotal to", 0.6),
    ("play an important role in", "are pivotal to", 0.6),
    ("the results show", "the findings demonstrate", 0.6),
    ("results show", "findings demonstrate", 0.6),
    ("this paper", "this study", 0.6),
    ("we show", "this study demonstrates", 0.55),
    ("we demonstrate", "this study demonstrates", 0.55),
    ("we found", "the findings revealed", 0.5),
    ("we propose", "this study proposes", 0.5),
    ("our results", "the findings", 0.55),
    ("our study", "this study", 0.55),
    ("our findings", "the findings", 0.55),
    ("shows that", "underscores that", 0.45),
    ("show that", "underscore that", 0.45),
    ("highlights", "underscores", 0.4),
    ("emphasizes", "underscores", 0.4),
    ("in the end", "ultimately", 0.6),
    ("first of all", "firstly", 0.6),
]

HEDGE_TRIPLES = [
    ("might", "may", 0.4),
    ("suggests", "indicates", 0.4),
    ("suggest", "indicate", 0.4),
    ("possibly", "potentially", 0.4),
]

MARKERS = ["Moreover,", "Furthermore,", "Additionally,", "Notably,",
           "Importantly,", "Consequently,", "In particular,", "Specifically,"]

CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "couldn't": "could not", "won't": "will not",
    "wouldn't": "would not", "shouldn't": "should not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "it's": "it is",
}


def _preserve_case(src, dst):
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


def sub_word(text, human, ai, p, rng):
    """Replace occurrences of human->ai with probability p each."""
    pat = re.compile(r"\b" + re.escape(human) + r"\b", re.IGNORECASE)

    def repl(m):
        if rng.random() < p:
            return _preserve_case(m.group(0), ai)
        return m.group(0)

    return pat.sub(repl, text)


def split_sentences(text):
    # simple sentence splitter (good enough for abstracts)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def polish(text, seed):
    rng = random.Random(seed)
    # 0. expand contractions
    for c, e in CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(c) + r"\b", e, text,
                      flags=re.IGNORECASE)

    # 1. phrase-level upgrades
    for human, ai, p in PHRASE_TRIPLES:
        text = sub_word(text, human, ai, p, rng)
    # 2. word-level upgrades
    for human, ai, p in LEX_TRIPLES:
        text = sub_word(text, human, ai, p, rng)
    # 3. hedge -> confident
    for human, ai, p in HEDGE_TRIPLES:
        text = sub_word(text, human, ai, p, rng)

    # 4. em-dash parentheticals (ChatGPT signature; skip citation-like parens)
    def paren_repl(m):
        inner = m.group(1)
        if re.search(r"\d", inner):
            return m.group(0)
        if rng.random() < 0.4:
            return " \u2014 " + inner.strip() + " \u2014 "
        return m.group(0)

    text = re.sub(r"\(([^()]{4,90})\)", paren_repl, text)

    # 5. sentence-level: discourse markers ONLY (no combining — combining
    #    creates a structural artifact the model latches onto instead of style)
    sents = split_sentences(text)
    out = []
    markers = MARKERS[:]
    rng.shuffle(markers)
    mi = 0
    for i, s in enumerate(sents):
        if not s:
            continue
        # discourse marker
        if (i > 0 and mi < len(markers) and rng.random() < 0.35
                and not re.match(r"^(Moreover|Furthermore|Additionally|"
                                 r"Notably|Importantly|Consequently|In |"
                                 r"Specifically|However|Therefore|Thus|"
                                 r"Although|While|Despite|Moreover)\b",
                                 s, re.IGNORECASE)):
            s = markers[mi] + " " + s[0].lower() + s[1:]
            mi += 1
        out.append(s)
    text = " ".join(out)
    # tidy whitespace / double punctuation
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def main():
    pairs = [json.loads(l) for l in
             open(os.path.join(DATA, "polish_pairs.jsonl"), encoding="utf-8")
             if l.strip()]
    out_path = os.path.join(DATA, "ai_polished_train.jsonl")
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in pairs:
            orig = r["original"].strip()
            seed = int(hashlib.md5(r["doc_id"].encode()).hexdigest(), 16) % (2 ** 31)
            pol = polish(orig, seed)
            # sanity: polished must differ but stay close in length
            if pol == orig:
                continue
            wr = len(pol.split()) / max(1, len(orig.split()))
            if not (0.80 <= wr <= 1.25):
                continue
            f.write(json.dumps({
                "text": orig, "label": 0, "source": "ai_polish_pair",
                "genre": "academic_abstract", "doc_id": r["doc_id"],
                "side": "human",
            }, ensure_ascii=False) + "\n")
            f.write(json.dumps({
                "text": pol, "label": 1, "source": "ai_polish_pair",
                "genre": "academic_abstract", "doc_id": r["doc_id"],
                "side": "ai_polished",
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {out_path}: {n} pairs ({2*n} rows)", flush=True)


if __name__ == "__main__":
    main()
