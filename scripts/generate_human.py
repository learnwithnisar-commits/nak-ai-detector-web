#!/usr/bin/env python3
"""Generate 2,300 original human-style text samples for the NAK AI Detector dataset.
Assembles samples from hand-written fragment banks with persona voice passes,
then validates output. Usage: python3 generate_human.py [--n 2300] [--seed 7]
"""
import argparse, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_banks_topics import TOPICS, GENERIC_BITS
from gen_banks_people import (PERSONAS, ANECDOTES, RHETORICALS, FRAGMENTS, ASIDES,
                              TYPOS, NAMES_PK, NAMES_WEST, NAMES_ALL, BRITISH_PAIRS)
from gen_banks_genres import (GENRE_TOPICS, GENRE_PERSONAS, OPENERS, CLOSERS, TRANSITIONS)

GENRES = list(GENRE_TOPICS.keys())
OUT = os.path.expanduser("~/workspace/ai-detector/data/human_samples.jsonl")

def wc(t): return len(t.split())

def pick_name(persona, rng):
    pk_tags = {"pk_student","pk_shopkeeper","pk_retired_teacher","young_mother","pk_engineer",
               "pk_farmer","karachi_girl","cricket_fan","female_teacher","freelancer",
               "grandfather","small_biz","travel_blogger","chef","rickshaw_driver"}
    pool = NAMES_PK if persona in pk_tags else (NAMES_WEST if rng.random() < 0.6 else NAMES_ALL)
    return rng.choice(pool)

def britishize(text, rng):
    for us, uk in BRITISH_PAIRS:
        # only swap some occurrences to look natural
        if us in text and rng.random() < 0.9:
            text = re.sub(r'\b' + us + r'\b', uk, text)
    return text

def voice_pass(text, persona, rng):
    cfg = PERSONAS[persona]
    # sprinkle persona lexicon words occasionally: insert ", word" before a sentence's final punctuation
    if rng.random() < 0.45 and cfg["sprinkle"]:
        w = rng.choice(cfg["sprinkle"])
        sents = re.split(r'(?<=[.!?])\s+', text)
        cands = [j for j in range(1, len(sents) - 1)
                 if re.search(r'[.!?]$', sents[j].strip())
                 and not sents[j].strip().endswith((')', '"', '”'))]
        if cands:
            i = rng.choice(cands)
            s = sents[i].strip()
            sents[i] = s[:-1] + ", " + w + s[-1]
            text = " ".join(sents)
    if cfg["british"]:
        text = britishize(text, rng)
    # rare typo for human imperfection
    if rng.random() < 0.10:
        for good, bad in rng.sample(TYPOS, len(TYPOS)):
            if re.search(r'\b' + re.escape(good) + r'\b', text):
                text = re.sub(r'\b' + re.escape(good) + r'\b', bad, text, count=1)
                break
    # rare doubled space or lowercase start
    if rng.random() < 0.05:
        text = text.replace(". ", ".  ", 1)
    return text

def maybe_spice(paras, rng):
    out = []
    for p in paras:
        out.append(p)
        r = rng.random()
        if r < 0.30:
            out.append(rng.choice(RHETORICALS))
        elif r < 0.42:
            out.append(rng.choice(FRAGMENTS))
        elif r < 0.50:
            # attach aside to end of paragraph
            out[-1] = out[-1] + " " + rng.choice(ASIDES)
    return out

def build_sample(genre, persona, rng):
    name = pick_name(persona, rng)
    topic = rng.choice(GENRE_TOPICS[genre])
    opener = rng.choice(OPENERS[genre]).format(name=name)
    bodies = rng.sample(TOPICS[topic], 3)
    if rng.random() < 0.35:
        bodies.append(rng.choice(GENERIC_BITS))
        rng.shuffle(bodies)
    # persona anecdote: prefer matching, else any
    mine = [a for p, a in ANECDOTES if p == persona]
    anecdote = rng.choice(mine) if mine and rng.random() < 0.7 else rng.choice([a for _, a in ANECDOTES])
    closer = rng.choice(CLOSERS[genre]).format(name=name)

    paras = [opener]
    for i, b in enumerate(bodies):
        if i > 0 and rng.random() < 0.35:
            paras.append(rng.choice(TRANSITIONS) + " " + b)
        else:
            paras.append(b)
    if rng.random() < 0.8:
        paras.append(anecdote)
    paras.append(closer)
    paras = maybe_spice(paras, rng)
    text = "\n\n".join(paras)
    text = voice_pass(text, persona, rng)
    return text, topic

def enforce_length(text, genre, persona, rng):
    # add generic bits until >=120 words; bodies rarely exceed 350
    guard = 0
    while wc(text) < 120 and guard < 6:
        text += "\n\n" + rng.choice(GENERIC_BITS + [a for _, a in ANECDOTES])
        guard += 1
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    per = a.n // len(GENRES)
    counts = {g: per for g in GENRES}
    # distribute remainder
    rem = a.n - per * len(GENRES)
    for g in GENRES[:rem]:
        counts[g] += 1

    seen = set()
    samples = []
    attempts = 0
    order = []
    for g in GENRES:
        order += [g] * counts[g]
    rng.shuffle(order)

    for genre in order:
        persona = rng.choice(GENRE_PERSONAS[genre])
        text, topic = build_sample(genre, persona, rng)
        text = enforce_length(text, genre, persona, rng)
        w = wc(text)
        if not (120 <= w <= 350):
            # trim or pad: if over, drop a middle paragraph
            paras = text.split("\n\n")
            while wc(text) > 350 and len(paras) > 4:
                paras.pop(2)
                text = "\n\n".join(paras)
            w = wc(text)
            if not (120 <= w <= 350):
                continue
        h = hash(text)
        attempts += 1
        if h in seen:
            # regenerate with perturbed rng state
            for _ in range(5):
                persona = rng.choice(GENRE_PERSONAS[genre])
                text, topic = build_sample(genre, persona, rng)
                text = enforce_length(text, genre, persona, rng)
                if 120 <= wc(text) <= 350 and hash(text) not in seen:
                    break
            else:
                continue
            h = hash(text)
        seen.add(h)
        samples.append({"text": text, "label": 0, "genre": genre, "persona": persona})

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ---- validation ----
    errs = []
    gcounts, pcounts = {}, {}
    wmin, wmax = 10**9, 0
    with open(a.out, encoding="utf-8") as f:
        lines = f.readlines()
    texts = set()
    for i, ln in enumerate(lines):
        try:
            o = json.loads(ln)
        except Exception as e:
            errs.append(f"line {i}: bad json {e}"); continue
        if o.get("label") != 0: errs.append(f"line {i}: label != 0")
        w = wc(o.get("text", ""))
        wmin, wmax = min(wmin, w), max(wmax, w)
        if not (120 <= w <= 350): errs.append(f"line {i}: words={w}")
        if not o.get("text", "").strip(): errs.append(f"line {i}: empty")
        if o["text"] in texts: errs.append(f"line {i}: duplicate")
        texts.add(o["text"])
        gcounts[o["genre"]] = gcounts.get(o["genre"], 0) + 1
        pcounts[o["persona"]] = pcounts.get(o["persona"], 0) + 1

    print(f"TOTAL SAMPLES: {len(lines)} (target {a.n})")
    print(f"VALIDATION ERRORS: {len(errs)}")
    for e in errs[:10]: print("  ERR:", e)
    print(f"WORD RANGE: {wmin}-{wmax} (required 120-350)")
    print("GENRE COUNTS:")
    for g in GENRES: print(f"  {g}: {gcounts.get(g,0)}")
    print("PERSONA COUNTS:")
    for p, c in sorted(pcounts.items(), key=lambda x: -x[1]): print(f"  {p}: {c}")
    print(f"OUTPUT: {a.out}")

if __name__ == "__main__":
    main()
