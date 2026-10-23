#!/usr/bin/env python3
"""Trigram language model (Laplace smoothing) trained on the HUMAN corpus.

Used only as a *feature generator*: per-text predictability statistics
(mean token log-prob, spread, fraction of highly-predictable tokens,
surprisal burstiness, …). AI text tends to pick probable next words;
human writing is spikier.

The fitted LM is stored inside models/feature_artifacts.joblib under the
"lm" key and loaded lazily by src/features.py — no separate file, so the
serving contract is unchanged.
"""
import json
import math
import os
import re
import sys
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?\u2026])\s+|\n+")
ALPHA = 1.0          # Laplace smoothing
MIN_COUNT = 2        # training words rarer than this -> <UNK>
ROLL = 10            # rolling window (tokens) for surprisal burstiness

LM_FEATURE_NAMES = [
    "lm_mean_logprob",
    "lm_std_logprob",
    "lm_frac_predictable",     # logprob > -1.5
    "lm_frac_surprising",       # logprob < -7.0
    "lm_surprisal_burstiness",  # variance of rolling-window mean surprisal
    "lm_p05_logprob",           # 5th percentile (spikiness)
    "lm_oov_rate",
    "lm_min_window_logprob",    # least predictable stretch
]


def _toks(text):
    return [w.lower() for w in WORD_RE.findall(text)]


def _sents(text):
    return [p.strip() for p in SENT_SPLIT_RE.split(text.strip()) if p.strip()]


def build_lm(texts):
    """Fit trigram counts on an iterable of human texts. Returns a plain dict."""
    wc = Counter()
    for t in texts:
        wc.update(_toks(t))
    vocab = {w for w, c in wc.items() if c >= MIN_COUNT}
    vocab |= {"<UNK>", "<s>"}

    tri = Counter()
    bi = Counter()
    for t in texts:
        for s in _sents(t):
            toks = ["<s>", "<s>"] + [w if w in vocab else "<UNK>"
                                     for w in _toks(s)]
            for a, b, c in zip(toks, toks[1:], toks[2:]):
                tri[(a, b, c)] += 1
                bi[(a, b)] += 1
    V = len(vocab)
    return {"vocab": vocab, "tri": dict(tri), "bi": dict(bi), "V": V,
            "alpha": ALPHA}


def _logprob(lm, a, b, c):
    tri, bi, V, alpha = lm["tri"], lm["bi"], lm["V"], lm["alpha"]
    return math.log((tri.get((a, b, c), 0) + alpha) /
                    (bi.get((a, b), 0) + alpha * V))


def token_logprobs(lm, text):
    """Log-probability of each token's trigram given its 2-word history."""
    vocab = lm["vocab"]
    lps = []
    oov = 0
    n = 0
    for s in _sents(text):
        toks = ["<s>", "<s>"] + [w if w in vocab else "<UNK>"
                                 for w in _toks(s)]
        for a, b, c in zip(toks, toks[1:], toks[2:]):
            lps.append(_logprob(lm, a, b, c))
            n += 1
            if c == "<UNK>":
                oov += 1
    return lps, (oov / n if n else 0.0)


def lm_features(lm, text):
    """8 predictability features for *text* under the human trigram LM."""
    lps, oov_rate = token_logprobs(lm, text)
    if not lps:
        return [0.0] * len(LM_FEATURE_NAMES)
    n = len(lps)
    mean = sum(lps) / n
    var = sum((x - mean) ** 2 for x in lps) / n
    std = math.sqrt(var)
    frac_pred = sum(1 for x in lps if x > -1.5) / n
    frac_surp = sum(1 for x in lps if x < -7.0) / n
    srt = sorted(lps)
    p05 = srt[max(0, int(0.05 * n))]
    # rolling-window mean logprob -> burstiness of surprisal
    if n >= ROLL:
        wins = [sum(lps[i:i + ROLL]) / ROLL for i in range(0, n - ROLL + 1, ROLL)]
        wmean = sum(wins) / len(wins)
        burst = sum((w - wmean) ** 2 for w in wins) / len(wins)
        min_win = min(wins)
    else:
        burst = 0.0
        min_win = mean
    return [mean, std, frac_pred, frac_surp, burst, p05, oov_rate, min_win]


if __name__ == "__main__":
    # Quick class-separation check on train (human-built LM).
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    recs = [json.loads(l) for l in open(f"{root}/data/train.jsonl")
            if l.strip()]
    human_texts = [r["text"] for r in recs if r["label"] == 0]
    print(f"building LM on {len(human_texts)} human texts ...", flush=True)
    lm = build_lm(human_texts)
    print(f"vocab={lm['V']} trigrams={len(lm['tri'])}", flush=True)
    for label, tag in ((1, "AI"), (0, "HUMAN")):
        feats = [lm_features(lm, r["text"]) for r in recs if r["label"] == label]
        cols = list(zip(*feats))
        print(f"--- {tag} (n={len(feats)}) ---")
        for nm, col in zip(LM_FEATURE_NAMES, cols):
            print(f"  {nm:26s} mean={sum(col)/len(col):9.4f}")
