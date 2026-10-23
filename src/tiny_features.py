"""NAK v4 'tiny' model — small interpretable WITHIN-REGISTER feature set.

Every feature here is a property of *how* text is written (repetition,
burstiness, punctuation habits), never *what* it is about. No topic or
register proxies (no jargon lists, no domain cues).

Features (10), computed on an 8-sentence window:
  1. ttr                  type-token ratio (word repetition)
  2. sent_len_cv          SD/mean of sentence lengths in words (burstiness)
  3. punct_per1k          (,;:) per 1000 words
  4. mean_sent_len        mean words per sentence
  5. long_word_rate       fraction of words with >= 7 letters
  6. sents_per_100w       sentences per 100 words
  7. comma_per1k          commas per 1000 words
  8. semicolon_colon_per1k  (;:) per 1000 words
  9. ai_tell_per1k        curated AI-style connective/cliche phrases per 1k
 10. mean_word_len        mean letters per word
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.scan import split_sentences

FEATURE_NAMES = [
    "ttr",
    "sent_len_cv",
    "punct_per1k",
    "mean_sent_len",
    "long_word_rate",
    "sents_per_100w",
    "comma_per1k",
    "semicolon_colon_per1k",
    "ai_tell_per1k",
    "mean_word_len",
]

AI_TELLS = [
    "however", "moreover", "furthermore", "in addition", "consequently",
    "therefore", "additionally", "notably", "ultimately", "in conclusion",
    "in summary", "to sum up", "to conclude", "delve", "tapestry",
    "multifaceted", "nuanced", "testament to", "embark", "foster",
    "leverage", "realm", "intricate", "vibrant", "landscape",
    "ever-evolving", "rapidly evolving", "in today's fast-paced",
    "game changer", "underscores", "crucial", "pivotal",
]
_TELL_RES = [re.compile(r"\b" + re.escape(p) + r"\b") for p in AI_TELLS]

_WORD_RE = re.compile(r"[A-Za-z']+")


def extract_tiny_features(text: str) -> list[float]:
    sents = [s for s in split_sentences(text) if s.strip()]
    words = _WORD_RE.findall(text.lower())
    n = len(words)
    if n == 0:
        return [0.0] * len(FEATURE_NAMES)
    sent_lens = [len(_WORD_RE.findall(s)) for s in sents] or [0]
    mean_sl = sum(sent_lens) / len(sent_lens)
    var_sl = sum((l - mean_sl) ** 2 for l in sent_lens) / len(sent_lens)
    sd_sl = var_sl ** 0.5
    ttr = len(set(words)) / n
    per1k = 1000.0 / n
    commas = text.count(",") * per1k
    semicol = (text.count(";") + text.count(":")) * per1k
    punct = commas + semicol
    long_rate = sum(1 for w in words if len(w) >= 7) / n
    low = text.lower()
    tells = sum(len(rx.findall(low)) for rx in _TELL_RES) * per1k
    mean_wl = sum(len(w) for w in words) / n
    return [
        ttr,
        (sd_sl / mean_sl) if mean_sl else 0.0,
        punct,
        mean_sl,
        long_rate,
        len(sents) / n * 100.0,
        commas,
        semicol,
        tells,
        mean_wl,
    ]
