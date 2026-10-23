#!/usr/bin/env python3
"""Round-4 candidate NEW features: within-register discriminators.

Extension to src/features.py (separate module/function, does NOT touch the
187-dim production schema). All features are computed on an 8-sentence window
string.

Design goal (from oct16r3 diagnosis): separate AI-student-writing from
human-student-writing WITHIN the same register, where the 187 base features
only shift the whole register upward.

Validated 2026-10-16 on Turnitin's own sentence-highlight labels
(843 AI + 1,915 human sentences from 33 mixed docs): kept 10/49 features with
|Cohen's d| >= 0.25 or AUC >= 0.58 AND within-doc sign consistency >= 0.60.
Kept (in KEPT10 order): proper_cap, digit, date, nominal, past_ed,
sent_start_bland_ratio, sent_len_gini, sent_len_range_cv, clause_density,
repeat_trigram.

Key empirical findings (window-level, AI vs human):
  - human student reports are MORE concrete: more digits, dates, proper nouns
  - AI student writing: more uniform sentence lengths (lower gini),
    more nominalizations, blander sentence starts, denser clauses,
    fewer repeated trigrams
"""
import re
import numpy as np

KEPT10 = ["r4_proper_cap_1k", "r4_digit_1k", "r4_date_1k", "r4_nominal_1k",
          "r4_past_ed_1k", "r4_sent_start_bland_ratio", "r4_sent_len_gini",
          "r4_sent_len_range_cv", "r4_clause_density", "r4_repeat_trigram_1k"]

_WORD_RE = re.compile(r"[A-Za-z']+")
_WSENT_RE = re.compile(r"(?<=[.!?\u2026])\s+|\n+")


def _words(t):
    return [w.lower() for w in _WORD_RE.findall(t)]


def _sents(t):
    return [s.strip() for s in _WSENT_RE.split(t.strip()) if s.strip()]


SUBORD = {"because", "although", "though", "while", "whereas", "since", "unless",
          "until", "after", "before", "once", "if", "when", "whenever", "wherever",
          "as", "so", "provided", "assuming", "lest", "whereby"}
_PASSIVE_RE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has been|have been|had been|will be|"
    r"would be|could be|should be|can be|may be|might be)\s+"
    r"(?:\w+ly\s+)?[a-z]+(?:ed|en)\b", re.IGNORECASE)
_NOM_RE = re.compile(r"\b[a-z]{4,}(?:tion|sion|ment|ness|ity|ance|ence|ship|hood)\b",
                     re.IGNORECASE)
_ENUM_RE = re.compile(r"\b(?:firstly|secondly|thirdly|fourthly|finally|lastly)\b",
                      re.IGNORECASE)
_OPENER_RE = re.compile(
    r"\bthis report (?:aims|presents|focuses|discusses|outlines|examines|provides)|"
    r"\bthe purpose of this (?:report|project|study)|"
    r"\bthis project (?:aims|focuses|seeks)|"
    r"\bthis study (?:aims|investigates|examines)", re.IGNORECASE)
_INORDERTO_RE = re.compile(r"\bin order to\b|\bso as to\b", re.IGNORECASE)
_BYING_RE = re.compile(
    r"\bby (?:implementing|using|applying|leveraging|utilizing|integrating|"
    r"adopting|employing|incorporating)\b", re.IGNORECASE)
_HEDGE_RE = re.compile(
    r"\b(?:may|might|could|possibly|probably|likely|tend to|tends to|"
    r"suggests? that|appears? to|seems? to)\b", re.IGNORECASE)
_BOOST_RE = re.compile(
    r"\b(?:very|highly|extremely|significantly|remarkably|notably|crucially|"
    r"tremendously|immensely)\b", re.IGNORECASE)
_BLAND_STARTS = {"the", "this", "it", "these", "those", "a", "an"}
EXTRA_FW = ["which", "while", "within", "between", "through", "during",
            "such", "upon", "whereas", "whose", "whom"]


def extract_new_features(text):
    """Return (names, np.array) of candidate new features for window *text*."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    words = _words(text)
    n_w = len(words)
    sents = _sents(text)
    per1k = 1000.0 / n_w if n_w else 0.0
    feats, names = [], []

    def add(nm, v):
        names.append(nm)
        v = float(v)
        feats.append(v if np.isfinite(v) else 0.0)

    # ---- R: rhythm -----------------------------------------------------
    sent_lens = [len(_WORD_RE.findall(s)) for s in sents]
    if sent_lens:
        mu = sum(sent_lens) / len(sent_lens)
        sl = sorted(sent_lens)
        n = len(sl)
        gini = (sum((2 * i - n + 1) * x for i, x in enumerate(sl, 1))
                / (n * sum(sl))) if sum(sl) else 0.0
        add("r4_sent_len_gini", gini)
        q = np.quantile(sent_lens, [0.25, 0.75])
        add("r4_sent_len_iqr", float(q[1] - q[0]))
        add("r4_sent_len_range_cv", (max(sent_lens) - min(sent_lens)) / mu if mu else 0.0)
    else:
        add("r4_sent_len_gini", 0.0)
        add("r4_sent_len_iqr", 0.0)
        add("r4_sent_len_range_cv", 0.0)
    firsts = [(_WORD_RE.findall(s) or [""])[0].lower() for s in sents]
    add("r4_sent_start_diversity",
        len(set(firsts)) / len(firsts) if firsts else 0.0)
    add("r4_sent_start_bland_ratio",
        sum(1 for f in firsts if f in _BLAND_STARTS) / len(firsts) if firsts else 0.0)

    # ---- S: syntactic proxies ------------------------------------------
    wc = {}
    for w in words:
        wc[w] = wc.get(w, 0) + 1
    add("r4_subord_1k", sum(wc.get(w, 0) for w in SUBORD) * per1k)
    add("r4_passive_1k", len(_PASSIVE_RE.findall(text)) * per1k)
    add("r4_nominal_1k", len(_NOM_RE.findall(text)) * per1k)
    add("r4_comma_per_sent", text.count(",") / len(sents) if sents else 0.0)
    add("r4_paren_1k", (text.count("(") + text.count(")")) * per1k)

    # ---- F: extra function words ----------------------------------------
    for fw in EXTRA_FW:
        add(f"r4_fw_{fw}", wc.get(fw, 0) * per1k)

    # ---- D: self-similarity ----------------------------------------------
    if len(sents) >= 2:
        sets = [set(_words(s)) for s in sents]
        cos, cnt = 0.0, 0
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                inter = len(sets[i] & sets[j])
                d = (len(sets[i]) * len(sets[j])) ** 0.5
                if d:
                    cos += inter / d
                    cnt += 1
        add("r4_sent_pair_cosine", cos / cnt if cnt else 0.0)
        jac = []
        for i in range(len(sets) - 1):
            u = sets[i] | sets[i + 1]
            jac.append(len(sets[i] & sets[i + 1]) / len(u) if u else 0.0)
        add("r4_adj_sent_jaccard", sum(jac) / len(jac) if jac else 0.0)
    else:
        add("r4_sent_pair_cosine", 0.0)
        add("r4_adj_sent_jaccard", 0.0)
    bigrams3 = list(zip(words, words[1:], words[2:]))
    if bigrams3:
        from collections import Counter
        c3 = Counter(bigrams3)
        add("r4_repeat_trigram_1k",
            sum(1 for c in c3.values() if c >= 2) * per1k)
    else:
        add("r4_repeat_trigram_1k", 0.0)

    # ---- T: discourse tells ----------------------------------------------
    add("r4_tell_enumerate_1k", len(_ENUM_RE.findall(text)) * per1k)
    add("r4_tell_report_opener_1k", len(_OPENER_RE.findall(text)) * per1k)
    add("r4_tell_inorderto_1k", len(_INORDERTO_RE.findall(text)) * per1k)
    add("r4_tell_by_ing_1k", len(_BYING_RE.findall(text)) * per1k)
    add("r4_hedge_general_1k", len(_HEDGE_RE.findall(text)) * per1k)
    add("r4_booster_1k", len(_BOOST_RE.findall(text)) * per1k)

    # ---- L: lexical/shape extras -----------------------------------------
    add("r4_digit_1k", len(re.findall(r"\d", text)) * per1k)

    # ---- C: concreteness / specificity ------------------------------------
    add("r4_measure_unit_1k", len(re.findall(
        r"\d[\d,.]*\s*(?:kg|g\b|mg|km\b|cm\b|mm\b|\bm\b|s\b|ms\b|hz\b|v\b|"
        r"a\b|w\b|pa\b|n\b|j\b|°c|c\b|percent|%|rpm|db\b)",
        text, re.IGNORECASE)) * per1k)
    add("r4_citation_1k", len(re.findall(
        r"\[\d+(?:,\s*\d+)*\]|\(\d{4}[a-z]?\)|\bet al\.?", text)) * per1k)
    add("r4_figref_1k", len(re.findall(
        r"\b(?:figure|fig\.|table|eq\.|equation|section|appendix)\b",
        text, re.IGNORECASE)) * per1k)
    add("r4_eg_ie_etc_1k", len(re.findall(
        r"\be\.g\.|\bi\.e\.|\betc\.", text)) * per1k)
    add("r4_date_1k", len(re.findall(
        r"\b(?:january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?|"
        r"\b(?:19|20)\d{2}\b", text, re.IGNORECASE)) * per1k)
    add("r4_eqsym_1k", len(re.findall(r"[=+×±→<>≤≥^]", text)) * per1k)
    add("r4_past_ed_1k", len(re.findall(r"\b[a-z]{3,}ed\b", text.lower())) * per1k)
    add("r4_hyphen_1k", len(re.findall(r"[a-zA-Z]+-[a-zA-Z]+", text)) * per1k)
    starts = [(_WORD_RE.findall(s) or [""])[0] for s in sents]
    add("r4_informal_start_ratio",
        sum(1 for f in starts if f in {"And", "But", "So", "Well", "Also", "Then"})
        / len(starts) if starts else 0.0)
    coord = len(re.findall(
        r"\b(?:and|but|or|nor|for|so|yet|which|that|who|whom|whose)\b",
        text.lower()))
    add("r4_clause_density", coord / len(sents) if sents else 0.0)
    add("r4_tell_overall_1k", len(re.findall(
        r"\boverall\b|\bin general\b|\bgenerally speaking\b",
        text, re.IGNORECASE)) * per1k)
    add("r4_tell_itadj_that_1k", len(re.findall(
        r"\bit is [a-z]+ that\b", text, re.IGNORECASE)) * per1k)
    add("r4_tell_canseen_1k", len(re.findall(
        r"\bcan be seen\b|\bas shown\b|\bas illustrated\b|\bas presented\b",
        text, re.IGNORECASE)) * per1k)
    add("r4_tell_observed_1k", len(re.findall(
        r"\b(?:we|i)\s+(?:observed|found|noticed|measured|recorded|"
        r"calculated|conducted|performed)\b", text, re.IGNORECASE)) * per1k)

    # ---- C2: concreteness robust variants ----------------------------------
    cap_mid = 0
    for s in sents:
        toks = re.findall(r"[A-Za-z']+", s)
        for tok in toks[1:]:
            if tok and tok[0].isupper() and not tok.isupper():
                cap_mid += 1
    add("r4_proper_cap_1k", cap_mid * per1k)
    add("r4_sent_with_digit_ratio",
        sum(1 for s in sents if re.search(r"\d", s)) / len(sents) if sents else 0.0)
    add("r4_long_num_1k", len(re.findall(r"\b\d{3,}\b", text)) * per1k)
    add("r4_vague_quant_1k", len(re.findall(
        r"\bseveral\b|\bvarious\b|\bnumerous\b|\ba number of\b|"
        r"\ba wide range of\b|\ba variety of\b", text, re.IGNORECASE)) * per1k)

    return names, np.asarray(feats, dtype=np.float64)


def feature_names_new():
    names, _ = extract_new_features("warm up sentence for names")
    return names
