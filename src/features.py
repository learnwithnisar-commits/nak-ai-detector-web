"""NAK AI Detector - stylometric feature extractor.

extract_features(text) -> np.ndarray (1-D, fixed length, float64).

Pure Python + numpy. No fitted vectorizer is used: every feature is computed
directly from the raw text, so the extractor works in any fresh process.

Standardization is the only fitted step. The StandardScaler is stored in
models/feature_artifacts.joblib and loaded LAZILY by this module (first call
that needs it). The artifacts directory is resolved from the MODEL_PATH
environment variable when set, otherwise from the default project layout
(<project>/models/). If the artifacts file is absent, raw (unscaled)
features are returned so the extractor never crashes.

Deliberate design choice (see EVAL.md): this extractor does NOT use
typo/double-space/lowercase-start detectors. The human training samples were
synthetically assembled with injected typos (~10%) and double spaces (~5%),
so those would be pure assembly fingerprints, not universal human signals.

The main model file models/nak_detector.joblib is a RAW sklearn classifier
with .predict_proba, so the contract
    joblib.load(path).predict_proba([extract_features(text)])
works in a fresh process.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- tokenizers

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?\u2026])\s+|\n+")
_CONTRACTION_RE = re.compile(
    r"\b\w+(?:n't|'re|'ve|'ll|'d|'m|'s|'t)\b", re.IGNORECASE)


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


# ---------------------------------------------------------------- word lists

# ~64 common function words (fixed list, deterministic).
FUNCTION_WORDS = [
    "the", "a", "an", "of", "and", "to", "in", "that", "it", "is", "was",
    "were", "are", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "for", "with", "on", "as", "at", "by", "from",
    "or", "but", "not", "no", "this", "these", "those", "they", "them",
    "their", "theirs", "he", "him", "his", "she", "her", "hers", "we", "us",
    "our", "ours", "you", "your", "yours", "i", "me", "my", "mine",
    "which", "who", "whom", "what", "when", "where", "how", "why", "if",
    "then", "than", "so", "such", "because", "while", "although", "though",
    "also", "very", "more", "most", "can", "could", "should", "would",
    "will", "may", "might", "must", "shall", "its", "itself", "into",
    "over", "under", "between", "through", "during", "each", "other",
    "some", "any", "all", "both", "many", "much", "own",
]

PRON_1SG = {"i", "me", "my", "mine", "myself"}
PRON_1PL = {"we", "us", "our", "ours", "ourselves"}
PRON_2ND = {"you", "your", "yours", "yourself", "yourselves"}
PRON_3RD = {"he", "him", "his", "himself", "she", "her", "hers", "herself",
            "they", "them", "their", "theirs", "themselves", "it"}

# Genuine AI-style discourse markers (also present in real AI output;
# the synthetic AI banks were seeded with them, so EVAL.md reports how much
# the model leans on them vs. deeper stylometric features).
_TELL_GROUPS: list[tuple[str, list[str]]] = [
    ("tell_formal_trans", [
        "moreover", "furthermore", "additionally", "in addition",
        "consequently", "as a result", "nevertheless", "nonetheless",
        "notwithstanding", "accordingly", "subsequently",
    ]),
    ("tell_contrast", [
        "however", "on the other hand", "that said", "despite this",
        "even so", "in contrast", "conversely",
    ]),
    ("tell_hedge_academic", [
        "it is important to note", "it is worth noting", "it's worth noting",
        "it should be noted", "worth mentioning", "crucial to",
        "plays a crucial role", "plays a vital role", "it is essential",
        "it is clear that", "needless to say",
    ]),
    ("tell_closer", [
        "in conclusion", "in summary", "to sum up", "ultimately",
        "all in all", "to conclude",
    ]),
    ("tell_cliche", [
        "delve", "tapestry", "multifaceted", "nuanced", "testament to",
        "embark", "foster", "leverage", "realm", "intricate", "vibrant",
        "ever-evolving", "rapidly evolving", "in today's fast-paced",
        "game changer", "landscape",
    ]),
]
_TELL_RES = [(name, [re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE)
                     for p in pats])
             for name, pats in _TELL_GROUPS]

_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022\u2023\u25e6]|\d{1,2}[.)])\s+\S")

_PUNCT = [",", ".", ";", ":", "!", "?", "\u2026", "'", '"', "\u201c",
          "\u201d", "(", ")", "-", "\u2013", "\u2014", "/", "&", "@",
          "#", "%", "$", "+"]

FEATURE_NAMES: list[str] = []


def _register(names: list[str]) -> None:
    FEATURE_NAMES.extend(names)


# ---------------------------------------------------------------- helpers

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float], mu: float | None = None) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs) if mu is None else mu
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _entropy(probs: list[float]) -> float:
    return -sum(p * np.log(p) for p in probs if p > 0.0)


def _count_res(res_list: list[re.Pattern], text: str) -> int:
    return sum(len(rx.findall(text)) for rx in res_list)


# ---------------------------------------------------------------- scaler

_scaler = None          # fitted sklearn StandardScaler or None
_scaler_tried = False   # only warn once about a missing artifacts file
_lm = None              # trigram LM dict (or None)
_lm_tried = False
_lm_override = None     # set by train.py while the LM is freshly built
_lm_enabled = True      # False -> pure 179-feature mode (no-LM fallback)
_mode_explicit = False  # True when train.py set the mode; bundle must not override


def set_lm_enabled(enabled: bool):
    """Switch the extractor between 187-dim (LM) and 179-dim (no-LM) mode."""
    global _lm_enabled, _mode_explicit
    _lm_enabled = bool(enabled)
    _mode_explicit = True


def _get_bundle():
    """Load models/feature_artifacts.joblib once (lazy); None if absent."""
    path = _artifacts_path()
    if not path.exists():
        return None
    try:
        import joblib
        bundle = joblib.load(path)
        # Fallback bundles carry no "lm": the extractor must then return
        # exactly 179 features so the serving contract holds with no app change.
        # (Skipped when train.py set the mode explicitly for this process.)
        if isinstance(bundle, dict) and "lm" not in bundle and not _mode_explicit:
            set_lm_enabled(False)
        return bundle
    except Exception as exc:
        warnings.warn(f"feature artifacts unreadable ({path}): {exc}")
        return None


def _artifacts_path() -> Path:
    """Resolve models/feature_artifacts.joblib from MODEL_PATH env or layout."""
    env = os.environ.get("MODEL_PATH")
    if env:
        p = Path(env)
        # MODEL_PATH may point at the model file (app/main.py default) or a
        # directory: in both cases the artifacts live alongside the model.
        base = p.parent if p.suffix == ".joblib" else p
        return base / "feature_artifacts.joblib"
    here = Path(__file__).resolve()
    # src/features.py -> project root is parent of src/
    root = here.parent.parent
    return root / "models" / "feature_artifacts.joblib"


def _get_scaler():
    global _scaler, _scaler_tried
    if _scaler is None and not _scaler_tried:
        _scaler_tried = True
        bundle = _get_bundle()
        if bundle is None:
            warnings.warn("feature artifacts not found; using raw features")
        else:
            _scaler = bundle.get("scaler") if isinstance(bundle, dict) else bundle
    return _scaler


def set_lm(lm):
    """Inject a freshly built trigram LM (used by train.py before saving)."""
    global _lm_override
    _lm_override = lm


def _get_lm():
    global _lm, _lm_tried
    if not _lm_enabled:
        return None
    if _lm_override is not None:
        return _lm_override
    if _lm is None and not _lm_tried:
        _lm_tried = True
        bundle = _get_bundle()
        if bundle is not None and isinstance(bundle, dict):
            _lm = bundle.get("lm")
    return _lm


# ---------------------------------------------------------------- extractor

def _raw_features(text: str) -> np.ndarray:
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    words = _words(text)
    n_w = len(words)
    sents = _sentences(text)
    paras = _paragraphs(text)
    lines = text.splitlines()
    per1k = 1000.0 / n_w if n_w else 0.0

    feats: list[float] = []
    names: list[str] = []

    def add(name: str, value: float):
        names.append(name)
        v = float(value)
        feats.append(v if np.isfinite(v) else 0.0)

    # ---- basic counts --------------------------------------------------
    add("n_chars", len(text))
    add("n_words", n_w)
    add("n_sents", len(sents))
    add("n_paras", len(paras))
    add("mean_word_len", _mean([len(w) for w in words]))
    add("std_word_len", _std([len(w) for w in words]))
    add("pct_long_words6", sum(1 for w in words if len(w) > 6) * _safe_div(100.0, n_w))
    add("pct_long_words8", sum(1 for w in words if len(w) > 8) * _safe_div(100.0, n_w))

    # ---- sentence length / rhythm --------------------------------------
    sent_lens = [len(_WORD_RE.findall(s)) for s in sents]
    mu = _mean(sent_lens)
    sd = _std(sent_lens, mu)
    add("sent_mean_w", mu)
    add("sent_std_w", sd)
    add("sent_min_w", min(sent_lens) if sent_lens else 0.0)
    add("sent_max_w", max(sent_lens) if sent_lens else 0.0)
    add("sent_burstiness", _safe_div(sd, mu))          # std/mean
    add("pct_short_sent8", sum(1 for L in sent_lens if L < 8) * _safe_div(100.0, len(sent_lens)))
    add("pct_long_sent30", sum(1 for L in sent_lens if L > 30) * _safe_div(100.0, len(sent_lens)))
    add("sent_median_w", float(np.median(sent_lens)) if sent_lens else 0.0)
    add("sent_mean_chars", _mean([len(s) for s in sents]))
    # entropy of binned sentence lengths (rhythm unpredictability)
    if sent_lens:
        bins = [min(L // 5, 9) for L in sent_lens]
        bc = [bins.count(b) for b in range(10)]
        add("sent_len_entropy", _entropy([c / len(bins) for c in bc]))
    else:
        add("sent_len_entropy", 0.0)

    # ---- lexical diversity ----------------------------------------------
    types = set(words)
    n_t = len(types)
    add("ttr", _safe_div(n_t, n_w))
    add("guiraud_r", _safe_div(n_t, n_w ** 0.5))
    from collections import Counter
    wc = Counter(words)
    hapax = sum(1 for c in wc.values() if c == 1)
    dis = sum(1 for c in wc.values() if c == 2)
    add("hapax_ratio", _safe_div(hapax, n_t))
    add("dis_ratio", _safe_div(dis, n_t))
    # Yule's K (vocabulary concentration)
    if n_w:
        m1 = n_w
        freq_of_freq = Counter(wc.values())
        s2 = sum((f ** 2) * v for f, v in freq_of_freq.items())
        yule_k = 1e4 * _safe_div(s2 - m1, m1 ** 2)
        add("yule_k", yule_k)
        probs = [c / m1 for c in wc.values()]
        add("word_entropy", _entropy(probs))
    else:
        add("yule_k", 0.0)
        add("word_entropy", 0.0)

    # ---- repetition ------------------------------------------------------
    add("pct_words_repeated", sum(1 for w in words if wc[w] > 1) * _safe_div(100.0, n_w))
    bigrams = list(zip(words, words[1:]))
    trigrams = list(zip(words, words[1:], words[2:]))
    add("bigram_unique_ratio", _safe_div(len(set(bigrams)), len(bigrams)))
    add("trigram_unique_ratio", _safe_div(len(set(trigrams)), len(trigrams)))

    # ---- punctuation (per 1000 words) ------------------------------------
    for p in _PUNCT:
        add(f"punct_{ord(p):04x}", text.count(p) * per1k)
    n_end_excl = sum(1 for s in sents if s.rstrip().endswith("!"))
    n_end_q = sum(1 for s in sents if s.rstrip().endswith("?"))
    add("pct_sent_end_excl", n_end_excl * _safe_div(100.0, len(sents)))
    add("pct_sent_end_quest", n_end_q * _safe_div(100.0, len(sents)))

    # ---- function words (per 1000 words) ----------------------------------
    for fw in FUNCTION_WORDS:
        add(f"fw_{fw}", wc.get(fw, 0) * per1k)

    # ---- pronouns / contractions ------------------------------------------
    add("pron_1sg", sum(wc.get(p, 0) for p in PRON_1SG) * per1k)
    add("pron_1pl", sum(wc.get(p, 0) for p in PRON_1PL) * per1k)
    add("pron_2nd", sum(wc.get(p, 0) for p in PRON_2ND) * per1k)
    add("pron_3rd", sum(wc.get(p, 0) for p in PRON_3RD) * per1k)
    add("contractions_1k", len(_CONTRACTION_RE.findall(text)) * per1k)

    # ---- AI-discourse tell markers (per 1000 words) ------------------------
    for name, res_list in _TELL_RES:
        add(name, _count_res(res_list, text) * per1k)

    # ---- paragraphs --------------------------------------------------------
    para_lens = [len(_WORD_RE.findall(p)) for p in paras]
    add("para_mean_w", _mean(para_lens))
    add("para_std_w", _std(para_lens))
    add("para_max_w", max(para_lens) if para_lens else 0.0)
    add("pct_single_sent_para",
        sum(1 for p in paras if len(_sentences(p)) <= 1) * _safe_div(100.0, len(paras)))

    # ---- structure -----------------------------------------------------------
    content_lines = [ln for ln in lines if ln.strip()]
    add("bullet_line_ratio",
        sum(1 for ln in lines if _BULLET_RE.match(ln)) * _safe_div(1.0, len(content_lines)))
    # heading-like lines: short, no terminal punctuation, title/upper case or ends with ':'
    head = 0
    for ln in content_lines:
        s = ln.strip()
        if len(s) <= 60 and not re.search(r"[.!?]$", s) and (
                s.endswith(":") or s.isupper() or s.istitle()):
            head += 1
    add("heading_line_ratio", head * _safe_div(1.0, len(content_lines)))
    add("colon_1k", text.count(":") * per1k)
    add("newline_ratio", text.count("\n") * _safe_div(1.0, max(len(text), 1)) * 1000.0)

    # ---- capitalization --------------------------------------------------------
    cap_mid = 0
    cap_total = 0
    allcaps = 0
    for s in sents:
        toks = re.findall(r"[A-Za-z']+", s)
        for j, tok in enumerate(toks):
            if tok and tok[0].isupper():
                cap_total += 1
                if j > 0:
                    cap_mid += 1
                if tok.isupper() and len(tok) > 1:
                    allcaps += 1
    add("pct_cap_mid_sent", cap_mid * _safe_div(100.0, max(cap_total, 1)))
    add("allcaps_1k", allcaps * per1k)

    # ---- quotes / dialogue -------------------------------------------------------
    add("quote_chars_1k", (text.count('"') + text.count("\u201c") + text.count("\u201d")
                           + text.count("'")) * per1k)
    add("dialogue_line_ratio",
        sum(1 for ln in lines if re.match(r"^\s*[\u2014\u2013-]\s*\S", ln) and len(ln.strip()) > 2)
        * _safe_div(1.0, len(content_lines)))

    if not FEATURE_NAMES:
        _register(names)
    elif len(names) != len(FEATURE_NAMES):
        raise RuntimeError("feature schema changed mid-process")

    return np.asarray(feats, dtype=np.float64)


def extract_features_unscaled(text: str) -> np.ndarray:
    """Full unscaled vector: 179 stylometric [+ 8 LM predictability] features."""
    raw = _raw_features(text)
    # NOTE: _get_lm() may flip _lm_enabled off when the bundle has no "lm"
    # (no-LM fallback), so re-check after the call.
    lm = _get_lm() if _lm_enabled else None
    if not _lm_enabled:
        return raw
    from src.lm import LM_FEATURE_NAMES, lm_features
    if lm is None:
        lm_vec = np.zeros(len(LM_FEATURE_NAMES))
    else:
        lm_vec = np.asarray(lm_features(lm, text), dtype=np.float64)
        lm_vec = np.where(np.isfinite(lm_vec), lm_vec, 0.0)
    return np.concatenate([raw, lm_vec])


def extract_features(text: str) -> np.ndarray:
    """Return the (scaled) 1-D feature vector for *text*."""
    vec = extract_features_unscaled(text)
    scaler = _get_scaler()
    if scaler is None:
        return vec
    return np.asarray(scaler.transform(vec.reshape(1, -1))[0], dtype=np.float64)


def feature_names() -> list[str]:
    """Return the ordered feature names (populated after first extraction)."""
    from src.lm import LM_FEATURE_NAMES
    if not FEATURE_NAMES:
        _raw_features("Warm-up sentence for schema registration.")
    _get_lm()  # ensure _lm_enabled reflects the saved bundle (no-LM fallback)
    names = list(FEATURE_NAMES)
    if _lm_enabled and names and not names[-1].startswith("lm_"):
        names = names + list(LM_FEATURE_NAMES)
    return names
