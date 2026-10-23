#!/usr/bin/env python3
"""Stylometric features for AI detection (research method #7).
68 interpretable features: no neural LM needed, pure Python stdlib.
These complement the existing neural/statistical features.
"""
import re
from collections import Counter

# Common English function words (closed class)
FUNCTION_WORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "but", "for",
    "with", "on", "at", "by", "from", "as", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might", "must",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these",
    "those", "my", "your", "his", "her", "its", "our", "their",
    "not", "no", "yes", "so", "if", "then", "than", "too", "very",
}

def extract_stylometric(text: str) -> dict:
    """Extract 20 core stylometric features (subset of the 68 from research).
    Returns dict of feature_name -> float.
    """
    if not text or not text.strip():
        return {}
    
    # Tokenize
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    sents = [s for s in sents if s.strip()]
    
    n_words = len(words)
    n_sents = len(sents)
    if n_words < 10 or n_sents < 2:
        return {}
    
    feats = {}
    
    # 1. Type-token ratio (vocabulary diversity)
    feats["ttr"] = len(set(words)) / n_words
    
    # 2. Hapax legomena rate (words appearing exactly once)
    counts = Counter(words)
    hapax = sum(1 for c in counts.values() if c == 1)
    feats["hapax_rate"] = hapax / n_words
    
    # 3. Average word length
    feats["avg_word_len"] = sum(len(w) for w in words) / n_words
    
    # 4. Average sentence length (words)
    sent_lens = [len(re.findall(r"\b[a-zA-Z]+\b", s)) for s in sents]
    feats["avg_sent_len"] = sum(sent_lens) / n_sents
    
    # 5. Sentence length std (burstiness)
    mean_sl = feats["avg_sent_len"]
    feats["sent_len_std"] = (sum((x - mean_sl) ** 2 for x in sent_lens) / n_sents) ** 0.5
    
    # 6. Function word ratio
    func_count = sum(1 for w in words if w in FUNCTION_WORDS)
    feats["func_word_ratio"] = func_count / n_words
    
    # 7-10. Punctuation rates (per sentence)
    feats["comma_rate"] = text.count(",") / n_sents
    feats["semicolon_rate"] = text.count(";") / n_sents
    feats["colon_rate"] = text.count(":") / n_sents
    feats["exclaim_rate"] = text.count("!") / n_sents
    feats["question_rate"] = text.count("?") / n_sents
    
    # 11. Capitalization rate (words starting with capital, excluding sentence starts)
    caps = len(re.findall(r"(?<=[a-z] )[A-Z][a-z]+", text))
    feats["mid_sent_caps"] = caps / n_words
    
    # 12. Long word rate (>6 chars)
    feats["long_word_rate"] = sum(1 for w in words if len(w) > 6) / n_words
    
    # 13. Short word rate (<4 chars)
    feats["short_word_rate"] = sum(1 for w in words if len(w) < 4) / n_words
    
    # 14. Digit rate
    feats["digit_rate"] = len(re.findall(r"\d", text)) / max(1, len(text))
    
    # 15. Quotation rate
    feats["quote_rate"] = (text.count('"') + text.count("'")) / n_sents
    
    # 16. Parenthesis rate
    feats["paren_rate"] = (text.count("(") + text.count(")")) / n_sents
    
    # 17. Unique bigram ratio
    bigrams = list(zip(words, words[1:]))
    feats["bigram_diversity"] = len(set(bigrams)) / max(1, len(bigrams))
    
    # 18. Transition word rate (however, therefore, moreover, etc.)
    transitions = {"however", "therefore", "moreover", "furthermore", "consequently",
                   "nevertheless", "accordingly", "meanwhile", "otherwise", "indeed"}
    feats["transition_rate"] = sum(1 for w in words if w in transitions) / n_sents
    
    # 19. Personal pronoun rate (I, we, you)
    pronouns = {"i", "we", "you", "my", "our", "your"}
    feats["pronoun_rate"] = sum(1 for w in words if w in pronouns) / n_words
    
    # 20. Passive voice indicator (be + past participle pattern, approximate)
    passive = len(re.findall(r"\b(am|is|are|was|were|be|been|being)\s+\w+ed\b", text.lower()))
    feats["passive_rate"] = passive / n_sents
    
    return feats

def feature_names() -> list:
    """Ordered feature names for ML."""
    return ["ttr", "hapax_rate", "avg_word_len", "avg_sent_len", "sent_len_std",
            "func_word_ratio", "comma_rate", "semicolon_rate", "colon_rate",
            "exclaim_rate", "question_rate", "mid_sent_caps", "long_word_rate",
            "short_word_rate", "digit_rate", "quote_rate", "paren_rate",
            "bigram_diversity", "transition_rate", "pronoun_rate", "passive_rate"]

if __name__ == "__main__":
    # Test on human vs AI-like text
    human = ("It was a bright cold day in April, and the clocks were striking thirteen. "
             "Winston Smith, his chin nuzzled into his breast in an effort to escape the "
             "vile wind, slipped quickly through the glass doors of Victory Mansions.")
    ai_like = ("The implementation of comprehensive strategies facilitates optimal outcomes. "
               "Furthermore, stakeholders should consider various factors. Moreover, "
               "it is important to note that effective communication is essential.")
    
    fh = extract_stylometric(human)
    fa = extract_stylometric(ai_like)
    print("Human-like TTR:", round(fh.get("ttr", 0), 3), " AI-like TTR:", round(fa.get("ttr", 0), 3))
    print("Human transition rate:", round(fh.get("transition_rate", 0), 3),
          " AI transition rate:", round(fa.get("transition_rate", 0), 3))
    print("Features extracted:", len(fh))
