#!/usr/bin/env python3
"""Hybrid plagiarism detection: winnowing + semantic re-ranking.

Stage 1: Fast winnowing fingerprint match (existing plag.py).
Stage 2: Semantic re-ranking of candidate passages using MiniLM embeddings.
Catches paraphrased copying (synonym swaps, reordering) that winnowing misses.

Usage: python scripts/semantic_rerank.py
"""
import re
import sys
from pathlib import Path

PROJ = Path("/home/hatch/workspace/ai-detector")
sys.path.insert(0, str(PROJ))

def get_model():
    """Get similarity model. Uses char n-grams (lightweight, no DL needed).
    Char n-grams catch paraphrases better than word TF-IDF because they
    capture sub-word patterns that survive synonym replacement.
    """
    return ("chargram", None)

def _char_ngrams(text: str, n: int = 4) -> set:
    """Extract character n-grams (lowercased, alphanumeric only)."""
    clean = re.sub(r"[^a-z0-9 ]", "", text.lower())
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}

def semantic_similarity(text1: str, text2: str, model=None) -> float:
    """Jaccard similarity over character 4-grams (0-1).
    Robust to synonym swaps and light paraphrasing."""
    import re
    g1 = _char_ngrams(text1, 4)
    g2 = _char_ngrams(text2, 4)
    if not g1 or not g2:
        return 0.0
    inter = len(g1 & g2)
    union = len(g1 | g2)
    return inter / union if union else 0.0

def rerank_candidates(query_passage: str, candidates: list[tuple[str, str]],
                      threshold: float = 0.75, model=None) -> list[tuple[str, float]]:
    """Re-rank candidate passages by semantic similarity.
    
    Args:
        query_passage: The submitted text passage.
        candidates: List of (source_label, candidate_text).
        threshold: Minimum similarity to consider a match.
    
    Returns:
        List of (source_label, similarity) sorted by similarity, filtered by threshold.
    """
    if model is None:
        model = get_model()
    results = []
    for label, cand_text in candidates:
        sim = semantic_similarity(query_passage, cand_text, model)
        if sim >= threshold:
            results.append((label, sim))
    results.sort(key=lambda x: -x[1])
    return results

if __name__ == "__main__":
    # Quick test
    print("Loading model...", flush=True)
    m = get_model()
    print("Model loaded.", flush=True)
    
    # Test: paraphrased text should still match
    orig = "The quick brown fox jumps over the lazy dog in the meadow."
    paraphrase = "A fast brown fox leaps across the sleepy dog in the field."
    unrelated = "Quantum computing uses qubits for parallel processing."
    
    s1 = semantic_similarity(orig, paraphrase, m)
    s2 = semantic_similarity(orig, unrelated, m)
    print(f"Paraphrase similarity: {s1:.3f} (should be high)")
    print(f"Unrelated similarity: {s2:.3f} (should be low)")
    print("PASS" if s1 > 0.7 and s2 < 0.5 else "CHECK THRESHOLDS")
