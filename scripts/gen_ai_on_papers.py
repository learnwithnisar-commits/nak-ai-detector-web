#!/usr/bin/env python3
"""Generate AI-style academic abstracts on the SAME topics as collected papers.

Combinatorial, seeded, reproducible. For each human paper (title+abstract),
builds an AI-voiced abstract from large pools of LLM-typical academic
sentences across 5 style families, grounded on the paper's own topic words
so topic cannot separate the classes.

Input : data/arxiv_train.jsonl, data/europepmc_train.jsonl
        (or ..._eval.jsonl when --eval is passed)
Output: data/real_ai/real_ai_train.jsonl (or real_ai_eval.jsonl)
        {text,label:1,genre,style,paper_id,...}

Train/eval paper sets are disjoint (see prepare_real_data.py), so AI
topics never leak across the split.

Known limitation (same as data/README.md): single-AI-source synthetic data.
Styles are varied deliberately to reduce overfitting to one LLM voice.
"""
import json
import os
import random
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
HUMAN_DIR = os.path.join(ROOT, "data", "real_human")
OUTDIR = os.path.join(ROOT, "data", "real_ai")
os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, "ai_on_papers.jsonl")

SEED = 20261012

STOP = set("""a an the of and in on to for with by from as at is are was were be
been being this that these those it its our we they their he she his her
into over under between through during via per using use used uses based
novel new approach via within without within""".split())

# ---------------------------------------------------------------- pools
# {t} = topic phrase (from title), {k} = keyword phrase
OPENERS = [
    ("hedged-formal", "This paper investigates {t}, a topic that has attracted growing attention in recent years."),
    ("hedged-formal", "In this work, we study {t} and examine several questions that remain open in the literature."),
    ("hedged-formal", "We consider the problem of {t}, which plays an increasingly central role in modern research."),
    ("concise-direct", "This paper studies {t}."),
    ("concise-direct", "We analyze {t} and report new findings."),
    ("concise-direct", "This work addresses {t} directly."),
    ("verbose-transitional", "In recent years, {t} has emerged as an area of considerable interest, motivating the present study."),
    ("verbose-transitional", "The rapid development of research on {t} has raised important questions, which this paper seeks to address."),
    ("verbose-transitional", "Understanding {t} is widely regarded as important; here we contribute new insights into the problem."),
    ("structured", "We present a study of {t}, organized as follows: motivation, methodology, results, and implications."),
    ("structured", "This paper makes three contributions to the study of {t}: a framework, an analysis, and an empirical evaluation."),
    ("narrative", "The question of how to understand {t} has puzzled researchers for some time."),
    ("narrative", "Behind recent progress on {t} lies a set of ideas we aim to clarify in this work."),
    ("hedged-formal", "Recent advances have renewed interest in {t}, yet several fundamental aspects remain poorly understood."),
    ("concise-direct", "We revisit {t} from a new angle."),
    ("verbose-transitional", "Although {t} has been studied extensively, existing approaches leave important gaps that we attempt to fill."),
    ("structured", "Our focus is {t}. We (1) formalize the setting, (2) develop new tools, and (3) validate them empirically."),
    ("narrative", "Progress on {t} has been uneven, with theory and practice advancing at different speeds."),
    ("hedged-formal", "This study is concerned with {t}, particularly in settings where prior methods are difficult to apply."),
    ("concise-direct", "We propose a fresh look at {t}."),
    ("verbose-transitional", "It is by now well established that {t} matters; what is less clear is how best to approach it, and this is our starting point."),
    ("structured", "We tackle {t} in two parts: a theoretical development followed by a practical demonstration."),
    ("narrative", "The literature on {t} is rich but fragmented; we try to bring some order to it."),
    ("hedged-formal", "Motivated by practical challenges, this paper examines {t} under realistic assumptions."),
    ("concise-direct", "This abstract summarizes our work on {t}."),
    ("verbose-transitional", "Few topics have generated as much discussion lately as {t}, and the present paper adds a new perspective."),
    ("structured", "We address {t} through a combination of analysis and experiment, described below."),
    ("narrative", "What drives performance in {t}? We take a step toward answering this question."),
    ("hedged-formal", "We undertake a systematic investigation of {t}, with emphasis on {k}."),
    ("concise-direct", "New results on {t} are presented."),
]

METHODS = [
    ("hedged-formal", "Our approach builds on {k}, extending it to handle cases that previous work could not address."),
    ("hedged-formal", "We develop a framework centered on {k}, which we show to be both flexible and principled."),
    ("hedged-formal", "The methodology combines {k} with careful theoretical analysis."),
    ("concise-direct", "We use {k} to attack the problem."),
    ("concise-direct", "Our method relies on {k}."),
    ("concise-direct", "We apply {k} and measure the outcome."),
    ("verbose-transitional", "To make progress, we introduce a new perspective based on {k}; furthermore, we show how it connects to existing techniques."),
    ("verbose-transitional", "Our strategy is to leverage {k}. In addition, we provide supporting analysis that clarifies why the method works."),
    ("verbose-transitional", "We begin by formalizing the problem in terms of {k}. Subsequently, we derive the main technical results."),
    ("structured", "First, we define the setup around {k}. Second, we prove the key properties. Third, we test them."),
    ("structured", "The method has two components: a model of {k} and an algorithm that exploits its structure."),
    ("narrative", "The key idea is surprisingly simple: {k} already contains most of what is needed."),
    ("narrative", "We follow the thread of {k} and see where it leads."),
    ("hedged-formal", "It is worth noting that our treatment of {k} differs from earlier accounts in several respects."),
    ("concise-direct", "Experiments use {k} throughout."),
    ("verbose-transitional", "Moreover, we extend the analysis to {k}, demonstrating that the conclusions are robust."),
    ("structured", "We proceed in stages, each building on {k} in a different way."),
    ("narrative", "Instead of starting from scratch, we adapt {k} to the new setting."),
    ("hedged-formal", "A central element of our contribution is a refined understanding of {k}."),
    ("concise-direct", "The core technique is {k}."),
    ("verbose-transitional", "Importantly, the framework accommodates {k} naturally, which simplifies the subsequent development."),
    ("structured", "Concretely, we (i) specify {k}, (ii) analyze its behavior, and (iii) evaluate it."),
    ("narrative", "The turning point came when we realized {k} could be reinterpreted."),
    ("hedged-formal", "We place particular emphasis on {k}, as it appears to drive the main phenomena."),
    ("concise-direct", "We formalize {k} and test it."),
    ("verbose-transitional", "On the technical side, we rely on {k}; on the conceptual side, we offer a new interpretation."),
    ("structured", "Our pipeline takes {k} as input and produces validated conclusions."),
    ("narrative", "{k} turns out to be the right lens for the problem."),
    ("hedged-formal", "While {k} is well known, its implications for {t} have not been fully explored."),
    ("concise-direct", "{k} is our main tool."),
]

RESULTS = [
    ("hedged-formal", "Our results demonstrate that the proposed approach performs favorably across a range of conditions."),
    ("hedged-formal", "We find consistent improvements, suggesting that the method captures something essential."),
    ("hedged-formal", "The analysis reveals patterns that were not apparent in earlier studies."),
    ("concise-direct", "Results are strong across all tests."),
    ("concise-direct", "We observe clear gains."),
    ("concise-direct", "The numbers support our claims."),
    ("verbose-transitional", "Extensive experiments confirm the effectiveness of the approach; furthermore, ablation studies isolate the contributing factors."),
    ("verbose-transitional", "The findings are robust: across variations in setup, the conclusions remain unchanged."),
    ("verbose-transitional", "Notably, the method succeeds precisely where baselines struggle, which underscores its value."),
    ("structured", "Quantitatively, we improve on prior work. Qualitatively, the behavior is more interpretable."),
    ("structured", "Three findings stand out: accuracy, efficiency, and generality."),
    ("narrative", "The experiments told a clear story, and it matched the theory."),
    ("narrative", "What we saw in practice lined up remarkably well with the predictions."),
    ("hedged-formal", "To the best of our knowledge, these are the first results of their kind for {t}."),
    ("concise-direct", "We beat the baselines."),
    ("verbose-transitional", "In addition, we document several secondary effects that may interest practitioners."),
    ("structured", "We report (a) main results, (b) ablations, and (c) a case study."),
    ("narrative", "The data left little doubt about the direction of the effect."),
    ("hedged-formal", "While some variability remains, the overall picture is encouraging."),
    ("concise-direct", "Performance is consistent."),
    ("verbose-transitional", "Crucially, the improvements persist under stricter evaluation protocols."),
    ("structured", "Results are summarized in terms of {k}, where the gains are most visible."),
    ("narrative", "Even the failure cases were informative, pointing to natural next steps."),
    ("hedged-formal", "The evidence, taken together, supports the central hypothesis."),
    ("concise-direct", "All key metrics improve."),
    ("verbose-transitional", "It is important to emphasize that these gains come without additional complexity."),
    ("structured", "We validate on multiple benchmarks covering {k}."),
    ("narrative", "Seeing the theory work in practice was the most satisfying part."),
    ("hedged-formal", "Preliminary results are promising, though further validation is warranted."),
    ("concise-direct", "The approach works as intended."),
]

CLOSERS = [
    ("hedged-formal", "In conclusion, this work advances the understanding of {t} and opens several avenues for future research."),
    ("hedged-formal", "We hope these findings will inform subsequent work on {t}."),
    ("hedged-formal", "The implications of this study extend beyond {t} to related problems."),
    ("concise-direct", "We conclude with directions for future work."),
    ("concise-direct", "This completes our study of {t}."),
    ("verbose-transitional", "In summary, we have presented a new view of {t}; moreover, we believe the ideas will prove broadly useful."),
    ("verbose-transitional", "Taken together, our contributions provide a solid foundation for future investigation of {t}."),
    ("structured", "To summarize: we formalized {t}, developed new tools, and validated them. Future work will extend the scope."),
    ("narrative", "The story of {t} is far from over, but this chapter clarifies an important part of it."),
    ("hedged-formal", "Future research might profitably explore the extensions suggested by our analysis of {k}."),
    ("concise-direct", "Future work will generalize these results."),
    ("verbose-transitional", "Ultimately, we view this as a step toward a deeper understanding of {t}, with much remaining to be done."),
    ("structured", "We close by noting limitations and outlining concrete next steps."),
    ("narrative", "If this work prompts others to look again at {t}, it will have served its purpose."),
    ("hedged-formal", "We leave a fuller exploration of {k} to future work."),
    ("concise-direct", "We suggest several extensions."),
    ("verbose-transitional", "In closing, it is worth reiterating that {t} remains a fertile ground for research."),
    ("structured", "Conclusion: {t} benefits from the new perspective, and several open problems remain."),
    ("narrative", "The next chapter of {t} is waiting to be written."),
    ("hedged-formal", "These results, while preliminary in some respects, point to a promising research direction."),
]

STYLES = ["hedged-formal", "concise-direct", "verbose-transitional", "structured", "narrative"]

def keywords(title, abstract, n=3):
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", (title + " " + abstract).lower())
    freq = {}
    for w in words:
        if w not in STOP:
            freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq, key=lambda w: -freq[w])
    out = []
    for w in ranked:
        if w not in out:
            out.append(w)
        if len(out) >= n:
            break
    return out or ["models"]

def topic_phrase(title):
    t = re.sub(r"\s+", " ", title).strip().rstrip(".")
    # strip leading "On/On the/A/An" for natural embedding
    t = re.sub(r"^(on|a|an|the)\s+", "", t, flags=re.I)
    return t[0].lower() + t[1:] if t else "the problem"

def build_abstract(rec, rng):
    title = rec.get("title", "")
    t = topic_phrase(title)
    kws = keywords(title, rec.get("text", ""))
    k = ", ".join(kws[:2]) if len(kws) > 1 else kws[0]

    def pick(pool, style):
        cands = [s for st, s in pool if st == style] or [s for _, s in pool]
        return rng.choice(cands)

    # dominant style for this abstract (mix allowed: 1 slot may differ)
    dom = rng.choice(STYLES)
    parts = [pick(OPENERS, dom).format(t=t, k=k)]
    n_mid = rng.choice([4, 5, 6])
    for _ in range(n_mid):
        st = dom if rng.random() < 0.75 else rng.choice(STYLES)
        pool = rng.choice([METHODS, RESULTS])
        parts.append(pick(pool, st).format(t=t, k=k))
    parts.append(pick(CLOSERS, dom).format(t=t, k=k))
    if rng.random() < 0.4:
        parts.append(pick(CLOSERS, rng.choice(STYLES)).format(t=t, k=k))

    text = " ".join(parts)
    words = text.split()
    # enforce 120-350 words by trimming/adding
    if len(words) > 350:
        text = " ".join(words[:340])
    return text, dom

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true",
                    help="generate on eval-split papers -> real_ai_eval.jsonl")
    a = ap.parse_args()
    suffix = "eval" if a.eval else "train"
    seed = SEED * 100003 + (777001 if a.eval else 0)
    recs = []
    for fname in (f"arxiv_{suffix}.jsonl", f"europepmc_{suffix}.jsonl"):
        p = os.path.join(DATA, fname)
        if not os.path.exists(p):
            print(f"skip missing {fname}")
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    print(f"papers ({suffix}): {len(recs)}")
    rng = random.Random(SEED)
    seen_texts = set()
    n_out = 0
    OUT = os.path.join(OUTDIR, f"real_ai_{suffix}.jsonl")
    with open(OUT, "w", encoding="utf-8") as f:
        for i, rec in enumerate(recs):
            prng = random.Random(seed + i)
            text, style = build_abstract(rec, prng)
            words = len(text.split())
            if not (100 <= words <= 360) or text in seen_texts:
                continue
            seen_texts.add(text)
            pid = rec.get("arxiv_id") or rec.get("pmid") or f"p{i}"
            f.write(json.dumps({
                "text": text, "label": 1, "genre": "academic_abstract",
                "style": style, "source": "synthetic_ai_on_topic",
                "paper_id": pid, "paper_source": rec.get("source"),
                "paper_title": rec.get("title", "")[:120], "words": words,
            }, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"DONE: {n_out} AI abstracts -> {OUT}")

if __name__ == "__main__":
    main()
