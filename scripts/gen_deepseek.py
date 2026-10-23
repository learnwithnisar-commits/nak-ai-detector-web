#!/usr/bin/env python3
"""Generate academic-register AI samples via DeepSeek API (Nisar's key).

Needs: DEEPSEEK_API_KEY env (from platform.deepseek.com, Nisar's account).
Cost: ~$0.14/1M tokens — 100 samples ≈ a few cents.

Usage: DEEPSEEK_API_KEY=sk-... .venv/bin/python scripts/gen_deepseek.py --n 100
       DEEPSEEK_API_KEY=sk-... .venv/bin/python scripts/gen_deepseek.py --mode manuscripts --n-manuscripts 16
Output: data/family_deepseek_train.jsonl  (schema: text/family/register/label [+topic])
"""
import argparse, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TOPICS = [
    ("academic-ml", "contrastive learning for job-resume matching"),
    ("academic-ml", "Bayesian networks for personnel selection"),
    ("academic-ml", "multilingual sentence embeddings in hiring"),
    ("academic-ml", "uncertainty quantification in neural rankers"),
    ("academic-bio", "antimicrobial resistance surveillance in Salmonella"),
    ("academic-bio", "CRISPR diagnostics for field pathogen detection"),
    ("academic-business", "supply chain resilience modeling"),
    ("academic-business", "CBDC adoption in developing economies"),
    ("academic-humanities", "digital public sphere and affective publics"),
    ("academic-humanities", "oral history methodology in Partition studies"),
]

SYSTEM = ("You are writing a single academic paper paragraph (180-280 words). "
          "Formal scholarly tone, no headings, no bullet points, one paragraph only.")

# --- Full-manuscript generation mode -------------------------------------
# Lesson (2026-10-12): 20 copies of ONE paper teach the model that one paper,
# not "AI manuscript". So: many topics x many fields x several genres.
MS_SYSTEM = ("You are an academic author writing ONE section of a scholarly paper. "
             "Formal academic prose in continuous paragraphs only — no bullet points, "
             "no markdown headings inside the section, no meta-commentary.")

MANUSCRIPT_SECTIONS = [
    ("abstract",
     "Write the abstract (150-200 words): problem, method, key findings, implication.",
     450),
    ("introduction",
     "Write the introduction (300-400 words): background, the research gap, "
     "research questions, and the paper's contributions.",
     800),
    ("literature review",
     "Write the literature review (350-450 words): the key strands of prior work, "
     "open debates, and the gap this paper fills.",
     900),
    ("methodology",
     "Write the methodology (300-400 words): research design, data or sample, "
     "instruments, and analysis approach.",
     800),
    ("results",
     "Write the results (300-400 words): the main findings, with illustrative "
     "statistics or qualitative evidence.",
     800),
    ("discussion",
     "Write the discussion (300-400 words): interpretation of findings, practical "
     "implications, and limitations.",
     800),
    ("conclusion",
     "Write the conclusion (150-200 words): summary of contributions and future directions.",
     450),
]

# (field-register, paper title, genre) — all distinct topics by design.
MANUSCRIPT_TOPICS = [
    ("ml", "Self-Supervised Pretraining for Low-Resource Urdu Speech Recognition", "empirical quantitative study"),
    ("ml", "Federated Learning for Hospital Readmission Prediction Under Data Heterogeneity", "empirical quantitative study"),
    ("ml", "Retrieval-Augmented Generation for Legal Document Question Answering: A Survey", "literature review"),
    ("ml", "Benchmark Design for Code-Mixed NLP Evaluation", "methods paper"),
    ("bio", "CRISPR-Based Field Diagnostics for Wheat Rust Detection", "empirical quantitative study"),
    ("bio", "Antimicrobial Resistance Surveillance in Poultry Farms of Punjab", "empirical quantitative study"),
    ("bio", "Dengue Vector Control in South Asia: A Systematic Review", "literature review"),
    ("health", "Telemedicine Adoption in Rural Pakistan: A District Case Study", "case study"),
    ("health", "Lady Health Workers and Maternal Health Outcomes in Sindh", "empirical quantitative study"),
    ("health", "Vaccine Hesitancy in Urban Informal Settlements: A Mixed-Methods Study", "empirical quantitative study"),
    ("business", "Supply Chain Resilience Among Textile Exporters After the 2022 Floods", "empirical quantitative study"),
    ("business", "Central Bank Digital Currencies in Developing Economies: A Conceptual Framework", "theoretical paper"),
    ("business", "Microfinance and Women's Entrepreneurship in Rural Sindh", "case study"),
    ("environment", "Heatwave Mortality and Urban Green Cover in Karachi", "empirical quantitative study"),
    ("environment", "Mangrove Restoration in the Indus Delta: A Community Case Study", "case study"),
    ("environment", "PM2.5 Exposure and School Absenteeism in Lahore", "empirical quantitative study"),
    ("education", "AI Tutoring and Learning Outcomes in Public Secondary Schools", "empirical quantitative study"),
    ("education", "Language-of-Instruction Policy and Early-Grade Literacy: A Policy Analysis", "theoretical paper"),
    ("humanities", "The Digital Public Sphere and Affective Publics", "theoretical paper"),
    ("humanities", "Oral History Methodology in Partition Studies", "methods paper"),
    ("humanities", "Social Media and Political Mobilization Among Urban Youth", "empirical quantitative study"),
    ("engineering", "Low-Cost Arsenic Filtration for Rural Groundwater", "methods paper"),
    ("engineering", "Seismic Retrofitting of Unreinforced Masonry: A Review", "literature review"),
    ("agriculture", "Drip Irrigation Adoption and Water Productivity in Punjab", "empirical quantitative study"),
]


def _api_call(client_key, system, user, max_tokens, temperature=0.9):
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps({
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature, "max_tokens": max_tokens,
        }).encode(),
        headers={"Authorization": f"Bearer {client_key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def gen_manuscript(client_key, title, genre, field):
    """Generate a full ~2000-2500 word manuscript section by section."""
    sections, prior = [], []
    for name, instruction, mt in MANUSCRIPT_SECTIONS:
        ctx = "\n".join(f"- {n}: {s[:280]}..." for n, s in prior) or "(none written yet)"
        user = (f"Paper title: {title}\nGenre: {genre} ({field}).\n"
                f"Sections written so far:\n{ctx}\n\n{instruction}")
        for attempt in (1, 2):
            try:
                t = _api_call(client_key, MS_SYSTEM, user, mt)
                break
            except Exception as e:
                print(f"    [{name}] error (try {attempt}): {e}; retry in 20s", flush=True)
                time.sleep(20)
        else:
            raise RuntimeError(f"section '{name}' failed twice")
        sections.append((name, t))
        prior.append((name, t))
        time.sleep(2)
    full = "# " + title + "\n\n" + "\n\n".join(
        f"## {n.title()}\n\n{t}" for n, t in sections)
    return full

def gen(client_key, topic, register):
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps({
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Write the paragraph about: {topic}."},
            ],
            "temperature": 0.9, "max_tokens": 500,
        }).encode(),
        headers={"Authorization": f"Bearer {client_key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--mode", choices=["paragraphs", "manuscripts"], default="paragraphs")
    ap.add_argument("--n-manuscripts", type=int, default=16,
                    help="how many full manuscripts to generate in manuscripts mode")
    a = ap.parse_args()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("DEEPSEEK_API_KEY not set")
    out = os.path.join(ROOT, "data", "family_deepseek_train.jsonl")
    if a.mode == "manuscripts":
        have = set()
        if os.path.exists(out):
            for line in open(out, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                if o.get("register", "").startswith("manuscript-") and o.get("topic"):
                    have.add(o["topic"])
        targets = [t for t in MANUSCRIPT_TOPICS if t[1] not in have][:a.n_manuscripts]
        print(f"{len(have)} manuscripts already present; generating {len(targets)} new", flush=True)
        with open(out, "a", encoding="utf-8") as f:
            for i, (field, title, genre) in enumerate(targets):
                print(f"[ms {i+1}/{len(targets)}] {field} | {title}", flush=True)
                full = gen_manuscript(key, title, genre, field)
                words = len(full.split())
                f.write(json.dumps({
                    "text": full, "family": "deepseek",
                    "register": f"manuscript-{field}", "label": 1,
                    "topic": title, "words": words,
                }) + "\n")
                f.flush()
                print(f"    done ({words} words)", flush=True)
        print("wrote", out)
        return
    done = sum(1 for _ in open(out)) if os.path.exists(out) else 0
    with open(out, "a", encoding="utf-8") as f:
        for i in range(done, a.n):
            reg, topic = TOPICS[i % len(TOPICS)]
            try:
                t = gen(key, topic, reg)
            except Exception as e:
                print(f"[{i}] error: {e}; retry in 20s", flush=True)
                time.sleep(20)
                continue
            f.write(json.dumps({"text": t, "family": "deepseek",
                                "register": reg, "label": 1}) + "\n")
            f.flush()
            print(f"[{i+1}/{a.n}] {reg}", flush=True)
            time.sleep(2)
    print("wrote", out)

if __name__ == "__main__":
    main()
