# Plagiarism Detection Engine — Technical Audit & Architecture Plan

**Date:** 2026-10-17 | **Status:** AUDIT + PLAN (no implementation) | **Module:** separate from AI-writing detector (v17 untouched)

---

## PART 1 — AUDIT OF EXISTING ASSETS

### 1.1 Production matching engine: `app/plag.py` (1,098 lines, LIVE)

| Component | Implementation |
|---|---|
| Fingerprinting | Winnowing, K=5 word shingles, W=4 window → detects shared runs ≥ 8 words |
| Hash | md5 → signed int64 (`_h64`) |
| Normalization | `norm_word`: lowercase + strip punctuation |
| Core classes | `StreamWinnower`, `OAIndex`, `MultiOAIndex` |
| Matching | `match_blocks()` — span merging (MERGE_GAP=12), noise floor MIN_MATCH_WORDS=8 |
| Exclusions | Quote exclusion (inline `"…"`, block quotes); bibliography exclusion exists |
| Categories | NAK taxonomy: "Submitted documents" / "Internet" / "Publication" |
| Caps | MIN_PLAG_WORDS=30 (skip short docs), MAX_SOURCES=25, DOC_FP_LOOKUP_CAP=200k |
| Output | Similarity % cover + passage-level highlighted report (own design) |

**Gaps vs research-grade:** no MinHash/LSH near-duplicate, no semantic similarity, no fuzzy/paraphrase layer, no cross-language, passage offsets not surfaced as data (only rendered HTML), no per-field precision/recall eval, doc-vs-doc only via retained fingerprints.

### 1.2 Index assets

| Index | Docs | Fingerprints | Size | Status |
|---|---|---|---|---|
| **Seed** `plag_index/plag_index.db` | 43,733 | ~5.0M | ~108 MB | **LIVE on Railway** |
| **Mega** `data/mega_index/plag_index.db` | 11,081,233 | 788,387,803 | 34.75 GB | Built, on HF (`Nisarahmedkatyar/nak-plag-index-r3`, 17 shards); NOT live |
| **Web** `data/web_academic_index/plag_web.db` | 6,820 | 15.7M | 576 MB | Built, self-test passed; NOT live (no hosted URL) |

**Schema (all three, identical):**
```sql
CREATE TABLE docs(id INTEGER PRIMARY KEY, doc_id TEXT UNIQUE, title TEXT, source TEXT);
CREATE TABLE fps(fp INTEGER NOT NULL, doc INTEGER NOT NULL, PRIMARY KEY (fp, doc)) WITHOUT ROWID;
```

**Seed sources:** arxiv 19,253 · doaj 7,261 · europepmc 8,496 + 917 FT · semanticscholar 7,800 · wikipedia 5.
**Web sources:** mdpi.com 6,275 · frontiersin.org 545 (doc_id = canonical URL → linkable).
**Mega sources** (verified 2026-10-17): web 6,057,582 · pubmed 3,841,742 · crossref 499,999 · thesis 353,346 · openalex 230,038 · wikipedia 54,784 · arxiv 19,253 · europepmc 8,496 · semanticscholar 7,800 · doaj 7,261 · europepmc_ft 917 · ia 15. Fingerprint-only, no full text.

### 1.3 Harvest scripts (~157 in `scripts/`)

`fetch_arxiv.py`, `harvest_openalex.py` (polite pool, mailto), `harvest_universities.py`, `harvest_uok.py`, `harvest_pastic.py`, `harvest_famous.py`, `mega_crossref.py`, `r3_commoncrawl.py` (WARC range downloads — no origin-server hits), `r3_crossref.py`, `build_plag_index.py`, `build_plag_index_db.py`, `mega_plag_index.py`.

**Corpus intermediates:** `data/oa_corpus/` (`fp_*.jsonl`, `arxiv_docs.jsonl`, `doaj_docs.jsonl`, …).

### 1.4 Reference/eval material

`reference/plagiarism/` — sample pairs (11454, COM211, XDR) + originals. No formal benchmark yet (no planted-copy suite with ground-truth offsets).

### 1.5 What is already production-ready vs missing

✅ Exact-match winnowing pipeline, quote/bib exclusion, report rendering, doc-vs-doc via retained fingerprints, seed index serving live.
❌ Near-duplicate (MinHash/LSH), semantic candidate discovery, fuzzy/paraphrase matching, cross-language, source-URL offsets as structured data, benchmark suite, precision/recall/FPR eval, coverage dashboards.

---

## PART 2 — ARCHITECTURE PLAN

### 2.1 Source coverage (7 categories) + honest limitations

| # | Category | Planned connector(s) | Status |
|---|---|---|---|
| 1 | Public websites/blogs/articles | Common Crawl WARC (CDX API, no origin hits); domain sitemaps (robots.txt-checked) | Web index built (mdpi/frontiers); batch-2 domains pending |
| 2 | OA papers (journals/conf/preprints) | OpenAlex (metadata+OA URLs), Crossref (DOI→metadata), Unpaywall (OA version resolution), publisher JATS/XML APIs | Mega index (11M) built; FT fetch partial |
| 3 | University repos / theses | OAI-PMH harvesters (per-repo), `harvest_universities.py` pattern | Partial (uok, pastic scripts exist) |
| 4 | Government publications | Gov open-data portals, standards bodies' OA sections | Not started |
| 5 | Books (lawful only) | Project Gutenberg, Internet Archive (lending metadata only), Google Books API (snippets, not full text) | Not started |
| 6 | User docs + internal corpus | Existing: retained fingerprints of scanned docs (30-min auto-delete of text, fingerprints kept, disclosed) | **Live** |
| 7 | Version/duplicate detection | MinHash/LSH layer (planned §2.2.3); doc-family grouping by doc_id lineage | Planned |

**Explicit limitations (do not claim otherwise):**
- No access to Turnitin's proprietary student-paper repository (~67% of its sources per internal analysis — not replicable).
- No paywall bypass, ever. Metadata availability ≠ full-text access.
- Common Crawl coverage is crawl-snapshot-dependent (some domains bot-blocked: nature.com, biomedcentral.com).
- Books limited to public-domain/lawfully licensed text.
- Coverage claims must be per-index (docs count + source breakdown), never "the whole web".

### 2.2 Hybrid detection pipeline

```
input doc → normalize → [L1 exact hashes] ─┐
                 ├→ [L2 winnowing fps] ─────┼→ candidate retrieval (per-index)
                 ├→ [L3 MinHash/LSH] ──────┘
                 └→ [L4 semantic embeddings] → candidate discovery only
candidates → passage alignment (offsets) → scoring → quote/bib/common-phrase filters → report
```

1. **L1 Exact matching** — normalized full-text SHA-256 (identical-document detection) + exact sentence hashes. Use: instant duplicate/version detection.
2. **L2 N-gram fingerprints** — existing winnowing (K=5/W=4) + optional K=8 variant for high-precision tier. Rolling hash retained for stream processing.
3. **L3 Near-duplicate (MinHash/LSH)** — 128-permutation MinHash over 5-shingles; LSH bands for sub-linear candidate retrieval at 11M-doc scale. Enables version detection + corpus dedup. (New build.)
4. **L4 Semantic similarity** — sentence embeddings (e.g. multilingual-e5) as *candidate discovery only*; never scored as plagiarism evidence. Threshold-gated, always paired with lexical verification. (New build.)
5. **Passage-level matching** — every hit returns: source URL/title, doc_id, char/word offsets in both documents, matched word count, match class. Structured JSON, not just rendered HTML.
6. **Fuzzy matching** — edit-distance-bounded shingle variants + stopword-insensitive normalization tier for minor rewording; paraphrase *patterns* flagged as "potential paraphrase — review", never auto-verdict.
7. **Cross-language** — separate module, separate eval (e.g. EN↔UR pilot); disabled by default; metrics reported independently.

### 2.3 Source acquisition (lawful-first)

- **Crossref:** metadata + DOI resolution (mailto polite pool). Metadata only.
- **OpenAlex:** discovery + `best_oa_location` URLs. Full text fetched only from OA locations.
- **Unpaywall:** OA version resolution (email param). No key needed for basic use.
- **OA repositories:** OAI-PMH / SWORD / JATS XML where offered; sitemap-based where robots.txt permits.
- **Common Crawl:** CDX API + WARC range downloads (never hammer origin servers).
- **User corpora:** private doc-vs-doc via retained fingerprints (existing pattern).
- Rules: respect robots.txt, API rate limits, publisher terms; user-agent identifies the project; backoff on 429; stop on 403 (hard stop, no circumvention).

### 2.4 Research-document processing

- **Formats:** PDF (text layer; OCR fallback flagged as lower-confidence), DOCX, TXT.
- **Segmentation:** preserve page numbers + char/word offsets end-to-end.
- **Special handling:** references/bibliography (excluded from %, listed separately), block/inline quotations (excluded, shown as "properly quoted"), equations/tables (normalized, matched separately), headers/footers (stripped), OCR text (confidence-flagged).
- **Match classes (evidence, not verdicts):** exact copy · lightly modified · potential paraphrase (review) · proper quote · bibliography match · common phrase/terminology (below noise floor, not counted).

### 2.5 Evaluation

- **Benchmark** `plagiarism/benchmark/`: planted-copy suite — originals + known sources + {verbatim copies, lightly-edited copies, paraphrased passages, legitimate quotes, reference lists, unrelated docs} with ground-truth passage offsets.
- **Splits:** by source-document family (all versions of one paper in the same split). No fingerprint leakage across splits.
- **Metrics:** precision, recall, F1 (passage-level), false-positive rate on unrelated docs, source-retrieval recall@k, localization accuracy (offset IoU).
- **Stratified:** by field (CS/bio/med/eng/humanities) and by document length bands.
- **Supervised ML only** where labelled data justifies it; retrieval and matching stay rule-based and separate from any classifier.

### 2.6 Index schema & dedup (v2)

```sql
-- v2 extends existing schema; v1 tables unchanged for backward compat
CREATE TABLE docs(id INTEGER PRIMARY KEY, doc_id TEXT UNIQUE, title TEXT,
  source TEXT, url TEXT, published DATE, lang TEXT, n_words INT, family_id TEXT);
CREATE TABLE fps(fp INTEGER NOT NULL, doc INTEGER NOT NULL, PRIMARY KEY (fp, doc)) WITHOUT ROWID;
CREATE TABLE minhash(doc INTEGER PRIMARY KEY, sig BLOB NOT NULL);      -- 128 x int32
CREATE TABLE passages(fp INTEGER NOT NULL, doc INTEGER NOT NULL,
  wstart INTEGER, wend INTEGER);                                        -- offset index
```
- **Dedup:** doc_id canonicalization + exact-hash dedup on ingest; MinHash ≥0.9 → same family_id.
- **Sharding:** keep per-source SQLite shards (HF-hosted); query fan-out with score merge.

### 2.7 API contract (planned)

```
POST /api/plagiarism/scan   {file, options{exclude_quotes, exclude_bib, lang}}
  → {scan_id, status}
GET  /api/plagiarism/result/{scan_id}
  → {similarity_pct, sources:[{doc_id,title,url,category,matched_words,
      passages:[{wstart,wend,src_wstart,src_wend,match_class}]}], 
      stats{...}, coverage_note}
```
- Every response carries `coverage_note` (which indexes were searched, doc counts).
- Test cases: planted verbatim (expect ≥95% retrieval), light edit, paraphrase (candidate only), proper quote (0% counted), unrelated (FPR), cross-doc version pair.

### 2.8 Phased implementation (no big-bang)

- **P0:** Benchmark suite + eval harness (scripts only, uses seed index). *Cost: ~0.*
- **P1:** Passage offsets as structured JSON from existing winnowing (extend `match_blocks`).
- **P2:** MinHash/LSH layer + `minhash` table; corpus dedup run on seed index.
- **P3:** Host web index (576 MB) → wire `WEB_INDEX_URL` boot-download (already supported by app).
- **P4:** Source connectors (Unpaywall → OpenAlex OA fetch → repo OAI-PMH), one domain at a time.
- **P5:** Semantic candidate discovery (embeddings), gated behind lexical verification.
- **P6:** Cross-language pilot (EN↔UR), separately evaluated.
- Mega-index serving (34 GB) stays deferred: needs sharded query service + hosting budget.

### 2.9 Progress log

- 2026-10-17: Audit complete; architecture plan written. Seed index live (43,733 docs). Mega index on HF (11.08M docs, not live). Web index built (6,820 docs, not live). No implementation started.

---

*Constraints honored: no AI-detector changes, no retraining, no Turnitin-equivalence claims, no costly crawls.*
