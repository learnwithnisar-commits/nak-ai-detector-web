# NEXT_SESSION.md — V25 Ablation Experiments (COMPLETE)

**Date:** 2026-10-17
**Status:** All 5 ablations complete. Verdict below.

## Completed
- v24 root-cause analysis: `data/v24_rootcause_analysis.md`
- v24 rejected: DO_NOT_DEPLOY (MAE 23.59 vs 15.33; Full Paper 42.4 vs 79.7)
- Live v17 md5 `3e24dc7c79bc40b20c597e37e9aaaaf0` verified before/after EVERY experiment ✅

## Ablation Results (all vs v17 baseline, identical protocol)

| Exp | Change | Full Paper | PhD | MAE | within5 | Verdict |
|-----|--------|-----------|-----|-----|---------|---------|
| v17 (live) | baseline | 79.7% | 93.0% | 15.33 | 3/9 | LIVE |
| v24 | 31 AI cleaned @3x | 42.4% | 69.5% | 23.59 | 0/9 | DO_NOT_DEPLOY |
| E4 | v17 MINUS 72 quarantined | 79.1% | 91.8% | 15.47 | 3/9 | ≈v17 ✅ |
| E1 | v17 + Tier-1 cleaned text | 75.6% | 87.6% | 18.20 | 2/9 | < v17 ⚠️ |
| E2 | v24 + 400 real_ai @1x | 40.0% | 66.3% | 24.44 | 1/9 | NO recovery ❌ |
| E3 | v24 + petro @3x | 74.2% | 69.4% | 19.61 | 1/9 | **+31.8 recovery** ✅ |
| E5 | v24 with 31 AI @10x | 38.9% | 65.1% | 23.43 | 0/9 | WORSE ❌ |

## Key Findings

1. **E4 ≈ v17**: 72-record quarantine EXONERATED. The chrome-only records contributed 0 genuine signal.
2. **E1 < v17**: Tier-1 cleaning is NOT innocent — v17 was exploiting chrome→label correlations. Removing chrome hurts in-sample scores (expected: removes spurious signal).
3. **E2 FAILED**: real_ai @1x alone did NOT recover Full Paper (40.0% vs 42.4%). The "400 abstracts" hypothesis was WRONG as a single factor.
4. **E3 SUCCEEDED**: petro @3x recovered Full Paper 42.4% → 74.2% (+31.8 points!). A single 3,092-word ChatGPT petroleum doc at 3x weight is the dominant Full Paper signal. Domain-specificity matters more than volume.
5. **E5 FAILED**: @10x upsampling made it WORSE (38.9%). The failure is STRUCTURAL (missing register diversity), not reweightable.

## Revised Diagnosis

The v24 regression was caused by dropping **petro @3x** (primary, domain-specific) plus the **chrome signal** v17 exploited (secondary, spurious). The 400 real_ai abstracts were NOT the dominant factor — E2 proves this.

## Model Artifacts

| Model | MD5 | Path |
|-------|-----|------|
| v17 (live) | `3e24dc7c79bc40b20c597e37e9aaaaf0` | `models/nak_detector.joblib` |
| E4 | `cd835d4a3739c773a95378a49c22a381` | `models/candidates/allfields_v25_E4.joblib` |
| E1 | `c330d1d37ef6ed5ae4014c163726df7d` | `models/candidates/allfields_v25_E1.joblib` |
| E2 | `c89e0f7e8a898d348b1d65d5240dc0b7` | `models/candidates/allfields_v25_E2.joblib` |
| E3 | `781acc23407355ef020bdf52a4b79271` | `models/candidates/allfields_v25_E3.joblib` |
| E5 | `4b47746c22f245f5c3ebc7c0b5bbf958` | `models/candidates/allfields_v25_E5.joblib` |
| v24 | `58e7730f3d309e40492feef50a34b8de` | `models/candidates/allfields_v24.joblib` |

Eval reports: `models/candidates/allfields_v25_E{N}_eval_report.json`

## Honest Limitations

1. All legacy eval docs are in-sample/optimistic (V22 finding). No ablation establishes ±5.
2. The 31-doc human holdout was locked and untouched (specificity only).
3. No ablation fixes ash_coal (no ChatGPT-essay data exists).
4. E4 shows v17's scores depend partly on chrome exploitation — "recovering v17" means recovering in-sample fit including spurious signal.
5. A genuine ±5 claim needs an independent paired AI test set with verified Turnitin scores (does not exist).

## Recommended next action

**v25 = E3 recipe + real_ai @1x** (combine both genuine signals: petro @3x + 400 abstracts @1x on the cleaned base, 72 quarantined). This tests whether the two genuine-signal restorations are additive. Then evaluate with identical protocol.

## V26 — Controlled Combination (E3 + real_ai @1x)

**Status:** COMPLETE 2026-10-17 ~15:05 PKT. Live v17 md5 `3e24dc7c79bc40b20c597e37e9aaaaf0` verified before/after ✅

**Recipe:** E3 EXACTLY (= v24 + petro @3x) + `data/real_ai/real_ai_train.jsonl` first 400 docs @1x added to router tech data AND T (matching E2's documented recipe).
- V26 model: `models/candidates/allfields_v26.joblib` (md5 `3263e221bb7cb8882076e4c259db7523`)
- Eval report: `models/candidates/allfields_v26_eval_report.json` (verdict DO_NOT_DEPLOY)
- Train script: `scripts/segfull_train_allfields_v26.py` (97s); Eval script: `scripts/eval_allfields_v26.py` (E3 script transformed: candidate E3→V26, v17 baseline kept)
- Train acc: router 0.9686 (1592 segs), tech 0.8734 (8295 segs, pos=2370)
- real_ai loaded: 400 docs / 49,793 words. Leakage: real_ai(first400) vs holdout-31 = 0 overlaps ✅
- Router confounder: router retrained with real_ai included (same as E1–E5 protocol)

**Three-way comparison (identical pipeline; E3 numbers from saved E3 report):**

| Metric | V26 | E3 | v17 (live) |
|--------|-----|----|-----------|
| Full Paper (T=80) | 72.4 | 74.2 | 79.7 |
| 22103 (T=57) | 70.9 | 70.7 | 80.3 |
| PhD (T=100) | 63.4 | 69.4 | 93.0 |
| 8-anchor mean gap | 19.98 | 20.98 | 17.56 |
| 8-anchor within±5 | 0/8 | 1/8 | 2/8 |
| MAE | 20.29 | 19.61 | 15.33 |
| within-5 | 0/9 | 1/9 | 3/9 |
| within-10 | 3/9 | 3/9 | 5/9 |
| signed bias | −14.16 | −13.21 | −4.93 |
| median abs err | 13.7 | 15.1 | 7.0 |
| verified_human breach | 0.2 | 0.2 | 0.2 |
| drive_human breach | 0.0 | 0.0 | 0.0 |
| wikipedia mean | 26.51 | 26.97 | 27.66 |

**Finding:** The combination is NOT additive — real_ai @1x slightly DEGRADES E3 (Full Paper −1.8, PhD −6.0, MAE +0.68, within-5 1/9→0/9). The general-domain AI abstracts dilute the petroleum-specific ChatGPT register signal rather than complementing it. Consistent with E2's failure. Domain-specificity (petro) beats volume (400 abstracts).

**Per-doc V26 abs errors:** FullPaper 7.6 | 22103 13.9 | PhD 36.6 | Listeria 6.1 | XDR_Typhoid 24.0 | AIGC 8.1 | ash_coal 59.7 | Liu_Yueru 12.9 | ES1002 13.7

**Recommended next action:** No further combination of these two sources is justified. The genuine-signal ceiling on this corpus appears reached for the cleaned base: E3 (petro @3x) is the best cleaned-data candidate, still well below v17 on PhD/MAE. Any future work needs NEW verified data (esp. technical AI writing with numeric Turnitin scores), not reweighting. Keep v17 live unchanged.

## FINAL VERDICT: DO_NOT_DEPLOY

No candidate beats or matches v17 on the full gate suite. E4 matches v17 but offers no improvement (same in-sample optimism, no generalization gain). E3 recovers Full Paper but PhD stays at 69.4% (vs 93.0%). Keep v17 live unchanged.

---

## NEW DATASET INGESTION — 2026-10-17 (new_001)

**Files:** `~/workspace/user/files/22103.docx` (266KB, md5 dcdb26a8), `22103.pdf` (914KB, similarity report), `22103__1.pdf` (894KB, AI report)

**Inventory/matching:**
- `22103.docx` = ORIGINAL (1275 words, student Azhar Imran Mudassir, BUS 230 Final Assignment)
- `22103.pdf` = Turnitin SIMILARITY report (8pp) — **Overall Similarity 23%** (verified)
- `22103__1.pdf` = Turnitin AI WRITING report (7pp) — **AI 57%** AI-generated only, 0% paraphrased (verified)
- Match key: submission `trn:oid:::3618:145342607`, filename, word count (1277 report ≈ 1275 extracted). Confidence: HIGH. Doc ID: `new_001`.

**Duplicate finding:** ALL THREE are duplicates of existing data.
- AI report jaccard 0.888 (ai) / 0.970 (human) vs existing mass records (source `turnitin_22103__1.pdf`)
- docx jaccard 0.691 vs mass human record (contains docx text + chrome); 0.437/0.404 vs paired fragments
- 22103 already in: `turnitin_22103_paired.jsonl` (2), mass_paired (2), legacy 9-doc eval (in-sample)
- Holdout-31: 0 overlap ✅ (untouched)

**Value note:** docx is the CLEAN original (no report chrome) — existing records are chrome-contaminated fragments. But 22103 is already in-sample in training+eval; re-adding adds no independent signal. Marked `duplicate_do_not_use`.

**Artifacts:** `data/new_data_manifest.csv`, `data/matching_review.csv` (empty), `data/duplicate_review.csv`, `data/extraction_errors.csv` (empty), `data/new_ingest_20261017/new_001_22103_original_{raw,clean}.txt`

**Counts:** inspected 3, matched pairs 1, verified AI labels 1 (57%), similarity-only 1 (23%), unresolved 0, duplicates 1 group, extraction failures 0, eligible for training/testing 0.

**Next action:** None required. Do NOT retrain on new_001 (duplicate). If parent wants cleaner 22103 signal, the clean docx exists at `data/new_ingest_20261017/` — but that is a retraining decision, not ingestion.

---

## PLAGIARISM ENGINE — AUDIT & ARCHITECTURE (2026-10-17)

**Doc:** [plagiarism/AUDIT_AND_ARCHITECTURE.md](sandbox://workspace/ai-detector/plagiarism/AUDIT_AND_ARCHITECTURE.md)

**Audit summary:**
- `app/plag.py` (1,098 lines, LIVE): winnowing K=5/W=4, quote/bib exclusion, NAK report rendering. No MinHash/LSH, no semantic, no fuzzy, no cross-lang.
- Seed index LIVE on Railway: 43,733 docs / ~5.0M fps (arxiv, doaj, europepmc, semanticscholar).
- Mega index: 11.08M docs / 788M fps / 34.75GB on HF (not live).
- Web index: 6,820 docs / 15.7M fps / 576MB, self-test passed (not live).
- Harvest scripts exist (OpenAlex, Crossref, arXiv, universities, Common Crawl WARC); no formal benchmark.

**Architecture (planned, NOT implemented):** L1 exact hashes → L2 winnowing → L3 MinHash/LSH → L4 semantic (candidate discovery only) → passage alignment with URL/title/offsets → quote/bib/common-phrase filters. 7 source categories with explicit limitations (no Turnitin DB, no paywall bypass). Lawful acquisition: Crossref/OpenAlex/Unpaywall/OAI-PMH/Common Crawl WARC. v2 schema adds url/lang/family_id + minhash + passages tables. Phases P0 (benchmark) → P6 (cross-lang pilot).

**Next action:** P0 benchmark suite + eval harness (scripts only, zero cost). Await parent approval before any implementation.

## Batch 2 Ingestion (2026-10-17 ~20:45 PKT) — new_002 through new_006

**Files processed:** 14 (5 groups: 10596_Assignment_2, 10596_Report, 13098_Report, 18376, 18441_task_1)
**Verified pairs:** 3 high-confidence (new_002, new_003, new_006)
**New unique docs eligible:** 0
**Duplicates excluded:** 3 (new_002, new_003, new_006 — all in turnitin_mass_paired.jsonl)
**Unmatched/unverified:** 2 (new_004: two submissions ambiguous + receipt-only; new_005: receipt-only, no original)
**Extraction failures:** 0

**Key findings:**
- 10596_Assignment_2: `*%` AI band, 13% sim, docx 1543w — DUPLICATE (jaccard 0.746)
- 10596_Report: `*%` AI band, 12% sim, docx 1647w exact WC match — DUPLICATE (jaccard 0.785)
- 13098_Report: TWO submissions (0% AI both), "original" is 120w receipt — UNVERIFIED, manual review needed
- 18376: `*%` AI band, "original" is 47w receipt — UNVERIFIED, needs original docx
- 18441_task_1: 28% AI numeric, "original" is 102w receipt — DUPLICATE (jaccard 0.912, ai=28 matches mass)

**`*%` band rule applied:** Per V22, `*%` = 1-19% band, NOT numeric. Never used as numeric label.

**Next action:** Process batch 3 (next 5 unprocessed groups from ~/workspace/user/files/). Skip new_001–new_006.
**v17 untouched:** md5 3e24dc7c79bc40b20c597e37e9aaaaf0 verified.

## New Human Samples (2026-10-17 ~20:42 PKT) — Nisar chat essays

**Essay 1 "Catastrophic Protagonists of Shakespeare Tragedies: Ill-fated Ladies" (197 words): DUPLICATE**
- Jaccard 1.000 (exact) vs `data/verified/human_006_shakespeare.txt` — already saved earlier today.
- NOT saved again. Nisar's "agr phly dya tha to theek" applies.

**Essay 2 "Science exhibition" (179 words): NEW — saved as `data/verified/human_008_science_exhibition.txt`**
- Max jaccard vs all human sources: 0.171 (human_005_science.txt — same topic, different text), 0.063 (drive_human), 0.086 (mass_paired human). Not a duplicate.
- Metadata header: source Nisar 2026-10-17, label human, word count.
- NOTE: human_007 slot skipped (reserved/empty); next free slot is human_009.

**v17 untouched:** no retraining, no dataset modification (additive file only).

## Batch 3 Ingestion (2026-10-17) — 5 groups, 16 files
- **new_007** (18798): AI=0% numeric, Sim=10%, 1281w. HIGH match (trn:oid:::3618:144326459). `duplicate_existing_training` (mass line 11, label=human, consistent).
- **new_008** (19): AI=100% numeric, Sim=1%, 1496w. HIGH match (trn:oid:::3618:132056396). `duplicate_existing_training` BUT **LABEL MISMATCH**: mass file says 'human', verified AI=100%.
- **new_009** (21924): AI=`*%` band (reported_band_1_19, numeric=null), Sim=1%, 3887w. HIGH match (trn:oid:::3618:145773537). `duplicate_existing_training`.
- **new_010** (22107): AI=`*%` band (reported_band_1_19, numeric=null), Sim=10%, 1955w. HIGH match (trn:oid:::3618:145517110). `duplicate_existing_training`.
- **new_011** (23626): AI=35% numeric, Sim=50%, 520w. HIGH match (trn:oid:::3618:144326425). `duplicate_existing_training` (mass label 'human' questionable at 35% AI).
- **new_012** (23626 resubmission): AI=41% numeric, Sim=unknown, no original. MEDIUM match. `unmatched_or_unverified` — '23626 (1).docx' (trn:oid:::3618:144412620, Jun 28) not provided; jaccard 0.494 vs 23626.docx suggests revised resubmission. In matching_review.csv.
- Holdout-31: 0 fingerprint overlap. Legacy 9-doc eval: 0 submission-ID overlap.
- Clean texts: data/new_ingest_20261017/new_00{7-11}_*_original_{raw,clean}.txt
- v17 md5 verified unchanged (3e24dc7c79bc40b20c597e37e9aaaaf0).
- **Next:** Batch 4 — next 5 unprocessed groups from ~/workspace/user/files/ (skip new_001..new_012). NOTE: 3 consecutive batches, 0 new unique docs — if Batch 4 also yields 0, recommend full collection inventory per user instruction.

## FULL INVENTORY — 473 Files (2026-10-17)
- **Total:** 473 files (472 + 1 partial upload), 0 inaccessible
- **Formats:** 399 PDF, 69 DOCX, 1 partial, 1 ZIP, 1 7Z, 1 OGG, 1 other
- **Groups:** 285 document groups; 50 with both original+reports; 205 reports-only; 19 original-only
- **Reports classified:** 207 AI reports, 86 Similarity, 51 receipts, 55 unknown (non-Turnitin), 5 other
- **AI% distribution:** 74×0%, 5×100%, 82×`*%` band (reported_band_1_19), 46×other numeric
- **Duplicates vs mass_paired:** 64 groups (filename) + 0 (content)
- **34 candidates** verified unique vs mass_paired BUT **NOT cross-checked vs anchors/** — many look like known anchors (Full-Paper, PhD, Nikita, XDR, 77701). True new unique = UNKNOWN.
- **Label conflict:** new_008 (19.docx) — training says 'human', Turnitin says 100% AI. Marked `label_conflict_manual_review`.
- **Unmatched:** new_004, new_005, new_012 (in matching_review.csv)
- **Eligible for training/testing:** 0 confirmed
- **Artifacts:** inventory_472.csv, document_report_pairs.csv, duplicate_groups.csv, label_conflicts.csv, eligible_unique_documents.csv, inventory_summary.md
- **v17 md5** `3e24dc7c79bc40b20c597e37e9aaaaf0` verified unchanged. No training/deploy.
- **Next action:** Cross-check 34 candidates against `data/anchors_new_oct16/`, `data/anchors_oct16r2/`, legacy eval, WhatsApp docs before any training/testing use.

## PHASE 2 — Cross-check 34 Candidates (2026-10-17)
**Script:** scripts/phase2_crosscheck.py + scripts/phase2_finalize.py
**Output:** data/candidate_crosscheck.csv (34 rows, 15 cols)

**Final status counts:**
- `duplicate_other_dataset`: 14 (Full-Paper, PhD, PhD_updated, XDR_Typhoid, upload, check_nikita, 77701, BDO_Lebanon, AI_Automation_thesis, ECET202_PD2j, Research_Proposal-1, 11454, Hospital_AR1, _28804_CVL398)
- `unique_band_only`: 6 (33647, 5061, 55743, 58078, Dreams_Final_Report_AUB_3, Internship_Portofolio-Yasmina_Nasreddin) — AI report shows *% band only, numeric=null
- `unique_unverified_pair`: 14 (no AI report paired: 48481, AQEEL_DASAR, CVL398__PD2, FULL_THESIS_ABHISHEK, Final_Progress_Report_Guidelines__1, Final_Thesis_July_0_laux, Hasnain_synopsis__1, NIKE_Strategic_Plan, RWH_Thesis_RMT, Student_Welfare_Document, Thesis_FYP, Thesis_Zohaib_final, okeibunor_Godswill_PCandAD, fyp_final_thesis [unreadable docx])
- `unique_verified_numeric_label`: **0**
- `duplicate_locked_test`: 0 (holdout fingerprint check clean)
- `duplicate_existing_training`: 0 (no content dup vs mass_paired above threshold)

**new_008 conflict RESOLVED (not a conflict):** mass_paired has TWO records from turnitin_19__1.pdf — line13 label=ai (highlighted fragment) + line14 label=human (non-highlighted portion). turnitin_ai=100 on both = document-level score (standard paired schema). Submission IDs match (3618:132056396). label_conflicts.csv updated: resolved_no_conflict. Do NOT relabel.

**Verdict: 0 candidates suitable for independent testing.** No unique doc has a verified numeric AI label + confirmed independence. 6 band-only (*%) cannot serve as numeric ground truth. 14 unverified lack AI reports entirely.

**v17 md5** `3e24dc7c79bc40b20c597e37e9aaaaf0` verified unchanged before/after. No training, no deploy, no dataset modification.

**Next action:** None — the ±5 independent-test-set blocker stands. Future path needs NEW verified paired AI docs (original + Turnitin AI report with numeric %), not reprocessing this collection.

## Phase 3 — Low-Similarity Duplicate Flag Review (2026-10-18)

**5 candidates reviewed** from candidate_crosscheck.csv. All Phase 2 "closest_match" files were spurious name-based matches (low Jaccard correctly rejected those pairings). Content comparison vs TRUE alleged duplicates:

| Candidate | Decision | Jaccard | Testing? |
|-----------|----------|---------|----------|
| Personal_statement_PhD_updated.docx | revised_related_version | 0.383 vs legacy eval PhD | NO (leakage) |
| BDO_Lebanon.docx | confirmed_duplicate | 0.980 vs WhatsApp-21 | NO |
| check_nikita.docx | duplicate_other_dataset | Self = Nikita anchor (48%) | NO (anchor) |
| 77701.docx | confirmed_duplicate | 0.917 vs WhatsApp-21 | NO |
| AI_Automation_and_Employment_Thesis.docx | confirmed_duplicate | 0.985 vs family training | NO |

**Result: 0 qualify for independent testing.** Blocker stands — needs NEW verified paired AI docs.

**Artifacts:**
- data/candidate_crosscheck.csv (phase3_decision/evidence/similarity columns added)
- data/low_similarity_review.md (full evidence per candidate)

**v17 md5 verified:** 3e24dc7c79bc40b20c597e37e9aaaaf0 (unchanged)

**Next action:** None from this collection. Await new verified paired AI documents from Nisar.

## Human-Writing Sample Collection (2026-10-18)

Protocol: pre-ChatGPT academic papers for independent human-writing (specificity/FPR) evaluation. Collection at `data/human_eval_collection/` (pdfs/, texts/, metadata/, manifest.jsonl). NOT training data. v17 untouched.

- HEC-001: Alliez (2015), Theory Culture & Society — abstract only (197w), PRE_CHATGPT_HUMAN_CANDIDATE
- HEC-002: Osborne (2015), Theory Culture & Society — abstract only (159w), PRE_CHATGPT_HUMAN_CANDIDATE
- HEC-003: Davies (2013), Theory Culture & Society — abstract only (205w), PRE_CHATGPT_HUMAN_CANDIDATE
- HEC-004/005: added by parallel agent (SAGE papers)
- HEC-006: World Bank "International Essay Competition 2007" report + 10 winning youth essays — 126pp, 42,205w full text, clean extraction, PRE_CHATGPT_HUMAN_CANDIDATE, suitable. PDF: data/human_eval_collection/pdfs/HEC-006.pdf, text: data/human_eval_collection/texts/HEC-006.txt

**Next action:** Await more papers from Nisar (PDF/URL, one at a time).

## HEC-010 (2026-10-18)
- Wiley/Social Anthropology book review: Ronald Stade on Appadurai's "Fear of Small Numbers" (2007 issue / 2008 online)
- DOI 10.1111/j.0964-0282.2007.00023_1.x verified via Crossref; metadata only (paywalled, no text)
- Classification: PRE_CHATGPT_HUMAN_CANDIDATE; conditionally_suitable (needs lawful full text)
- Manifest: 10 records (HEC-001..HEC-010)

## HEC-011 (2026-10-18) — ResearchGate/PLoS ONE paper collected
- Title: "Looking for Landmarks: The Role of Expert Review and Bibliometric Analysis in Evaluating Scientific Publication Outputs"
- Authors: Liz Allen, Ceri Jones, Kevin Dolby, David Lynn, Mark Walport (Wellcome Trust, London)
- Published: PLoS ONE 4(6):e5910, epub 2009-06-18, DOI 10.1371/journal.pone.0005910, PMCID PMC2695409, PMID 19536339
- Open access CC-BY. Full text: JATS XML via EuropePMC REST API (lawful); PMC PDF blocked by bot challenge (not bypassed); ResearchGate full-text requires login (not used) — only publicly visible abstract/metadata read there.
- Words: main text 4,078 (4 sections: Intro/Methods/Results/Discussion) + abstract 246; 21 refs.
- Classification: VERIFIED_HUMAN (named authors, contribution roles, peer review, 2009, 13+ yrs pre-ChatGPT).
- Duplicate check: none (DOI/title absent everywhere; max jaccard 0.121 vs verified/ corpus).
- Artifacts: data/human_eval_collection/texts/HEC-011_source.xml, texts/HEC-011.txt. Manifest: 11 records (HEC-001–011).
- v17 md5 3e24dc7c79bc40b20c597e37e9aaaaf0 unchanged.
