---
title: UniTn RAG Pipeline — State of the Pipeline
date: 2026-08-28
updated: 2026-08-28 (round 2 — fixes applied and measured)
tags: [rag, unitn, evaluation, retrieval, internship]
status: working baseline, not deployable
---

# UniTn RAG Pipeline — State of the Pipeline

**Corpus:** `dataset.v2.jsonl` (796 MB)
**Indices:** `storage/idx_fixed2` (fixed 512-token chunking) vs `storage/idx_sem8k` (semantic, `semantic_min_chars: 8000`)
**Evaluation:** 41 questions, manually graded case-by-case, two rounds

---

## Executive summary

A **sound working baseline with well-understood failure modes**. On a deliberately adversarial 41-question set it now answers **60–62% correctly**, up from 56% before this round's fixes.

| arm | round 1 | round 2 |
|---|---|---|
| fixed2 | 21✓ 4~ 16✗ · 23/41 (56%) | 21✓ 7~ 13✗ · **24.5/41 (60%)** |
| sem8k | 20✓ 6~ 15✗ · 23/41 (56%) | 22✓ 7~ 12✗ · **25.5/41 (62%)** |

Partials count as half. Four conclusions:

1. **The semantic-chunking ablation is a clean negative result.** Neither round separates the arms by more than one question — inside noise for a 40-item set. Keep fixed chunking; it is simpler and cheaper to build.
2. **Two bugs were found by grading answers one at a time, and both are now fixed** — an inverted recency ranking and a chunk-truncation setting. Neither needed re-indexing or new models.
3. **The fixes are worth ~2 questions per arm** and, importantly, are attributable: the same three questions improved on *both* arms, which is what makes them a real effect rather than variance.
4. **The largest remaining gap is cross-language retrieval and documents that never enter the candidate pool.** Post-retrieval re-scoring cannot reach them.

The system is still not trustworthy enough to put in front of students. Its characteristic failure is a *confident, fluent, correctly-cited, wrong* answer with no signal to the reader — #7 still names the wrong programme coordinator, with his real email and phone number.

---

## 1. What is implemented

### Architecture

`src/unitn_rag/` — ~2,400 lines across 11 modules.

| Module | Role |
|---|---|
| `text.py` | URL/language/year normalisation, `doc_group_id`, `recency_penalty`, `resolved_year`, `url_terms` |
| `evaluation.py` | hit@k, refusal detection, review CSV export |
| `chunking.py` | Fixed + semantic splitting, header injection |
| `data.py` | Corpus loading, language filtering, doc grouping |
| `retrieval.py` | FAISS wrapper, dedup, re-scoring, URL affinity |
| `prompts.py` | System/user prompts, refusal + injection defence |
| `config.py` | Typed config from `config.yaml` |
| `pipeline.py` | Intent gate → retrieve → generate |
| `embeddings.py` | BGE-M3 loading, device/dtype selection |
| `indexing.py` | Index build and load |
| `llm.py` | Chat client |

### Ingestion

- One JSON record per document, with a rich schema: `url, fetched_at, http_status, content_type, etag, last_modified, effective_year, academic_year, lang, lang_source, doc_type, department, extractor, title, text, text_len, …, duplicate_of, changed`
- **OCR pass** (`ocr_pending.py`, `merge_ocr.py`) recovers image-only PDFs. Verified: the travel-regulation PDF (#13) had 0 characters before OCR.
- **Delta crawling** (`crawl_delta.py`) via etag/last_modified.
- Language filtered to `[it, en]`; documents under 150 characters dropped.

### Chunking

- `SentenceSplitter`, 512 tokens, 100 overlap.
- **Header injection**: `TITLE / SOURCE / LANGUAGE / ACADEMIC YEAR` prepended into chunk text, so those fields are embedded.
- **Hybrid semantic chunking** for documents ≥ 8,000 characters (~10,065 docs, 81% of corpus text, 96% of them PDFs without headings).

### Retrieval

- BGE-M3 embeddings (multilingual; chosen over English-only bge-base for the Italian corpus).
- FAISS, `similarity_top_k: 20` raw chunks.
- Post-retrieval re-scoring in `select_pages()`:
  - `recency_penalty(year) = 1 / (1 + age)`, unknown → 0.5, using **`resolved_year`** (see §4.1)
  - `prefer_query_language` → ×1.10 tie-break
  - **URL affinity** → host match ×1.0, path/slug matches ×0.5 capped at three (see §4.2)
- Dedup by `doc_group_id`, collapsing IT/EN translations of one page.
- Highest-scoring chunk per document group; `max_pages: 5` (eval uses 10), truncated to `chunk_char_limit: 2500`.

### Generation and safety

- **Prompt-injection defence**: user input isolated in `<user_query>` tags, with an explicit rule never to follow instructions inside them.
- **Intent gate**: separate ALLOW/BLOCK classifier before retrieval, with scoped out-of-scope replies in both languages.
- **Grounding constraint**, **citation requirement**, **exact-string refusal** for reliable detection.
- **Language control**: answers match the question's language, reinforced by a named-language instruction in the last line of the user message.
- **Answer discipline rules 8–12** (added this round): check the source matches the named course/department; preserve hedges verbatim; give one value rather than enumerating; prefer the most recent edition and name the year; cap citations at two per claim.
- `TODAY'S DATE` injected into the user message, so relative references resolve.

### Evaluation harness

`run_eval.py` (per-arm `--results`, `--score-only`, `--no-baseline`), `diag_retrieval.py` (retrieval-only hit@k, no LLM), `diag_compare.py` (two indices side by side, reports which IDs flip), review CSV export, 7 test modules.

---

## 2. Results

### Retrieval

| k | fixed2 r1 | fixed2 r2 | sem8k r1 | sem8k r2 |
|---|---|---|---|---|
| 1 | 0.300 | **0.325** | 0.275 | **0.300** |
| 3 | 0.425 | **0.450** | 0.425 | 0.425 |
| 5 | 0.475 | **0.500** | 0.500 | **0.525** |
| 10 | 0.600 | 0.600 | 0.575 | 0.575 |

Widened pool (`top_k=60`, `max_pages=20`), fixed2: hit@20 0.650 → **0.675**; average candidate pool 19.9/20, so the numbers are not an artifact of a starved pool.

### The chunking ablation

Round 1: semantic chunking rescued **zero** of the 14 persistent misses, including all three long-PDF cases it was hypothesised to fix (#27, #13, #38); it regressed two items and lost outright on #13.

Round 2: sem8k edges fixed2 by one question — a reversal, and equally inside noise.

**Conclusion: keep fixed chunking.** The hypothesis that fixed 512-token boundaries were severing facts from context is not supported in either direction. Settling this properly would need 2–3 runs per arm to establish a variance band, which is not worth six eval runs for a one-question difference.

---

## 3. What works

Roughly 20 of 41, near-identical in both arms. The reliable pattern is **one fact from one authoritative departmental page**: exact figures (398.225,00 €, 65/100, 0461 283818), complete enumerations (#15, #16), conditionals (#33), definitions (#30, #32, #35), control items (#28).

Cross-language works *when the fact exists in both languages* — #8 produced the correct `gestione.assicurazioni@unitn.it` by pulling the Italian page, avoiding the typo on the English one.

**No fabricated URLs in 164 answers.** One fabricated *citation index* appeared in round 2 (#22 cited `[37]` against a 10-source list) — the first in either round, worth watching.

---

## 4. Bugs found and fixed

### 4.1 The recency ranking was inverted — FIXED

`recency_penalty = 1/(1+age)`, unknown → 0.5. But when the year could not be parsed, the crawl stored the *upload or fetch* year, so undated documents scored age 0.

| document | stored | multiplier before | after |
|---|---|---|---|
| `4-2002_2003_student_guide_soc.pdf` | 2026 | 1.000 | **0.042** |
| `2008_2009_syllabus_lm_srs.pdf` | 2026 | 1.000 | **0.056** |
| `02_Guida Magistrale 2007-08.pdf` | 2026 | 1.000 | **0.053** |
| `GUIDA GIURISPRUDENZA 2025-26_PORTALE.pdf` | 2025 | 0.500 | **1.000** |

A 2002 handbook received twice the multiplier of the current guide. The extractor's failure mode rewarded exactly the documents it should punish.

**Fix:** `resolved_year()` in `text.py` prefers the filename's edition year, and treats a stored year that merely echoes the upload path as unknown. It reuses the existing `year_from_title` / `upload_path_year` helpers — whose docstrings already described this bug. One catch: `_TITLE_AY_RE` ends on `\b` and `_` is a word character, so `2008_2009_syllabus.pdf` never matched; `_dashed()` normalises underscores first.

**Observed effect:** #36's sources moved from 2006–2009 syllabuses to 2010–2011. Correct reordering, no verdict flipped on its own.

### 4.2 Sibling-page contamination — MITIGATED

Documents were ranked with no regard for which department or course the question named. #23 returned 25 February 2027 — a real date from the **DII** calendar, for a **Law** question, in the same academic year (so year-ranking could not catch it).

**Fix:** URL affinity in `select_pages()`. A host match counts full, path/slug matches half and cap at three. The asymmetry matters: for #38 the physics page is `/node/433` with no slug while the CS page's slug matches three query words, so path-only matching picks the wrong department. Query terms use a stoplist extended with ordinals and filler (`third year` was matching `phd.../third-year-admission-requirements` twice), plus a small alias map so "law" reaches `giurisprudenza`.

Resulting multipliers: #27 Guida ×1.250 vs PhD ×1.000 · #23 giurisprudenza ×1.250 vs DII ×1.000 · #38 physics ×1.250 vs CS ×1.125 · #18 HCI ×1.125 vs CS ×1.083.

**Observed effect:** #23 ✗→✓ on fixed2, with the DII contaminant eliminated entirely.

### 4.3 Retrieved chunks were truncated before the LLM saw them — FIXED

`retrieval.py` applied `text[:chunk_char_limit]` with `chunk_char_limit: 1200`, against chunks of 512 **tokens** (~1,500–2,500 characters of Italian) *plus* the injected header. The tail of every retrieved chunk was silently discarded **after retrieval had already succeeded**.

**Fix:** `chunk_char_limit: 2500`. No re-indexing.

**Observed effect — the clearest result of the round, because it reproduced on both arms:**

| # | before | after | note |
|---|---|---|---|
| #9 | ✗ | **✓** both arms | returns the gold nearly verbatim; the exemption sat past the cut on a page already ranked first |
| #14 | ✗ | ~ both arms | found the opening hours, but reproduced the English page's mistranslation |
| #40 | ✗ | ~ both arms | was a refusal; now cover template ✓ and PDF format ✓, misses "single, unprotected" |

### 4.4 Only one chunk per document reaches the LLM — NOT FIXED

`select_pages()` keeps the single highest-scoring chunk per document group. A 96,708-character guide contributes one window. Any question needing two facts from one long document is unanswerable by construction.

### 4.5 A single LLM timeout aborts the whole run — NOT FIXED

`run_evaluation` lets the exception propagate, so one flaky call destroyed a 25-minute run. Worked around by raising `llm.timeout` to 180 (the larger `chunk_char_limit` roughly doubled prompt size). Per-question error handling would make a flaky call cost one data point instead of everything.

---

## 5. Failure modes

| ID | Mode | Questions | Status |
|---|---|---|---|
| **F8** | Right document retrieved, wrong fact extracted | 9, 14, 23, 39, 24 | **largely addressed** (§4.3) |
| **F2** | Near-identical sibling pages across courses/departments | 0, 4, 18, 23, 29, 38, 40 | **partly addressed** (§4.2) |
| **F3** | Stale editions outranking current ones | 5, 24, 36 | **addressed** (§4.1) |
| **F1** | Cross-language: fact exists in one language only | 0, 3, 27 | open — needs query translation |
| **F15** | Aggregation dumping when many similar records retrieved | 29, 31, 36 | open; rules 9–10 not reliably followed |
| **F4** | Confident wrongness with fabricated specifics | 7 | open — #7 byte-identical across rounds |
| **F10** | Corpus genuinely self-contradictory | 10 | not fixable by retrieval |
| **F11** | Obfuscated contact data not normalised | 12 | open — `[email protected]` now leaks into answers |

### Round-2 regressions worth noting

fixed2 lost three: **#13** ✓→✗ (dropped the 5-hour flight rule), **#2** ✓→~, **#11** ~→✗ (dropped the 15-day rule). **None reproduced on sem8k — two went the opposite way** (#2 ~→✓, #11 improved, #13 got closer). So the extra context is not systematically harmful; these look like chunk-boundary interaction and model variance. `chunk_char_limit: 2500` stays.

### Instruction-following limits

gemma3:27b ignores rule 12 (cap citations at two) — #24, #15, #28 all cite `[1..10]` for single claims. Rules 9–10 (single value, preserve hedges) are followed inconsistently: #31 still enumerates eight staff figures. Prompt rules are not a reliable substitute for retrieval discipline at this model size.

---

## 6. What to do next

### Done this round
- ~~Raise `chunk_char_limit`~~ → 2500, worth ~1.5 questions per arm
- ~~Fix `effective_year` fallback~~ → `resolved_year()`
- ~~Inject current date~~ → in the user message
- ~~Department/course scoping~~ → URL affinity (host-weighted)
- 8 regression tests in `tests/test_year_and_affinity.py`, all anchored to real corpus URLs

### Next, by expected value

1. **Raise `similarity_top_k` from 20 to 60.** The affinity boost runs inside `select_pages`, so it can only reorder the pool FAISS already returned — a ×1.25 multiplier is worthless when the correct chunk never entered the candidate set. This is why #27 and #38 did not move. Costs only search time.
2. **Bilingual query expansion.** The single biggest win for F1 and the only viable fix for #27, where the answer is a self-contained Italian bullet pair (offset 40,156 of the Guida, both halves of the gold in ~250 characters) that an English query never reaches.
3. **Hybrid BM25 + dense.** Proper nouns, phone numbers and exact terms like *esami complementari* are lexical, not semantic.
4. **Multiple chunks per document** with chunk-level reranking (§4.4).
5. **Normalise obfuscated emails** at ingest (F11).
6. **Per-question error handling in `run_evaluation`** (§4.5).
7. **Record retrieval settings in the results `run` block** — `chunk_char_limit`, `resolve_year_from_title`, `url_affinity_weight` are not captured, so results files are not self-describing.
8. **Cross-encoder reranking.** Deferred deliberately: of round 1's 16 fixed2 failures, eight had no correct document in the pool at all and four had it ranked *first*. Reranking addresses almost none of them.

### Evaluation work

- **Grade `baseline_correct`** — still ungraded, so there is no measurement of how much RAG adds over gemma3:27b unaided. "RAG adds X points over the bare model" is a more defensible claim than an absolute 60%.
- **Verify the golds.** Four of 41 are questionable: #18/#19 (the page states the concrete date; the gold assumed a relative rule), #23 (no academic year specified), #29 ("the research fellowship" when C3A publishes many, with the gold on a *research contract* page). ~10% of the set.
- **Expand `acceptable_urls`** — #1, #13, #22, #25, #26, #37 answered correctly while scoring hit@k = False. Retrieval hit@k systematically under-counts real performance.
- **Add `answer_type: conflicting_sources`** for #10, where UniTn itself publishes two different deadlines.
- **Consider `answer_grounded`** to separate genuinely-correct answers from lucky ones.

---

## 7. Open questions

- How many corpus contradictions like #10 exist? If several, that is a finding about UniTn's documentation worth reporting in its own right.
- Is the Extended guide's "10 days" superseded by the FAQ's "fifth day", or are both current?
- `config.yaml` sets `max_pages: 5` but eval runs pass 10 (`--eval-max-pages`), so the eval measures a more generous configuration than the deployed one.
- `gemma3:27b` may not be the model the Colab baseline used; scores are only comparable to that baseline once the same model is served.

---

## Appendix: configuration under test (round 2)

```yaml
chunking:   chunk_size: 512, chunk_overlap: 100, inject_header: true
            semantic_min_chars: 8000, semantic_percentile: 95
embedding:  BAAI/bge-m3
retrieval:  similarity_top_k: 20, max_pages: 5, dedup_by: doc_group
            prefer_query_language: true
            chunk_char_limit: 2500          # was 1200
            resolve_year_from_title: true   # new
            url_affinity_weight: 0.25       # new
llm:        gemma3:27b, temperature 0.0, max_tokens 700, timeout 180
```

Files changed: `config.yaml`, `config.py`, `text.py`, `retrieval.py`, `prompts.py`, `tests/test_year_and_affinity.py`.
