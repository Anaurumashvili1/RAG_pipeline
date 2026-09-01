# UniTn RAG — progress since the scraping phase

Status of the pipeline between the end of crawling and the first index build.
Covers what changed from the Colab notebook, what was added, and what is still
open.

---

## 1. Corpus

### 1.1 Cleaning (completed before this phase)

The raw crawl was reduced from 201,550 to 62,040 documents.

| dropped | count |
|---|---:|
| catalogue (library records) | 61,875 |
| journal plumbing (OJS citation exports, viewer wrappers, logins) | 37,425 |
| duplicates | 32,920 |
| boilerplate | 7,506 |
| empty | 617 |
| refetch | 40 |
| **total dropped** | **140,383** |

Final corpus: `unitn_corpus.final.jsonl` (61,167) plus 873 restored scanned
PDFs = **62,040 documents**, of which 18,403 are PDFs (17,530 with a text
layer, 873 scanned with none).

### 1.2 Schema change

The crawl now emits a much richer record than the notebook expected. Where the
old format carried `url / title / text`, the v2 crawl adds `effective_year`,
`academic_year`, `lang`, `lang_source`, `doc_type`, `department`, `page_count`,
`content_sha256`, and the quality flags `low_content`, `boilerplate`,
`duplicate_of`, `needs_ocr`.

This mattered more than it sounds — see §3.1.

---

## 2. Port from the Colab notebook

### 2.1 Structural

| Notebook | Now |
|---|---|
| `drive.mount` + hardcoded Drive paths | `config.yaml`, paths relative to project root |
| API key pasted into a cell | `.env`, loaded via `python-dotenv`, gitignored |
| `DIM = 768` hardcoded | dimension probed from the embedding model |
| Index rebuilt every session | persisted, with a manifest recording the model |
| Rerun cells in order | CLI scripts (`build_index.py`, `ask.py`, `run_eval.py`) |
| One environment | `config.yaml` + `--config config.uni.yaml` override |

### 2.2 Correctness fixes

- **Duplicated context construction.** `retrieve_context` built a context
  string, then `answer_with_rag` rebuilt it from the same nodes with a
  different truncation (`node.text[:800]`), discarding the first. Reported
  sources could disagree with what the model actually saw. Context is now built
  once.
- **hit@k measured on a truncated list.** Retrieval returned at most 5 pages
  before metrics ran, so hit@1/@3/@5 were computed over the same 5 items and
  hit@5 was structurally capped at 1.0. Evaluation now retrieves 10 and applies
  k afterwards. *The reported numbers will shift; the new ones are the honest
  ones.* Pinned by
  `tests/test_retrieval.py::test_hit_at_k_is_not_capped_by_truncation`.
- **`max_tokens=256`** truncated longer procedural answers mid-sentence → 700.
- **Refusals were never counted programmatically.** `prompts.is_refusal` uses
  the exact refusal constants, so accuracy-on-attempted is computed rather than
  hand-tallied.

### 2.3 Retrieval-quality changes

- **Metadata header injection.** Every chunk is embedded with `TITLE / SOURCE /
  LANGUAGE / ACADEMIC YEAR` prepended, so a query matching a page title can
  match the chunk itself.
- **Chunking.** 256 → 512 tokens, overlap 50 → 100, paragraph breaks preferred
  as boundaries — keeps facts attached to their headers.
- **Bilingual handling.** Documents carry `lang` and `doc_group_id`; retrieval
  deduplicates by group so IT and EN versions of one page no longer consume two
  of five context slots. Query language wins ties; a materially fresher sibling
  wins over language.
- **Staleness.** `effective_year` drives a `1/(1+age)` decay.
- **Boilerplate.** Identical chunk bodies stored once, with every URL they
  appeared on recorded in `duplicate_urls`.
- **Guardrails.** Intent classifier short-circuits out-of-domain queries before
  retrieval; user input isolated in `<user_query>` tags.

---

## 3. Changes made during this phase

### 3.1 Corpus loader — freshness was broken on half the corpus

The loader re-derived `effective_year` from URL and text, ignoring the value
the crawler had already computed. Result: **30,491 documents (51%) had no year
at all**, so the recency decay silently did not apply to half the corpus. A
further ~200 documents were dated between 2027 and 2099 — parse errors from
malformed academic-year strings like `2016/2067` — and under a decay that
rewards recency, a document dated 2099 outranks everything on the site.

`resolve_effective_year()` now trusts the crawl first:

1. `academic_year` (`2025/2026`), rejected unless the span is exactly one year
2. `effective_year` from the crawl, clamped to ≤ current year + 1
3. the original regex, only where the crawl left a gap

| | before | after |
|---|---:|---:|
| documents with no year | 30,491 (51%) | 6 (0.0%) |
| documents dated beyond 2027 | ~200 | 0 |

### 3.2 Corpus loader — quality flags now honoured

The crawler saw the raw HTML and decided `low_content` / `duplicate_of` /
`boilerplate`; the loader ignored all three. Now dropped by default, each with
an off switch in `config.yaml`:

- `duplicate_of` — 2,426 documents
- `low_content` — 1,526 documents
- `boilerplate` — 0 in the current corpus (already removed during cleaning)

`Doc` also carries `doc_type` and `department` for future metadata filtering.

**Corpus after loading:** 57,657 documents · 46,680 doc groups · 10,977
translation pairs · 34,965 IT / 22,692 EN · ~494M characters.

### 3.3 LLM endpoint

Moved off the local MLX server onto the university platform
(`api.matita.net`, Ollama 0.32.5 behind a proxy, OpenAI-compatible at `/v1`).

Added **`scripts/check_llm.py`**, which isolates the three ways this fails:
`.env` missing, URL/key wrong, or model name wrong.

Finding worth recording: `/v1/models` lists the platform **catalogue** (10
models), not what is resident on the machine. `ollama-mac1` currently serves
only `llama3.2:latest`; everything else returns
`401 "Model X not loaded on machine"`. `gpt-oss:20b` — the model the notebook
used, and therefore the one that keeps eval numbers comparable — is presumably
on another host.

**The platform serves chat only.** Both `/api/embed` and `/v1/embeddings`
return `501 Not Implemented`, so embedding has to run locally.

### 3.4 OCR — new (`scripts/ocr_pending.py`)

873 PDFs were in the corpus as zero-text records. The crawler had kept
extracted text and discarded the PDF bytes, so this runs in two resumable
phases: **re-fetch**, then **rasterise + OCR**.

Stack: `pdftoppm` at 300 DPI → Tesseract 5.3.4 with `-l ita+eng`.

Design decisions:

- **`ita+eng` together.** Older UniTn scans mix languages within a page.
- **One page in memory at a time.** The largest scan is 112 pages; converting
  it whole at 300 DPI would put ~2 GB of PNGs in `/tmp`.
- **Both phases resumable.** An interrupted run costs one document.
- **SHA-256 verification.** Every re-fetched file is checked against the
  `content_sha256` the crawler recorded.

Three problems found by the first 20-document sample:

1. **AppleDouble stubs.** Files named `._name.pdf` — macOS metadata forks left
   on a web server — are served as `application/pdf` but contain no document.
   8 of the first 20 were these. Fetch now validates the `%PDF` magic bytes
   rather than trusting `Content-Type`. 20 exist in the corpus, all under
   `disi.unitn.it/locigno/didattica/`.
2. **SSL verification failures** on `disi.unitn.it`. Added `--insecure`, which
   retries unverified *only* on a verification failure and accepts the result
   *only* if the SHA-256 matches the crawl — unauthenticated transport, but
   content proven identical to the original fetch.
3. **Sampling order.** Sorting purely by page count floated every record with a
   missing `page_count` to the front, so the sample saw only the records we
   knew least about. Known page counts now sort first.

**Noise detection.** The initial quality check — characters per page — catches
blank scans but misses the more dangerous failure: Tesseract run over a graphic
poster hallucinates letterforms and produces *more* text than real prose. One
event poster scored 2,720 chars/page of pure noise (`"Cee] = ni Tri + d a rai =
È n ki i m n n a i |g"`), the highest in its batch.

A stopword-frequency check was tried and **failed** — noise contains enough
stray `a`, `i`, `e`, `di` to score 0.121, indistinguishable from real text. What
separates cleanly is **token length**: the share of tokens with 4+ characters
was 13.8% for noise versus 52–83% for everything else. Threshold set at 0.35.

Validated against the full run: the distribution has 4 documents at 0.20–0.30,
**nothing at 0.35**, then the real distribution starting at 0.40 and peaking at
0.60–0.65. The threshold sits in a genuine gap, though a narrow one — the ~20
documents in the 0.40–0.45 band are worth reviewing at some point.

**Results** — 39 minutes, 12 workers on 24 cores:

| | count |
|---|---:|
| documents OCR'd | 699 |
| pages processed | 7,379 |
| usable (not empty, not noise) | **668** |
| flagged empty | 27 |
| flagged noise | 4 |
| **characters recovered** | **13,754,954** |
| fetch failures | 154 |
| documents still without text | 205 |

**Failures are almost entirely one host.** 128 of 154 are `teseo.unitn.it`,
which returned HTML (`<!DOCTYP`) instead of PDFs — the OJS platform serves a
viewer wrapper, not the file, to a plain HTTP client. Excluding teseo, only 26
documents failed: a 97% success rate. **Decision: teseo is abandoned.** Those
records keep their empty text and are filtered out by `min_chars`.

Content recovered includes *Quaderni del Dipartimento di Politica Sociale*
(1983 onward — confirming the earlier guess about the scanned sociologia PDFs),
`lavoraconnoi` concorso exam materials, and administrative documents such as
`regolamento missioni`, the travel-expense regulation — exactly the kind of
document a university assistant is asked about, and until now a zero-text
record.

### 3.5 OCR merge (`scripts/merge_ocr.py`)

Writes recovered text back into the corpus. Never edits in place; produces
`dataset.ocr.jsonl` (62,040 rows, unchanged count — fields change, not records).

One detail that would have silently voided the entire OCR effort: the crawler
set `low_content: true` on these records, correctly, since they had no text
layer. `data.py` drops `low_content` documents — so merging text without
clearing that flag would have loaded all 668 recovered documents and then
discarded every one. The merge clears `low_content` and `needs_ocr` for
anything it fills in.

OCR quality fields (`long_token_ratio`, `ocr_pages`, `ocr_suspect`) are carried
into the corpus so retrieval can down-weight marginal scans later without
re-running OCR.

### 3.6 Embedding — benchmarked, decision pending

The server has **24 CPU cores and no GPU**, and the platform will not embed, so
this runs locally on CPU.

An earlier estimate of ~1.2M chunks was **wrong** — it treated `chunk_size: 512`
as characters when llama-index counts tokens. At 512 tokens with 100 overlap the
effective stride is ~412 tokens ≈ 1,600 characters, so ~507M characters gives
roughly **310,000 chunks**.

Measured on the server:

| model | chunks/sec | full corpus | dim |
|---|---:|---:|---:|
| BAAI/bge-m3 | 5.4 | 16.0 h | 1024 |
| intfloat/multilingual-e5-base | 15.0 | 5.8 h | 768 |
| intfloat/multilingual-e5-small | 60.0 | **1.4 h** | 384 |

Thread count and batch size make no difference — PyTorch already uses all 24
cores, and batches of 16/32/64 are within noise of each other. 5.4 chunks/sec is
the honest ceiling for BGE-M3 here; it is an XLM-RoBERTa-large with an 8k
context window and is slow by design.

**Proposed approach:** e5-small while tuning chunking and retrieval (a full
rebuild over lunch, repeatable), BGE-M3 for the final index. Switching is a
config change, not a code change, since the dimension is probed from the model.

### 3.7 Infrastructure

- Project under git, pushed to GitHub.
- `.gitignore` excludes `.env`, `*.jsonl` (the 713MB corpus), `storage/`,
  `results/`, `.venv/`.
- Server sync via SSH **agent forwarding** — the server borrows the Mac's key
  to reach GitHub, so no key material is stored on a shared machine.
- Working rule: edit locally, push, pull on the server. Server-side edits only
  where the tooling requires it.

---

### 3.8 Evaluation metric — `hit@k` could not see a bilingual hit

`select_pages()` collapses an IT/EN pair to one slot by `doc_group_id` and
reports whichever sibling ranked higher. `hit_at_k()` exact-matched a single
`target_url`. So whenever retrieval surfaced the *other* sibling — which it does
by design, not by accident — a correct retrieval was recorded as a miss. This is
the same failure shape as the `duplicate_of` bug: a field meaning one thing
(this document, in one language) used as if it meant another (this fact).

`evaluation.py` now takes a set:

- `targets_of(item)` returns every acceptable URL — `acceptable_urls` as a list
  or as the pipe-joined string the candidate worksheet emits, with `target_url`
  leading it so citation display is unchanged.
- `hit_at_k(sources, targets, k)` accepts one URL, a list, or a pipe-joined
  string. Old call sites keep working.
- Items `hit@k` cannot judge now score `None` rather than `False`, and
  `retrieval_metrics` reports `n` (scored) beside `n_excluded`. A guaranteed
  zero sitting in the denominator is not a measurement.
- The review CSV carries `objective`, so a red cell can be read.

**Which siblings were accepted.** `evalprep/sibling.py` rebuilds the pairing
from `out_links` — A and B are siblings if A links to B, B links back, same
host, different language — and finds a sibling for 11 of the 41 targets, zero
ambiguous, no re-crawl. But a reciprocal link proves two pages are the same page
in two languages, **not** that they carry the same content, so
`evalprep/sibcheck.py` tests each candidate against the gold answer first
(numbers, months and email addresses survive translation; the rest was read).

Nine were accepted. **Two were rejected**, and they matter more than the nine:

- **id 3** — the Italian sibling defers to an annex and never states the 66–110
  scale. The item exists *because* only the English page states it.
- **id 9** — the Italian sibling states the opposite of the gold answer
  (everyone writes a *Relazione finale*) and never carries the exemption.

Accepting those two would have traded a false negative for a false positive,
which is the worse error: a false negative understates a working system, a false
positive hides a real retrieval failure. Both rejections are recorded in the set
as `sibling_rejected` + `sibling_note` so the pairing is not re-applied blind.

**id 12** is marked `answer_only`: the fact is a contact block repeated across
53 documents and the question names no course, so no retriever can pick the
target. It leaves the retrieval denominator and stays in the generation grade.

Output: `data/evaluation_set.v3.json`, 41 items, 40 retrieval-scored, 9 with a
second acceptable URL. `config.yaml` still points `eval_set` at
`data/evaluation_set.json` — one line to switch when the set is final.

Tests: `tests/test_eval_metrics.py`, 14 cases. Also `tests/test_text.py` held a
stale assertion (`declared` beating the URL marker) that commit 07901fd
deliberately reversed and `test_text_v2.py` already covers; rewritten to the
current precedence with the reason attached. 62 tests pass.

### 3.9 Effective year — ranked by how the year was obtained

Two changes, one principle: **a year is only as good as where it came from.**

**An academic year resolves to its end.** `a.a. 2025/2026` is now 2026, not
2025. A guide for a.a. 2025/26 is the current guide through the whole of 2026;
dating it 2025 made `1/(1+age)` score it 0.5 the day it was published, while a
page dated 2026 from its upload path scored 1.0. The correctly tagged document
ranked *below* the carelessly dated one. Applied consistently — the
`academic_year` field, spans in text, spans in filenames — 5,953 documents
move forward one year.

**An upload-path year is the weakest evidence there is.** Drupal serves uploads
from a month-stamped directory, `/sites/cds/files/2025-02/`. That records when
someone put the file on the server. Measured: **5,380 documents carry such a
path and the crawl's `effective_year` equals the path year in every single one**
— so the path is not merely a possible source, it is *the* source for those
documents. Four confirmed wrong: the CEILS transfer guidelines (a 2022 document
uploaded in 2025), the travel regulation (2015 → 2024, and its Alfresco copy
2015 → 2026), the energy regolamenti (2016, 2023 and 2024 editions sharing one
December-2024 upload directory and all dated 2024), and the a.a. 2007-08 guides.

New order in `resolve_effective_year`:

1. `academic_year` from the crawl → end year
2. **what the document states about itself**, when the crawler had only the
   upload path or fell back to the current year
3. the filename's edition label, same condition
4. the crawl's `effective_year`
5. the regex fallback

**What counts as the document stating its own year.** Only two forms, and the
anchoring is the entire difficulty:

- *Emanation decree.* `Emanato con DR n. 480 del 29 luglio 2015` — accepted only
  when the same decree, number and date, appears **at least twice** and its
  first appearance is not introduced by `Visto` / `di cui` / `ai sensi` /
  `modificato`. Every UniTn decreto opens by citing three or four other decrees
  in identical language: a commissioni decree issued in July 2026 cites the
  Statute of 2024 and the Regolamento didattico of 2012. A running page header
  repeats verbatim; a preamble names each decree once. That difference is the
  test. Restricted to PDFs — an HTML page has no running header, and exactly one
  page in the corpus (`disi/node/1603`) would otherwise be backdated to 2015 for
  quoting the Conto Terzi regolamento twice.
- *Explicit revision date.* `Last updated on 22nd December 2022` /
  `Ultimo aggiornamento`. Latest wins, since a page listing several revisions is
  current as of the most recent.

Anything looser reads the wrong year off almost every one of these files. The
current Giurisprudenza guide says *"a partire dall'anno accademico 2011-2012"*
six thousand characters in; the energy regolamenti cite *"ai sensi del D.M. del
16.03.2007"*. Those are history and legal reference, not publication dates — the
first draft of this rule dated the 2026 regolamento to 2007.

**Filename edition labels.** `year_from_title` now also reads a bare year in the
last position of a filename stem — `regolamento-...-2023.pdf`. Position is what
makes it safe: at the end of the stem it names the file, loose in the middle
(`Premio 2019 assegnato`) it is part of a sentence. Returned as-is rather than as
a span, deliberately: the filename claims 2023 and reading it as a.a. 2023/24
would invent a claim it does not make — and would land the file back on the same
year as the upload path already known to be wrong.

**Effect.** 7,642 of 63,470 documents change (12.0%): 5,953 by the end-year
convention, 1,138 re-dated by their own text, 549 by a filename edition. Of the
5,380 upload-path documents, **1,937 (36%) now carry an evidence-backed year**
and 3,443 keep the path year because nothing better exists. The five energy
regolamenti resolve to five distinct years instead of collapsing three into
2024. Tests: `tests/test_year_evidence.py`, 13 cases built from corpus strings;
suite 76 passing.

**Found on the way, not fixed.** The crawl's own `academic_year` is sometimes
wrong, and it sits at priority 1. `economia.unitn.it/node/3390` is titled
*"Academic calendar 2026/27"* and carries `academic_year: 2025/2026`; the same
one-year lag repeats across the 2023/24, 2024/25 and 2025/26 editions, and
`guida-facolta-giurisprudenza-2024-2025.pdf` carries `2022/2023`. Of 967
documents where both a crawl `academic_year` and a filename span exist, **151
(16%) disagree**, and in every sample inspected the filename is right. Preferring
a title-stated span over a body-scraped one is the same principle as everything
above — a document naming its own year beats a year found somewhere inside it —
but it demotes `academic_year` from priority 1, so it is left as a decision
rather than taken.

## 4. Open items

**Before the first full index build**

- **e5 input prefixes.** e5 models require `passage: ` on documents and
  `query: ` on queries. Omitting them does not error — retrieval just gets
  quietly worse. Needs a model-conditional change in `embeddings.py`, since
  BGE-M3 requires no prefix.
- **Index/model mismatch.** A FAISS index at dim 384 is meaningless to a
  1024-dim model. `indexing.py` writes a manifest with the model name; confirm
  it refuses a mismatched index rather than returning nonsense.
- **`data/evaluation_set.json` does not exist.** `run_eval.py` has nothing to
  run against.
- **`build_index.py` has never been executed.** No index exists yet.

**Deferred**

- 128 teseo scanned PDFs — abandoned by decision, recoverable later since
  `pending_ocr.jsonl` still lists them.
- 20 AppleDouble stubs still present in the corpus as empty records.
- `gpt-oss:20b` — needs the endpoint that hosts it, for comparability with the
  notebook's eval numbers.
- Embeddings endpoint on the university platform — worth requesting; would turn
  16 hours into minutes.

**Not started (carried over from the notebook port)**

- ColBERTv2 / late interaction — `retrieval.py` is the only file that changes.
- Cross-encoder reranking — hook is `select_pages`, rerank before dedup.
- ANN index — `IndexFlatIP` is exact and fine at this scale; switch to IVF or
  HNSW in `indexing.py` at ~1M.
- Rate limiting — belongs in the API layer, which does not exist yet.
- `hreflang` capture — `doc_group_id` accepts and prefers an `hreflang_group`;
  the scraper needs to emit it.
