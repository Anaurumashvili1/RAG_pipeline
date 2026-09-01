---
tags: [rag, handoff, context, unitn, thesis]
type: handoff
updated: 2026-08-26
---

# UniTn RAG — project handoff

> [!info] Paste this whole note into a new conversation to bring it up to speed.

---

## The project

I'm building a retrieval-augmented QA system over University of Trento web
content, as an internship project that forms the basis of my thesis. It's
bilingual Italian/English, about 59,000 documents, and answers questions about
admissions, courses, deadlines, procedures and services with citations back to
source pages. The code is a Python package, `src/unitn_rag/` — `config`, `text`,
`data`, `chunking`, `embeddings`, `indexing`, `retrieval`, `prompts`, `llm`,
`pipeline`, `evaluation` — driven by scripts in `scripts/` and configured
entirely through `config.yaml`. It was ported from a Colab notebook, and that
port is itself part of the thesis: the notebook's evaluation was measuring
things it didn't claim to (hit@5 was computed over a list truncated to 5, so it
couldn't fail; the context was built twice with different truncation, so cited
sources could differ from what the model actually read).

## Machines

Three, and this matters constantly. My **Mac** holds the git repo and is where I
edit; it's the only machine with my SSH key, so file transfers route through it.
**`apa-dipsco` (10.216.20.126)** is a 24-core CPU server holding the crawler in
`~/scraper` (not in git) and the original corpus. **`gx10-3` (10.216.20.140)` is
a GB10 Grace-Blackwell box with 121GB unified memory, 819GB disk, aarch64, CUDA
13 — this is where indexing and evaluation now happen. It needs the UniTN VPN.
Code goes to it via `./sync.sh gpu` (rsync, excludes corpus/indexes/`.env`), and
the venv is `~/rag-venv`, activated per session. Generation runs on
`api.matita.net/ollama-mac1/v1`, an Ollama instance behind nginx, key in `.env`.
Model availability there is per-host and transient — `/v1/models` lists a
catalogue, not what's resident, and everything else returns
`401 "not loaded on machine"`. I'm using `llama3.2:latest` (3B) for development;
it's too weak for real evaluation. The platform serves chat only, `501` on
embeddings, so embedding runs locally.

## Corpus

Raw crawl 201,550 → cleaned 62,040 → plus a delta crawl → `dataset.v2.jsonl` →
about **59,000 documents after loader filters**. Roughly 60% Italian, ~11,000
IT/EN translation pairs, 18,000+ PDFs, median document 1,540 characters, and 81%
of the text sits in documents over 8,000 characters — 96% of those being PDFs
with no headings. Two recovery efforts fed into it: **OCR** of 873 scanned PDFs
with no text layer (668 recovered, 13.75M characters; teseo's 128 abandoned
because OJS serves HTML rather than the file), and a **delta crawl** of three
departments — Law, DICAM, Industrial Engineering — which were missing from the
crawler's allowlist entirely and are now recovered (2,330 pages, plus 100 more
OCR'd PDFs).

## What's been fixed, and the pattern

Every significant bug found has been the same shape: **the loader interpreted a
crawler field as meaning something slightly different from what it meant, and
failed silently.** Five instances. (1) Freshness was inactive on 51% of the
corpus because the loader re-derived `effective_year` instead of using the
crawl's, and ~200 documents were dated 2027–2099, which under `1/(1+age)` scored
*maximum* freshness. (2) `clean_text` destroyed every newline, so
`SentenceSplitter`'s `paragraph_separator` could never match and chunk
boundaries ignored document structure. (3) 2,468 documents had the wrong
language — Drupal serves unaliased `/node/N` under the site default, so English
bodies carried `<html lang="it">`; and separately, *every query* resolved to
English because the query detector required 20+ words and questions have 5–12.
(4) 250 PDFs were letter-spaced (`W e l c o m e  t o`) because design tools
position each glyph and pypdf emits a space between them — deterministically
repairable, `long_token_ratio` 0.03 → 0.66. (5) **`duplicate_of` is a canonical-
URL declaration, not an observed duplicate**: 2,426 records carry it, *none*
shares `content_sha256` with any other document, and 87% name a canonical that
was never fetched — so dropping on the flag deleted 2,079 documents whose text
exists exactly once. All fixed, 43 tests passing in `tests/`.

## Design decisions

BGE-M3 for embeddings (multilingual, cross-lingual, 1024-dim), **fp16 on CUDA**
— measured 50 → 231 chunks/sec on GB10, a 4.6× difference and the single largest
speedup available. Chunking is 512 tokens / 100 overlap with a TITLE/SOURCE/
LANGUAGE/YEAR header injected into the embedded text, because a query matching a
page title otherwise can't reach the chunk holding the fact. **Hybrid semantic
chunking** is implemented but gated on `semantic_min_chars: 8000` — the median
document is 1,540 characters, so applying it corpus-wide would fragment
single-topic pages; above 8k it covers 81% of the text. FAISS `IndexFlatIP`,
exact, ~400k vectors. Retrieval pulls top-20 chunks, collapses them by
`doc_group_id` (which pairs IT/EN translations), applies `1/(1+age)` freshness
decay and a ×1.10 same-language nudge (deliberately a nudge, not a filter, so an
English question can still be answered from an Italian-only page), and passes 5
distinct documents to the LLM.

## Where things stand

Indexes are being rebuilt as `storage/idx_fixed2` (`--semantic-min-chars 0`, the
ablation control) and `storage/idx_sem8k` (semantic), ~30 minutes each on GPU.
Two real queries have been tested: a *laurea magistrale* question answers
correctly from the right page; a *rimborso missioni* question retrieves the
right regulations but the 3B model refused — which is `retrieved_but_refused`,
the exact metric the redesign targets, reproduced live. A separate finding: with
mostly-English context, an Italian question got an English answer, because the
context language outweighed a generic prompt rule; fixed by naming the target
language explicitly in the last line of the user message.

The **evaluation set is the bottleneck**. The old one has 46 questions but
records URLs from an older version of the site — Drupal aliases changed
(`/en/867/nanoscience` → `/node/867`) and programme slugs were renamed
(`computer-science` → `computer-science-master`), and `hit@k` compares URLs
exactly, so correct retrievals were scoring zero. `scripts/fix_eval_urls.py`
resolves them by title and host: ~32 auto-resolved, ~8 need a human choice,
~4 genuinely absent (those become a free abstention test set).
`scripts/eval_candidates.py` builds a worksheet for a new, larger set —
harvesting ~190 real questions from FAQ PDFs plus stratified targets by category
— on the principle that code selects *what to write about* and never writes a
question.

## Next

Finish the two index builds, resolve the remaining eval URLs, run the ablation
(fixed vs semantic vs article-number splitting for regolamenti), then decide on
reranking versus ColBERT. The diagnostic that decides it: if hit@20 is high and
hit@1 is low, the problem is ranking and a cross-encoder reranker is the cheap
fix; if hit@20 is also low, it's recall and needs late interaction or hybrid
BM25. Worth knowing that **BGE-M3 natively emits ColBERT-style multi-vector
output**, so late interaction is available without adopting an English-only
ColBERTv2 model — and as a reranker over the top-20 it needs no new index at all.

## Conventions

Edit locally, push, pull or rsync to the server — two-way editing caused a
divergence once already. Long jobs under `screen`. `.gitignore` excludes `.env`,
`*.jsonl`, `storage/`, `results/`, `.venv/` (the corpus is 713MB; GitHub rejects
files over 100MB). Every ablation arm gets its own `--index-dir`;
`index_manifest.json` records model, dimension, dtype and chunking settings; and
`run_metadata` stamps every results file, so a number can always be traced to
what produced it.
