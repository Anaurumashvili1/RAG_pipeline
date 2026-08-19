# UniTn RAG

Retrieval-augmented question answering over University of Trento documents.
Local port of the Colab notebook, restructured for the v2 (internship) system.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then fill in LLM_BASE_URL and LLM_API_KEY
```

Put the crawl output at `data/unitn_crawl_normalized.jsonl` (one JSON object per
line: `url`, `title`, `text`, and optionally `lang`, `hreflang_group`,
`last_modified`), and the evaluation set at `data/evaluation_set.json`.

## Use

```bash
python scripts/build_index.py --max-docs 2000   # quick smoke test
python scripts/build_index.py                   # full index
python scripts/ask.py "When does enrolment open for environmental engineering?"
python scripts/ask.py                           # interactive
python scripts/run_eval.py
python -m pytest tests/ -q
```

Running on uni hardware is a config swap, not a code change: copy `config.yaml`
to `config.uni.yaml`, point the paths at the shared filesystem, set
`embedding.device: cuda`, and pass `--config config.uni.yaml`.

## Layout

```
config.yaml              all paths, model names, hyperparameters
.env                     secrets only (gitignored)
src/unitn_rag/
  config.py              typed config loading + device resolution
  text.py                cleaning, language detection, doc_group_id, effective_year
  data.py                corpus loading, filtering, enrichment
  chunking.py            splitting + metadata header injection + dedup
  embeddings.py          embedding model factory, dimension probing
  indexing.py            FAISS build / persist / load
  retrieval.py           chunk -> distinct document selection
  prompts.py             system + user prompts, refusal constants
  llm.py                 OpenAI-compatible client with retries
  pipeline.py            end-to-end RAG and baseline
  evaluation.py          hit@k, generation metrics, review export
scripts/                 build_index.py, ask.py, run_eval.py
tests/                   fast unit tests, no models or network needed
```

## What changed from the notebook

**Structural**

| Notebook | Here |
| --- | --- |
| `drive.mount` + hardcoded Drive paths | `config.yaml`, paths relative to project root |
| API key pasted in cell 7 | `.env`, loaded via `python-dotenv` |
| `DIM = 768` hardcoded | dimension probed from the embedding model |
| Index rebuilt on every session | persisted, with a manifest recording the model used |
| Rerun cells in order | three CLI scripts |

**Correctness**

- *Duplicated context construction.* `retrieve_context` built a context string,
  then `answer_with_rag` rebuilt it from the same nodes with a different
  truncation (`node.text[:800]`) and the first version was discarded. Reported
  sources could disagree with what the model saw. Context is now built once.
- *hit@k measured on a truncated list.* Retrieval returned at most 5 pages
  before the metrics ran, so hit@1/@3/@5 were computed over the same 5 items and
  hit@5 was structurally capped. Evaluation now retrieves 10 pages and applies
  k afterwards. **Expect the reported numbers to shift** — the new ones are the
  honest ones. `tests/test_retrieval.py::test_hit_at_k_is_not_capped_by_truncation`
  pins this.
- *`max_tokens=256`* truncated longer procedural answers mid-sentence. Now 700.
- *Refusals were never counted programmatically.* `prompts.is_refusal` uses the
  exact refusal constants, so accuracy-on-attempted is computed rather than
  hand-tallied.

**Fixes for the failure modes in the paper**

- *Metadata mismatch.* Every chunk is embedded with `TITLE / SOURCE / LANGUAGE /
  ACADEMIC YEAR` prepended to its text, so a query matching a page title can
  match the chunk itself. This is the direct fix for §7.2's fourth bullet.
- *Facts severed from headers.* Chunk size 256→512, overlap 50→100, paragraph
  breaks preferred as boundaries.
- *Bilingual corpus.* Documents carry `lang` and `doc_group_id`. Retrieval
  deduplicates by group, so the IT and EN versions of one page no longer consume
  two of five context slots; the user's language wins ties, but a materially
  fresher sibling wins over language.
- *Staleness.* `effective_year` is extracted from text ("A.A. 2025/2026"), URL
  path, then `Last-Modified`, and applied as the `1/(1+age)` decay from your notes.
- *Boilerplate.* Identical chunk bodies are stored once, with every URL they
  appeared on recorded in `duplicate_urls`.
- *Guardrails.* An intent classifier short-circuits out-of-domain queries before
  retrieval, and user input is isolated in `<user_query>` tags.

## Not done yet

- **ColBERTv2 / late interaction.** The retrieval layer is still dense FAISS.
  `retrieval.py` is the only file that needs to change.
- **Cross-encoder reranking.** The hook is `select_pages` — rerank before dedup.
- **ANN index.** `IndexFlatIP` is exact and fine at 71k docs; at 1M, switch to
  IVF or HNSW in `indexing.py`.
- **Rate limiting.** Belongs in the API layer, which does not exist yet.
- **hreflang capture.** `doc_group_id` accepts an `hreflang_group` argument and
  will prefer it; the scraper needs to emit it.
