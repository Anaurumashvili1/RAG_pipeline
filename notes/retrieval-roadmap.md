---
title: Retrieval Roadmap — what to build next, and why
date: 2026-08-28
tags: [rag, unitn, retrieval, colbert, hybrid, decision]
status: decision pending on probe results
---

# Retrieval Roadmap

What to change in retrieval next, what each option actually fixes, and in what order. Written after two graded evaluation rounds and the `similarity_top_k` experiments.

---

## The decision in one paragraph

**Do not choose an architecture yet.** Run `probe_rank.py` first. Every option below is justified by a different assumption about *where* failing documents sit in the ranking, and that is currently unmeasured — `hit@k` only tells you they missed the cut, not whether by two places or forty thousand. The probe costs five minutes and eliminates at least two of the four paths. After that, the likely order is: query translation → hybrid sparse → late-interaction rescoring → full multi-vector index, stopping as soon as the failures are gone.

---

## 1. What we actually have to fix

From the round-2 sem8k grading, cross-referenced against retrieval hits (40 retrieval-scored questions):

| group | count | who fixes it |
|---|---|---|
| Retrieval and generation both working | 21 | — |
| Answered correctly from a *different* valid source | 7 | eval set (`acceptable_urls`) |
| Eval-set error, not a system failure | 2 (#29, #13) | eval set |
| **Document retrieved, answer still wrong** | **6** | generation / prompt / model — **no retrieval work helps** |
| **Genuine recall failure** | **8** | this document |

The eight: **#0, #3, #4, #5, #7, #17, #27, #38**

That is the entire addressable set. Perfect retrieval buys at most eight questions, and less in practice — #7 also needs role→person attribution once the contacts page is found, and #5's gold is a negative fact ("transfer to 3rd year is not possible") that the model has hallucinated around even when related pages were retrieved.

### Sub-grouping the eight

| # | question | why it fails | candidate fix |
|---|---|---|---|
| #0 | EU application deadline, Env. Engineering | English query, Italian-only target page | translation, late interaction |
| #3 | HCI thesis grading scale | Italian query, English-only target page | translation, late interaction |
| #27 | complementary exams, 2019 Law cohort | English query, Italian guide; competing "third year" pages | translation, sparse (if translated) |
| #4 | CEILS max recognised ECTS | **unexplained** — see below | probe first |
| #5 | CEILS third-year transfer | **unexplained** — same document | probe first |
| #17 | CEILS final paper deadline | same course, different document | probe first |
| #7 | Civil Engineering coordinator | contacts page never surfaces; person pages dominate | sparse / entity handling |
| #38 | physics external research period | CS sibling page outranks physics | affinity (partly done), sparse |

### The CEILS anomaly — read this before choosing anything

`guidelines_trasfer_b-comparative-european_internation_legal_studies.pdf` contains both #4's and #5's gold in one contiguous ~800-character passage:

> ● 30 ECTS credits for exams/courses in a language other than English
> ● 60 ECTS credits for exams/courses in English
> […] **It is not possible to transfer to 3rd year of the CEILS course.**

And the document is:

- **in the corpus** — 7,575 characters, clean `pypdf` extraction, `low_content: False`
- **English** — same language as both questions
- **short** — below `semantic_min_chars: 8000`, so chunked identically in both arms; the ablation could never have touched it
- **current** — `effective_year: 2025`
- **lexically matching** — the passage contains "CEILS", "transfer", "ECTS", "recognize"

This is dense retrieval failing with none of the conditions that late interaction addresses: no pooling dilution, no language mismatch, no vocabulary gap. **Two, possibly three, of the eight failures are this one document.** If a new retrieval architecture is adopted before understanding this case, there is a real risk of building the wrong thing.

One concrete lead: the document writes **"3rd year"** and **"2nd year"**; the questions say **"third year"**. Grep confirms zero occurrences of the spelled-out ordinals. That breaks lexical matching outright and may weaken dense matching too.

---

## 2. The decision gate

```bash
python scripts/probe_rank.py --index-dir storage/idx_sem8k --eval-set-misses --top-k 2000
```

Reports the true rank of each target over 2000 raw chunks, before dedup and before re-scoring — so it measures what the embedding did, not what post-processing made of it.

### How to read it

| target depth | meaning | path |
|---|---|---|
| **21–100** | embedding matched well, ranking is the problem | **A** — rescoring / reranking |
| **101–1000** | weak but real match | **A** or **B**, depending on the failure type |
| **>1000 or absent** | embedding never matched | **B**, **C** or **D** — rescoring is useless |

Also run the CEILS phrasing comparison, which tests the ordinal hypothesis directly:

```bash
python scripts/probe_rank.py --index-dir storage/idx_sem8k --url-contains guidelines_trasfer \
  --question "Can I transfer into the third year of the CEILS bachelor at Trento?" \
  --question "Can I transfer into the 3rd year of the CEILS bachelor at Trento?" \
  --question "CEILS transfer credit recognition maximum ECTS"
```

If "3rd year" ranks far better than "third year", the fix is ordinal normalisation at index and query time — an afternoon, and it would resolve two questions with no architectural change at all.

---

## 3. Path A — Late-interaction rescoring (cheapest real test)

### What it is

Retrieve a deep candidate set with the existing dense index (200–500 chunks), then rescore those candidates with BGE-M3's ColBERT head using MaxSim:

$$s(q,d) = \sum_{i \in q} \max_{j \in d} E_i(q) \cdot E_j(d)$$

No new model — BGE-M3 already produces these vectors, they are simply discarded today. No new index.

### What it solves

Questions where the correct chunk is *within reach* of dense retrieval but ranked below the cut. Also serves as an honest experiment: if MaxSim scores your failing cases correctly here, that is direct evidence for building the full version. If it does not, you have saved yourself a month.

### What it cannot solve

**It inherits dense retrieval's recall ceiling.** If the pooled BGE-M3 vector never places a chunk in the top 500, MaxSim over those 500 cannot rescue it. This is the same structural objection that applies to cross-encoder reranking, and it applies here with equal force.

### Steps

1. Confirm from the probe that failing targets sit within your intended candidate depth.
2. Load BGE-M3 with multi-vector output enabled:
   ```python
   from FlagEmbedding import BGEM3FlagModel
   model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
   q = model.encode([question], return_colbert_vecs=True)
   d = model.encode(passages, return_colbert_vecs=True)
   scores = [model.colbert_score(q["colbert_vecs"][0], v) for v in d["colbert_vecs"]]
   ```
3. Insert between `self._retriever.retrieve()` and `select_pages()` in `Retriever.retrieve`, exactly where a cross-encoder would go.
4. **Normalise scores to (0, 1].** `select_pages` *multiplies* score by `recency_penalty` and the affinity boost; a negative score multiplied by 0.5 becomes larger, silently inverting the recency ranking.
5. Raise `similarity_top_k` for this path only — rescoring needs a deep pool to be worth anything, unlike the dense-only case where 60 was proven no better than 20.
6. Measure with `diag_retrieval.py` before spending a generation run.

### Cost

A day. No new dependencies, no index, no storage.

---

## 4. Path B — Hybrid sparse (BGE-M3 sparse head)

### What it is

BGE-M3's second head emits a learned weight per token actually present in the text:

$$w_t = \mathrm{ReLU}(W \cdot h_t), \qquad s(q,d) = \sum_{t \in q \cap d} w_t^q \cdot w_t^d$$

Structurally BM25 — a sum over shared terms — but with learned weights, so stopwords are crushed and discriminative terms amplified. Goes into an ordinary inverted index. Combine with the dense score by weighted sum or reciprocal rank fusion.

### What it solves

Exact-match retrieval within one language: proper nouns, phone numbers, email addresses, precise multi-word terms. **#7** (a named coordinator on a contacts page), **#12** (`stem.admissions.int@unitn.it`), and #25/#26-style lookups whose current success is partly luck.

### What it cannot solve

**Cross-language.** Unlike SPLADE, BGE-M3's sparse head performs *no vocabulary expansion* — it weights only tokens present in the text. An English query against an Italian document shares almost nothing. It does **not** fix #0, #3 or #27 on its own; it helps them only after translation.

The partial exception: XLM-R's subword vocabulary is shared across languages and Italian–English academic vocabulary is heavily Latinate, so `complementary`/`complementari` may share a stem token. That is a hope, not a mechanism.

Also note it would **not** rescue #5, because the document says "3rd year" where the question says "third year" — lexical matching needs ordinal normalisation to bridge that.

### Steps

1. Re-encode the corpus with `return_sparse=True` (one extra output from the same forward pass, so no second embedding run if done during a rebuild).
2. Build an inverted index over the sparse weights — a dict of `token → [(chunk_id, weight)]` is adequate at your corpus size; no Lucene required.
3. Normalise ordinals and common abbreviations at index and query time (`3rd`↔`third`, `2nd`↔`second`).
4. Fuse: reciprocal rank fusion is more robust than score-weighted sum, since dense cosine and sparse dot products are not on comparable scales.
5. Validate on the exact-match questions specifically, not just aggregate hit@k.

### Cost

Two to three days, mostly the index build and fusion tuning.

---

## 5. Path C — Query translation

### What it is

Detect the query language, translate it to the other language with the LLM already in the pipeline, retrieve for both, merge the candidate sets.

### What it solves

**#0, #3, #27** — the entire cross-language group. Directly and cheaply.

For #27 in particular this is the only cheap fix: the answer is a self-contained Italian bullet pair at offset ~40,156 of the Giurisprudenza guide, and an English query never reaches it.

### What it cannot solve

Anything monolingual — #4, #5, #7, #17, #38.

### Steps

1. Validate the hypothesis before building. Hand-translate two questions and probe:
   ```bash
   python scripts/probe_rank.py --index-dir storage/idx_sem8k \
     --url-contains "GUIDA%20GIURISPRUDENZA%202025-26" \
     --question "How many complementary exams from the third year onwards, Law, enrolled 2019?" \
     --question "Quanti esami complementari devo sostenere dal terzo anno in poi a Giurisprudenza?"
   ```
   If the Italian phrasing pulls the target into reach and the English one does not, the fix is confirmed.
2. Add a translation call in `Retriever.retrieve` — one extra LLM round trip per query (~1–3 s), or a cached small model.
3. Retrieve for both phrasings, merge by reciprocal rank fusion, then dedup as now.
4. Consider skipping translation when the query is unambiguous, to avoid latency on the majority of queries that do not need it.

### Cost

One to two days. Adds per-query latency, which matters for a chatbot in a way it does not for batch evaluation.

---

## 6. Path D — Full multi-vector index (the ColBERTv2 architecture)

### What it is

MaxSim as the **first-stage** scoring function over every token in the corpus, rather than a rescoring pass. Requires the two pieces of engineering ColBERTv2 contributed:

- **Residual compression** — k-means centroids over all token vectors; each token stored as *(centroid id, residual quantized to 1–2 bits/dim)*. Roughly 25 GB → ~3 GB for this corpus.
- **PLAID pruning cascade** — centroid pruning, then approximate scoring using centroid identities, then exact MaxSim on a few thousand survivors. Without it, MaxSim is a brute-force scan of ~100M vectors per query.

**The ColBERTv2 weights are English-only** (BERT-base, MS MARCO) and would be worse than what you have on an Italian-majority corpus. What transfers is the engineering. Use a multilingual late-interaction checkpoint — `jina-colbert-v2` (89 languages, packaged as a drop-in for ColBERT tooling) or `colbert-xm` — with RAGatouille or `colbert-ai` providing the index.

BGE-M3's ColBERT head is *not* the right vehicle here: it produces good token vectors, but nothing packages them into a PLAID-compatible index, so you would be writing the compression and pruning yourself.

### What it solves

Recall failures that no candidate-set method can reach: a passage surfaces because one phrase matched strongly, with no pooled vector standing between the query and the match. Token-level granularity *and* semantic (cross-lingual) matching in one mechanism — the only option that has both.

### What it cannot solve

The 6 questions where the document was retrieved and the answer was still wrong. Nothing in this document touches those.

### Steps

1. **Only after** the probe shows failing targets beyond any tractable candidate depth, and after Path A shows MaxSim scores them correctly when it can see them.
2. `pip install ragatouille`, index a subset first (one department) to measure build time and quality before committing the full corpus.
3. Index with a multilingual checkpoint; expect hours on the GPU for ~200k chunks.
4. Run it *alongside* FAISS initially, comparing on the same eval set, rather than replacing it.
5. Decide on maintenance: the index must be rebuilt when the corpus changes, and you now have two retrieval systems.

### Cost

Two to four weeks, ~3 GB of index, a multi-hour rebuild per corpus refresh, and a second system to maintain. **This is a significant commitment for an internship deliverable and should be evidence-driven, not theory-driven.**

---

## 7. Comparison

| | A · rescoring | B · sparse | C · translation | D · multi-vector |
|---|---|---|---|---|
| new model | no | no | no (reuses LLM) | yes |
| new index | no | inverted | no | ~3 GB multi-vector |
| effort | ~1 day | 2–3 days | 1–2 days | 2–4 weeks |
| query latency | +ms | +ms | +1–3 s | +ms |
| fixes recall | no | partly | yes (cross-lang) | yes |
| fixes ranking | yes | partly | no | yes |
| target questions | ranking-depth cases | #7, #12, exact match | #0, #3, #27 | #0, #3, #27, possibly #4/#5/#17 |

---

## 8. Recommended order

1. **Run the probe.** Five minutes. Eliminates paths and may reveal the CEILS anomaly is something mundane.
2. **Fix the eval set** — widen `acceptable_urls`, correct the three wrong golds. An afternoon, removes noise from every future measurement, and would have changed how at least two rounds of results were read. Competitive with any pipeline change on value per hour.
3. **Ordinal normalisation** if the CEILS probe confirms it. An afternoon, possibly two questions.
4. **Path C (translation)** — the cross-language group is three questions, the mechanism is understood, and the cost is low.
5. **Path A (rescoring)** — cheap, and doubles as the experiment that justifies or kills Path D.
6. **Path B (sparse)** — for the exact-match group, once translation is in place so it can help cross-language cases too.
7. **Path D** — only with evidence from steps 1 and 5.

## 9. What none of this fixes

Six questions fail with the correct document already retrieved: #10 (the corpus genuinely contradicts itself), #39 (right page, wrong checklist extracted), and partials #14, #23, #24, #31. Those need prompt work, a stronger model, or extraction fixes.

Worth keeping in proportion: the retrieval work above has a ceiling of eight questions out of forty, and generation failures account for six. **Retrieval is not the only thing standing between 62% and a usable chatbot.**
