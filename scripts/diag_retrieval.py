#!/usr/bin/env python3
"""Retrieval-only diagnostic: hit@k with a wider candidate pool, no LLM calls.

Retriever.retrieve() is pure embedding + FAISS search + Python dedup - it never
touches the ChatClient. So the hit@k question the ablation actually asks
(did the right document surface in this arm's chunking) does not need the
~25-minute generation loop run_eval.py pays for on every question.

It also fixes a measurement ceiling in the eval numbers as normally run:
config.yaml's similarity_top_k=20 pulls only 20 raw chunks before doc-level
dedup, and a long PDF can supply 3-4 of its own chunks to that top-20 - so
hit@10 can be capped by how few distinct documents made it into the pool at
all, not by whether the embeddings found the right one. --top-k widens that
pool so hit@20 is a real number, not an artifact.

    python scripts/diag_retrieval.py --index-dir storage/idx_fixed2
    python scripts/diag_retrieval.py --index-dir storage/idx_sem8k
    python scripts/diag_retrieval.py --index-dir storage/idx_fixed2 --show-misses
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.config import load_config              # noqa: E402
from unitn_rag.evaluation import (                     # noqa: E402
    hit_at_k,
    is_retrieval_scored,
    targets_of,
)
from unitn_rag.indexing import load_index              # noqa: E402
from unitn_rag.retrieval import Retriever, detect_query_language  # noqa: E402

K_VALUES = (1, 3, 5, 10, 20)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--eval-set", default=None)
    ap.add_argument("--top-k", type=int, default=60,
                     help="Raw chunks pulled from FAISS before dedup - config.yaml's "
                          "20 is too small a pool to fairly measure hit@20")
    ap.add_argument("--max-pages", type=int, default=20,
                     help="Distinct documents kept after dedup, per question")
    ap.add_argument("--show-misses", action="store_true",
                     help="Print each item that misses at max k, with its objective")
    ap.add_argument("--no-translate", action="store_true",
                     help="Skip query translation for this run, whatever config says")
    ap.add_argument("--rerank", dest="rerank", action="store_true", default=None,
                     help="Force reranking on for this run, whatever config.yaml says")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false",
                     help="Force reranking off for this run")
    ap.add_argument("--recency-weight", type=float, default=None,
                     help="Override retrieval.recency_weight (1.0 original, 0.0 off)")
    ap.add_argument("--recency-unknown", type=float, default=None,
                     help="Override retrieval.recency_unknown - what an undated doc scores")
    ap.add_argument("--no-recency", action="store_true",
                     help="Disable the 1/(1+age) freshness multiplier. A sole-authority "
                          "document that is merely old takes a x0.5 or worse handicap, "
                          "which can outweigh a genuine relevance difference - especially "
                          "once reranker scores are squashed into (0,1] and the spread "
                          "between a great match and a mediocre one is narrow.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.index_dir:
        cfg.paths.index_dir = Path(args.index_dir)
    if args.eval_set:
        cfg.paths.eval_set = Path(args.eval_set)
    cfg.retrieval.similarity_top_k = args.top_k
    if args.rerank is not None:
        cfg.retrieval.rerank = args.rerank
    if args.recency_weight is not None:
        cfg.retrieval.recency_weight = args.recency_weight
    if args.recency_unknown is not None:
        cfg.retrieval.recency_unknown = args.recency_unknown

    print(f"[diag] index          : {cfg.paths.index_dir}")
    print(f"[diag] eval set       : {cfg.paths.eval_set}")
    print(f"[diag] similarity_top_k (raw chunks before dedup): {args.top_k}")
    print(f"[diag] translate: {cfg.retrieval.translate_query and not args.no_translate}")
    print(f"[diag] rerank: {cfg.retrieval.rerank}"
          f"  (backend={cfg.retrieval.rerank_backend}, pool={cfg.retrieval.rerank_top_k})")
    print(f"[diag] recency: {'OFF' if args.no_recency else 'on'}"
          f"  weight={cfg.retrieval.recency_weight}"
          f"  unknown={cfg.retrieval.recency_unknown}")

    eval_set = json.loads(Path(cfg.paths.eval_set).read_text(encoding="utf-8"))
    index = load_index(cfg)
    # Translation needs the LLM, which this diagnostic otherwise avoids. The
    # cache makes it a one-off cost: 40 questions translated once, then free
    # and deterministic across every subsequent run.
    translator = None
    if getattr(cfg.retrieval, "translate_query", False) and not args.no_translate:
        from unitn_rag.llm import ChatClient
        from unitn_rag.translate import QueryTranslator
        translator = QueryTranslator(ChatClient(cfg.llm),
                                     cache_path=cfg.retrieval.translation_cache)
    retriever = Retriever(index, cfg.retrieval, translator=translator)

    hits = {k: 0 for k in K_VALUES}
    n_scored = 0
    n_excluded = 0
    pool_sizes: list[int] = []
    misses_at_max: list[dict] = []

    for item in eval_set:
        if not is_retrieval_scored(item):
            n_excluded += 1
            continue
        n_scored += 1

        question = item["question"]
        lang = detect_query_language(question)
        pages = retriever.retrieve(
            question, query_lang=lang, max_pages=args.max_pages,
            apply_recency=not args.no_recency,
        )
        sources = [p.url for p in pages]
        pool_sizes.append(len(sources))

        targets = targets_of(item)
        for k in K_VALUES:
            if hit_at_k(sources, targets, k):
                hits[k] += 1

        if not hit_at_k(sources, targets, max(K_VALUES)):
            misses_at_max.append({
                "id": item.get("id"), "question": question,
                "objective": item.get("objective", ""),
                "pool_size": len(sources), "targets": targets,
            })

    print(f"\n[diag] scored {n_scored}, excluded {n_excluded} "
          f"(answer_only / out_of_scope)\n")
    print(f"{'k':>4}  {'hit rate':>10}  {'count':>8}")
    for k in K_VALUES:
        rate = hits[k] / n_scored if n_scored else 0.0
        print(f"{k:>4}  {rate:>10.3f}  {hits[k]:>4}/{n_scored}")

    avg_pool = sum(pool_sizes) / len(pool_sizes) if pool_sizes else 0
    print(f"\n[diag] avg distinct documents surfaced per question: {avg_pool:.1f} "
          f"(cap requested: {args.max_pages})")
    if avg_pool < args.max_pages * 0.7:
        print("[diag] pool is consistently smaller than the cap - long documents "
              "are likely crowding out the raw-chunk pool before dedup even runs")

    if args.show_misses and misses_at_max:
        print(f"\n--- {len(misses_at_max)} items miss even at k={max(K_VALUES)} ---")
        for m in misses_at_max:
            print(f"\n  id {m['id']}  pool={m['pool_size']}  {m['question'][:80]}")
            if m["objective"]:
                print(f"      objective: {m['objective'][:140]}")
            print(f"      target(s): {m['targets']}")


if __name__ == "__main__":
    main()
