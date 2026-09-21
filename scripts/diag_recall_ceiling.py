#!/usr/bin/env python3
"""Check whether similarity_top_k=20 is actually the bottleneck.

Every failure diagnosed today (#3, #4, #5, #13, #16) turned out to be a
scoring/selection problem, not a recall problem - the target was always
present somewhere in the raw top-20 dense-retrieval candidates, it just lost
out during rerank/recency/family_cap. This script checks whether that holds
across the whole eval set: for every scored question, where does the
target/acceptable URL land in RAW dense retrieval (translation union applied,
but before rerank and before select_pages) at a generous top_k?

If nearly everything is inside the current top 20, raising similarity_top_k
won't move the needle - it'll just hand the reranker more distractors at
higher compute cost. If a meaningful chunk of misses sit at rank 21-50,
that's real evidence for raising it. If some are absent even at 50, that's a
different problem entirely (chunking/embedding/query phrasing), not a top_k
question.

Usage:
    python scripts/diag_recall_ceiling.py [--top-k 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.config import load_config          # noqa: E402
from unitn_rag.pipeline import RagPipeline        # noqa: E402
from unitn_rag.retrieval import detect_query_language  # noqa: E402
from unitn_rag.translate import union_nodes       # noqa: E402


def raw_candidates(retriever, question, lang):
    raw = retriever._retriever.retrieve(question)
    if retriever.translator is not None and getattr(retriever.cfg, "translate_query", False):
        other = retriever.translator.translate(question, lang)
        if other:
            raw = union_nodes(raw, retriever._retriever.retrieve(other))
    # union_nodes deliberately does not sort (the reranker rescores everything
    # downstream, so it never needed to) - but for THIS diagnostic, list
    # position is the whole signal, so it must be sorted by score here or
    # "rank" is meaningless for anything only the translated query found.
    return sorted(raw, key=lambda r: r.score or 0.0, reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=None,
                    help="Override retrieval depth (rerank_top_k if rerank is on, "
                         "else similarity_top_k). Omit to measure production's actual "
                         "configured depth as-is.")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    # The retriever's real depth is rerank_top_k when reranking is on (see
    # retrieval.py Retriever.__init__) - similarity_top_k only applies when
    # rerank is off. Override whichever one actually governs retrieval depth,
    # unless --top-k was left at its default, in which case just measure
    # production's real, already-configured depth with no override at all.
    if args.top_k is not None:
        if getattr(cfg.retrieval, "rerank", False):
            cfg.retrieval.rerank_top_k = args.top_k
        else:
            cfg.retrieval.similarity_top_k = args.top_k
    effective_depth = cfg.retrieval.rerank_top_k if getattr(cfg.retrieval, "rerank", False) else cfg.retrieval.similarity_top_k
    pipeline = RagPipeline(cfg, guardrail=False)
    retriever = pipeline.retriever

    eval_set = json.loads(Path(cfg.paths.eval_set).read_text(encoding="utf-8"))

    buckets = {"top5": 0, "top10": 0, "top20": 0, "top30": 0, "top50": 0, "absent": 0}
    misses: list[tuple[int, str]] = []
    deep_hits: list[tuple[int, str, int]] = []

    scored = 0
    for item in eval_set:
        if item.get("retrieval_scored") is False:
            continue
        scored += 1
        question = item["question"]
        acceptable = set(item.get("acceptable_urls") or [item.get("target_url")])
        lang = detect_query_language(question)
        raw = raw_candidates(retriever, question, lang)

        rank = None
        for i, r in enumerate(raw, 1):
            if (r.node.metadata or {}).get("url") in acceptable:
                rank = i
                break
        top5_urls = [(r.node.metadata or {}).get("url") for r in raw[:5]]

        if rank is None:
            buckets["absent"] += 1
            misses.append((item["id"], question, top5_urls))
        elif rank <= 5:
            buckets["top5"] += 1
        elif rank <= 10:
            buckets["top10"] += 1
        elif rank <= 20:
            buckets["top20"] += 1
        else:
            deep_hits.append((item["id"], question, rank, top5_urls))
            if rank <= 30:
                buckets["top30"] += 1
            else:
                buckets["top50"] += 1

    print(f"[diag] {scored} scored questions, raw dense-retrieval recall "
          f"(depth={effective_depth} per language, translation "
          f"{'on' if cfg.retrieval.translate_query else 'off'}):")
    for k, v in buckets.items():
        print(f"    {k:>6}: {v}")

    if deep_hits:
        print(f"\n[diag] target beyond rank 20 - spot-check whether these top-5 are genuinely "
              f"off-topic or plausible-but-unlisted alternatives:")
        for qid, q, rank, top5_urls in deep_hits:
            print(f"    #{qid} (target rank {rank}): {q}")
            for u in top5_urls:
                print(f"          {u}")

    if misses:
        print(f"\n[diag] ABSENT even at depth {effective_depth} (not a top_k problem - check "
              f"chunking/query phrasing/embedding instead):")
        for qid, q, top5_urls in misses:
            print(f"    #{qid}: {q}")
            for u in top5_urls:
                print(f"          {u}")


if __name__ == "__main__":
    main()
