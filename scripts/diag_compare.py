#!/usr/bin/env python3
"""Compare retrieval hit@k between two indices on the same eval set, in one process.

Runs Retriever.retrieve() against index A and index B for every scored eval
item and reports:
  - a side-by-side hit@k table with the delta (B - A)
  - which specific item ids flip from hit->miss or miss->hit between the two
  - which ids miss in both (neither chunking strategy finds them - reranking
    or hybrid BM25/late-interaction territory, not a chunking fix)

This answers the ablation question directly instead of eyeballing two
separate diag_retrieval.py runs: it tells you *which* documents semantic
chunking actually rescues, not just whether the aggregate rate moved.

    python scripts/diag_compare.py --a storage/idx_fixed2 --b storage/idx_sem8k
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


def run_one(cfg, index_dir: str, eval_set: list, top_k: int, max_pages: int) -> dict:
    cfg.paths.index_dir = Path(index_dir)
    cfg.retrieval.similarity_top_k = top_k
    index = load_index(cfg)
    retriever = Retriever(index, cfg.retrieval)

    hits = {k: 0 for k in K_VALUES}
    n_scored = 0
    per_item_hit_max: dict = {}
    pool_sizes: list[int] = []

    for item in eval_set:
        if not is_retrieval_scored(item):
            continue
        n_scored += 1
        question = item["question"]
        lang = detect_query_language(question)
        pages = retriever.retrieve(question, query_lang=lang, max_pages=max_pages)
        sources = [p.url for p in pages]
        pool_sizes.append(len(sources))

        targets = targets_of(item)
        for k in K_VALUES:
            if hit_at_k(sources, targets, k):
                hits[k] += 1
        per_item_hit_max[item.get("id")] = hit_at_k(sources, targets, max(K_VALUES))

    avg_pool = sum(pool_sizes) / len(pool_sizes) if pool_sizes else 0.0
    return {
        "hits": hits,
        "n_scored": n_scored,
        "per_item_hit_max": per_item_hit_max,
        "avg_pool": avg_pool,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--a", required=True, help="First index dir, e.g. storage/idx_fixed2")
    ap.add_argument("--b", required=True, help="Second index dir, e.g. storage/idx_sem8k")
    ap.add_argument("--eval-set", default=None)
    ap.add_argument("--top-k", type=int, default=60,
                     help="Raw chunks pulled from FAISS before dedup (see diag_retrieval.py)")
    ap.add_argument("--max-pages", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.eval_set:
        cfg.paths.eval_set = Path(args.eval_set)
    eval_set = json.loads(Path(cfg.paths.eval_set).read_text(encoding="utf-8"))
    items_by_id = {item.get("id"): item for item in eval_set}

    print(f"[compare] A = {args.a}")
    print(f"[compare] B = {args.b}")
    print(f"[compare] eval set = {cfg.paths.eval_set}\n")

    res_a = run_one(cfg, args.a, eval_set, args.top_k, args.max_pages)
    res_b = run_one(cfg, args.b, eval_set, args.top_k, args.max_pages)

    n = res_a["n_scored"]
    print(f"{'k':>4}  {'A hit rate':>10}  {'B hit rate':>10}  {'delta':>8}")
    for k in K_VALUES:
        ra = res_a["hits"][k] / n if n else 0.0
        rb = res_b["hits"][k] / n if n else 0.0
        print(f"{k:>4}  {ra:>10.3f}  {rb:>10.3f}  {rb - ra:>+8.3f}")

    print(f"\n[compare] avg pool  A={res_a['avg_pool']:.1f}  B={res_b['avg_pool']:.1f}"
          f"  (n_scored={n})")

    ids = list(res_a["per_item_hit_max"].keys())
    a_only = [i for i in ids
              if res_a["per_item_hit_max"][i] and not res_b["per_item_hit_max"].get(i)]
    b_only = [i for i in ids
              if res_b["per_item_hit_max"][i] and not res_a["per_item_hit_max"].get(i)]
    both_miss = [i for i in ids
                 if not res_a["per_item_hit_max"][i] and not res_b["per_item_hit_max"].get(i)]
    both_hit = [i for i in ids
                if res_a["per_item_hit_max"][i] and res_b["per_item_hit_max"].get(i)]

    def describe(i):
        item = items_by_id.get(i, {})
        return f"  id {i}: {item.get('question', '')[:75]}"

    print(f"\n[compare] hit in A only, miss in B ({len(a_only)}):"
          "  <- B's chunking regressed these")
    for i in a_only:
        print(describe(i))

    print(f"\n[compare] hit in B only, miss in A ({len(b_only)}):"
          "  <- B's chunking rescued these")
    for i in b_only:
        print(describe(i))

    print(f"\n[compare] miss in both ({len(both_miss)}):"
          "  <- neither chunking finds these; not a chunking problem")
    for i in both_miss:
        print(describe(i))

    print(f"\n[compare] summary: {len(both_hit)} hit-both, {len(a_only)} A-only, "
          f"{len(b_only)} B-only, {len(both_miss)} miss-both  (n={n})")


if __name__ == "__main__":
    main()
