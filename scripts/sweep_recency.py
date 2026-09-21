#!/usr/bin/env python3
"""Sweep recency settings without re-running retrieval for each one.

Running diag_retrieval.py once per setting reloads BGE-M3 and the cross-encoder
every time, and re-scores 200 pairs per question - for parameters that cannot
change a single reranker score. Recency is applied downstream, in select_pages.

So: retrieve and rerank once per question, cache the scored nodes, then replay
select_pages against each setting. One slow pass, then instant comparisons.

    python scripts/sweep_recency.py --index-dir storage/idx_sem8k
    python scripts/sweep_recency.py --index-dir storage/idx_sem8k --show-flips
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.config import load_config                       # noqa: E402
from unitn_rag.evaluation import hit_at_k, is_retrieval_scored, targets_of  # noqa: E402
from unitn_rag.indexing import load_index                      # noqa: E402
from unitn_rag.retrieval import (                              # noqa: E402
    Retriever,
    detect_query_language,
    select_pages,
)

K_VALUES = (1, 3, 5, 10)

# (recency_weight, recency_unknown, url_affinity_weight).
# Both multipliers are applied after the reranker has scored, and the
# reranker's sigmoid saturates near 0 and 1 - so a multiplier that mattered
# against raw cosine spread may now be inert. That is what this measures.
SETTINGS = [
    (0.0, 0.5, 0.25),   # current pick: recency off, affinity as configured
    (0.0, 0.5, 0.00),   # is affinity doing anything at all?
    (0.0, 0.5, 0.50),   # would more of it help #21's cross-course leak?
    (1.0, 0.5, 0.25),   # recency on, for reference
    (1.0, 0.5, 0.00),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--eval-set", default=None)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--max-pages", type=int, default=10)
    ap.add_argument("--show-flips", action="store_true",
                    help="Print which item ids change between settings")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.index_dir:
        cfg.paths.index_dir = Path(args.index_dir)
    if args.eval_set:
        cfg.paths.eval_set = Path(args.eval_set)
    cfg.retrieval.similarity_top_k = args.top_k
    cfg.retrieval.rerank_top_k = args.top_k

    print(f"[sweep] index  : {cfg.paths.index_dir}")
    print(f"[sweep] eval   : {cfg.paths.eval_set}")
    print(f"[sweep] rerank : {cfg.retrieval.rerank}  pool={args.top_k}\n")

    eval_set = json.loads(Path(cfg.paths.eval_set).read_text(encoding="utf-8"))
    index = load_index(cfg)
    retriever = Retriever(index, cfg.retrieval)

    # ---- one expensive pass -------------------------------------------
    cache = []
    items = [it for it in eval_set if is_retrieval_scored(it)]
    for n, item in enumerate(items, 1):
        q = item["question"]
        print(f"[sweep] retrieving {n}/{len(items)}", end="\r", flush=True)
        raw = retriever._retriever.retrieve(q)
        if cfg.retrieval.rerank:
            from unitn_rag.rerank import rerank as _rr
            raw = _rr(q, raw,
                      model_name=cfg.retrieval.rerank_model,
                      backend=cfg.retrieval.rerank_backend,
                      strip_header=cfg.retrieval.rerank_strip_header,
                      batch_size=cfg.retrieval.rerank_batch_size)
        cache.append((item, raw, detect_query_language(q)))
    print(f"[sweep] retrieved {len(cache)} questions" + " " * 20)

    # ---- cheap replays -------------------------------------------------
    rows = []
    per_setting_hits: dict[tuple, set] = {}
    for weight, unknown, affinity in SETTINGS:
        cfg.retrieval.recency_weight = weight
        cfg.retrieval.recency_unknown = unknown
        cfg.retrieval.url_affinity_weight = affinity
        hits = {k: 0 for k in K_VALUES}
        found = set()
        for item, raw, lang in cache:
            pages = select_pages(raw, cfg.retrieval, query_lang=lang,
                                 max_pages=args.max_pages, question=item["question"])
            sources = [p.url for p in pages]
            tgt = targets_of(item)
            for k in K_VALUES:
                if hit_at_k(sources, tgt, k):
                    hits[k] += 1
            if hit_at_k(sources, tgt, args.max_pages):
                found.add(item.get("id"))
        rows.append(((weight, unknown, affinity), hits))
        per_setting_hits[(weight, unknown, affinity)] = found

    n = len(cache)
    print(f"\n{'recency':>7} {'unknown':>8} {'affin':>6} "
          + " ".join(f"{'hit@'+str(k):>11}" for k in K_VALUES))
    print("-" * 64)
    for (w, u, a), hits in rows:
        cells = " ".join(f"{hits[k]/n:>6.3f} ({hits[k]:>2})" for k in K_VALUES)
        print(f"{w:>7} {u:>8} {a:>6} {cells}")
    print(f"\nn = {n}")

    if args.show_flips:
        base = per_setting_hits[SETTINGS[0]]
        print(f"\nrelative to {SETTINGS[0]} at hit@{args.max_pages}:")
        for st in SETTINGS[1:]:
            cur = per_setting_hits[st]
            gained, lost = sorted(cur - base), sorted(base - cur)
            print(f"  recency={st[0]} unknown={st[1]} affinity={st[2]}:  "
                  f"+{gained or '-'}  -{lost or '-'}")


if __name__ == "__main__":
    main()
