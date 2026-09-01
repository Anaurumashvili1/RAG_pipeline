#!/usr/bin/env python3
"""Where does a specific document rank for a specific question?

hit@k answers "did it make the cut". It does not say whether the target missed
by two places or by four thousand, and those imply completely different fixes:
a near miss is a ranking problem a reranker can close, while a deep miss means
the embedding is not matching at all and no amount of reordering will help.

    # one question, one target
    python scripts/probe_rank.py --question "Can I transfer into the third year of the CEILS bachelor at Trento?" \
        --url-contains guidelines_trasfer

    # every eval item that misses, with its target's true rank
    python scripts/probe_rank.py --eval-set-misses --top-k 2000

    # compare phrasings - the point of the exercise for cross-language cases
    python scripts/probe_rank.py --url-contains GUIDA%20GIURISPRUDENZA%202025-26 \
        --question "How many complementary exams from the third year onwards for Law enrolled 2019?" \
        --question "Quanti esami complementari devo sostenere dal terzo anno in poi?"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.config import load_config                      # noqa: E402
from unitn_rag.evaluation import is_retrieval_scored, targets_of  # noqa: E402
from unitn_rag.indexing import load_index                     # noqa: E402
from unitn_rag.retrieval import detect_query_language         # noqa: E402


def rank_of(retriever, question: str, needle: str):
    """Rank of the first chunk whose URL contains `needle`, over raw FAISS hits.

    Deliberately works on raw chunks, before dedup and before any of the
    re-scoring in select_pages: this measures what the embedding actually did,
    not what the post-processing made of it.
    """
    nodes = retriever.retrieve(question)
    for i, n in enumerate(nodes, 1):
        url = (n.node.metadata or {}).get("url", "")
        if needle.lower() in url.lower():
            return i, len(nodes), url, float(n.score or 0.0)
    return None, len(nodes), None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--eval-set", default=None)
    ap.add_argument("--question", action="append", default=[],
                    help="Repeatable - compare phrasings of the same question")
    ap.add_argument("--url-contains", default=None,
                    help="Substring identifying the target document")
    ap.add_argument("--top-k", type=int, default=1000,
                    help="How deep to search. Large on purpose: the question is "
                         "how far down the target is, not whether it made top-20")
    ap.add_argument("--eval-set-misses", action="store_true",
                    help="Probe every eval item whose target misses the normal pool")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.index_dir:
        cfg.paths.index_dir = Path(args.index_dir)
    if args.eval_set:
        cfg.paths.eval_set = Path(args.eval_set)
    cfg.retrieval.similarity_top_k = args.top_k

    print(f"[probe] index : {cfg.paths.index_dir}")
    print(f"[probe] depth : {args.top_k} raw chunks\n")

    index = load_index(cfg)
    retriever = index.as_retriever(similarity_top_k=args.top_k)

    if args.eval_set_misses:
        eval_set = json.loads(Path(cfg.paths.eval_set).read_text(encoding="utf-8"))
        print(f"{'id':>4} {'rank':>7} {'score':>7}  question")
        print("-" * 110)
        buckets = {"top20": 0, "21-100": 0, "101-1000": 0, "beyond": 0}
        for item in eval_set:
            if not is_retrieval_scored(item):
                continue
            targets = targets_of(item)
            if not targets:
                continue
            needle = targets[0].rsplit("/", 1)[-1] or targets[0]
            rank, n, _, score = rank_of(retriever, item["question"], needle)
            if rank is None:
                label, bucket = f">{n}", "beyond"
            else:
                label = str(rank)
                bucket = ("top20" if rank <= 20 else
                          "21-100" if rank <= 100 else
                          "101-1000" if rank <= 1000 else "beyond")
            buckets[bucket] += 1
            if rank is None or rank > 20:
                s = f"{score:.3f}" if score is not None else "-"
                print(f"{str(item.get('id')):>4} {label:>7} {s:>7}  {item['question'][:78]}")
        print("\n[probe] target depth distribution:", buckets)
        print("[probe] 21-100 is reranker territory; 'beyond' means the embedding")
        print("[probe] never matched and only better retrieval can help.")
        return

    if not args.question or not args.url_contains:
        ap.error("give --question (repeatable) and --url-contains, or --eval-set-misses")

    for q in args.question:
        lang = detect_query_language(q)
        rank, n, url, score = rank_of(retriever, q, args.url_contains)
        print(f"  [{lang}] {q[:96]}")
        if rank is None:
            print(f"        NOT FOUND in {n} chunks\n")
        else:
            print(f"        rank {rank}/{n}   score {score:.4f}")
            print(f"        {url[:112]}\n")


if __name__ == "__main__":
    main()
