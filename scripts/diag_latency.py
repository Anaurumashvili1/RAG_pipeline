#!/usr/bin/env python3
"""What does a single chatbot turn actually cost, once models are warm?

Every diagnostic run today paid model-loading cost fresh because each was a
new process - a deployed server pays that once, not per user message. This
times single pipeline.answer() calls after warmup, and separately isolates
retrieval+rerank from the rest (guardrail + generation), so it's clear where
real per-turn latency actually goes.

Usage: python scripts/diag_latency.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.config import load_config    # noqa: E402
from unitn_rag.pipeline import RagPipeline  # noqa: E402

QUESTIONS = [
    ("en", "Who is the programme coordinator for the master's in Civil Engineering at Trento?"),
    ("it", "Quando iniziano le lezioni del secondo semestre per la facoltà di Giurisprudenza?"),
    ("en", "What are the opening hours of the BUD helpdesk for questions about electronic resources?"),
]


def main() -> None:
    cfg = load_config("config.yaml")
    print("[latency] constructing pipeline (loads embedding + reranker models - "
          "one-time cost, NOT paid per user turn in a deployed server)...")
    t0 = time.perf_counter()
    pipeline = RagPipeline(cfg, guardrail=True)  # guardrail on: matches real deployment
    load_s = time.perf_counter() - t0
    print(f"[latency] pipeline construction (model loading): {load_s:.1f}s (one-time)\n")

    print("[latency] warmup call (first real call triggers lazy model init inside "
          "the reranker singleton too - discard this one)...")
    _ = pipeline.answer(QUESTIONS[0][1])
    print("[latency] warmup done\n")

    for lang, q in QUESTIONS:
        t0 = time.perf_counter()
        pages = pipeline.retriever.retrieve(q, max_pages=cfg.retrieval.max_pages)
        retrieve_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        result = pipeline.answer(q)
        total_s = time.perf_counter() - t0

        other_s = max(total_s - retrieve_s, 0.0)
        print(f"=== [{lang}] {q}")
        print(f"    retrieval+translation+rerank : {retrieve_s:5.2f}s")
        print(f"    guardrail + generation (est.): {other_s:5.2f}s")
        print(f"    total (single real turn)     : {total_s:5.2f}s")
        print(f"    answer: {result.answer[:120]}{'...' if len(result.answer) > 120 else ''}\n")


if __name__ == "__main__":
    main()
