#!/usr/bin/env python3
"""Build the FAISS index from the crawl output.

    python scripts/build_index.py
    python scripts/build_index.py --config config.uni.yaml
    python scripts/build_index.py --max-docs 2000     # fast local smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.chunking import chunk_documents          # noqa: E402
from unitn_rag.config import load_config                # noqa: E402
from unitn_rag.data import corpus_stats, load_documents  # noqa: E402
from unitn_rag.indexing import build_index              # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="Override data.max_docs for a quick test run")
    ap.add_argument("--no-dedup", action="store_true",
                    help="Skip boilerplate chunk deduplication")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_docs:
        cfg.data.max_docs = args.max_docs

    print(f"[build] corpus: {cfg.paths.corpus}")
    docs = load_documents(
        cfg.paths.corpus,
        min_chars=cfg.data.min_chars,
        max_docs=cfg.data.max_docs,
        drop_duplicates=cfg.data.drop_duplicates,
        drop_low_content=cfg.data.drop_low_content,
        drop_boilerplate=cfg.data.drop_boilerplate,
        keep_languages=cfg.data.keep_languages,
    )
    if not docs:
        raise SystemExit("No documents loaded - check paths.corpus in config.yaml")

    stats = corpus_stats(docs)
    print("[build] corpus stats:")
    for k, v in stats.items():
        print(f"        {k}: {v}")

    nodes = chunk_documents(
        docs,
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
        inject_header=cfg.chunking.inject_header,
        deduplicate=not args.no_dedup,
    )
    print(f"[build] {len(nodes)} chunks after dedup")

    build_index(nodes, cfg)
    print("[build] done")


if __name__ == "__main__":
    main()
