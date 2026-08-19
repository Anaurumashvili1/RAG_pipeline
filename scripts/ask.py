#!/usr/bin/env python3
"""Ask a single question, or drop into an interactive loop.

    python scripts/ask.py "When does enrolment open for environmental engineering?"
    python scripts/ask.py                     # interactive
    python scripts/ask.py --no-guardrail "..."  # skip the intent classifier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.config import load_config    # noqa: E402
from unitn_rag.pipeline import RagPipeline  # noqa: E402


def show(result) -> None:
    print("\n" + "=" * 70)
    print(result.answer)
    if result.sources:
        print("\nSources:")
        for p in result.pages:
            year = f" ({p.effective_year})" if p.effective_year else ""
            marker = "*" if p.rank in result.cited else " "
            print(f" {marker}[{p.rank}] {p.title or '(untitled)'}{year}\n      {p.url}")
    print("=" * 70 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="Question (omit for interactive mode)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-guardrail", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pipeline = RagPipeline(cfg, guardrail=not args.no_guardrail)

    if args.question:
        show(pipeline.answer(" ".join(args.question)))
        return

    print("UniTn RAG - type a question, or 'quit' to exit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            show(pipeline.answer(q))


if __name__ == "__main__":
    main()
