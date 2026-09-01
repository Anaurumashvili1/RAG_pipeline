#!/usr/bin/env python3
"""Run the evaluation set and print metrics.

    python scripts/run_eval.py                 # RAG + baseline, saves results
    python scripts/run_eval.py --no-baseline   # RAG only (faster)
    python scripts/run_eval.py --score-only    # recompute metrics after manual grading
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.config import load_config        # noqa: E402
from unitn_rag.evaluation import (              # noqa: E402
    export_for_review,
    load_results,
    run_evaluation,
    save_results,
    summarise,
)
from unitn_rag.pipeline import RagPipeline      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--index-dir", default=None,
                    help="Override paths.index_dir - which ablation arm to score")
    ap.add_argument("--eval-set", default=None,
                    help="Override paths.eval_set, e.g. data/evaluation_set.resolved.json")
    ap.add_argument("--results", default=None,
                    help="Override paths.eval_results - keep one file per arm")
    ap.add_argument("--model", default=None, help="Override llm.model for this run")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--eval-max-pages", type=int, default=10,
                    help="Pages retrieved per question; hit@k is computed from this list")
    ap.add_argument("--score-only", action="store_true",
                    help="Skip generation, just recompute metrics from saved results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.index_dir:
        cfg.paths.index_dir = Path(args.index_dir)
    if args.eval_set:
        cfg.paths.eval_set = Path(args.eval_set)
    if args.results:
        cfg.paths.eval_results = Path(args.results)
    if args.model:
        cfg.llm.model = args.model

    # Printed because the commonest way to waste an evaluation run is to score
    # one index while believing you scored another. run_metadata records these
    # in the results file too.
    print(f"[eval] index    : {cfg.paths.index_dir}")
    print(f"[eval] eval set : {cfg.paths.eval_set}")
    print(f"[eval] model    : {cfg.llm.model}")
    print(f"[eval] results  : {cfg.paths.eval_results}")

    if args.score_only:
        results = load_results(cfg.paths.eval_results)
    else:
        eval_set = json.loads(Path(cfg.paths.eval_set).read_text(encoding="utf-8"))
        print(f"[eval] {len(eval_set)} questions")

        # The guardrail is off during evaluation: every eval question is in-domain
        # by construction, and the extra classifier call would double API cost.
        pipeline = RagPipeline(cfg, guardrail=False)

        results = run_evaluation(
            pipeline,
            eval_set,
            eval_max_pages=args.eval_max_pages,
            include_baseline=not args.no_baseline,
        )
        save_results(results, cfg.paths.eval_results, cfg=cfg)
        export_for_review(results, Path(cfg.paths.eval_results).with_suffix(".review.csv"))

    print("\n" + json.dumps(summarise(results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
