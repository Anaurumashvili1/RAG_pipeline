#!/usr/bin/env python3
"""Re-score a saved results file against a (possibly newer) evaluation set.

``run_eval.py --score-only`` cannot do this. The results file embeds a snapshot
of ``target_url`` / ``acceptable_urls`` taken when the run happened, and the
summariser reads that snapshot - so correcting the evaluation set has no effect
on a re-score, and the numbers come back identical. Which looks exactly like
"the fix did nothing", and is not.

This joins saved answers to a fresh evaluation set by ``id`` and recomputes
hit@k from the stored ``rag_sources``. No LLM, no index, no re-run.

    python scripts/rescore.py results/eval_sem8k_v2.json data/evaluation_set.v4.json
    python scripts/rescore.py results/eval_sem8k_v2.json data/evaluation_set.v4.json \
        --against data/evaluation_set.v3.json      # show the delta
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

K_VALUES = (1, 3, 5, 10, 20)


def norm(url: str | None) -> str:
    return (url or "").rstrip("/").lower()


def targets(spec: dict) -> set[str]:
    urls = spec.get("acceptable_urls") or [spec.get("target_url", "")]
    return {norm(u) for u in urls if u}


def scored(spec: dict) -> bool:
    return not (spec.get("answer_only") or spec.get("out_of_scope"))


def tally(results: list[dict], by_id: dict[int, dict]):
    hits = {k: 0 for k in K_VALUES}
    n = 0
    per_item: dict[int, bool] = {}
    for i, item in enumerate(results):
        spec = by_id.get(item.get("id", i))
        if spec is None or not scored(spec):
            continue
        n += 1
        tgt = targets(spec)
        srcs = [norm(s) for s in item.get("rag_sources", [])]
        for k in K_VALUES:
            if any(s in tgt for s in srcs[:k]):
                hits[k] += 1
        per_item[item.get("id", i)] = any(s in tgt for s in srcs[:10])
    return hits, n, per_item


def load_set(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {it["id"]: it for it in data}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="a results/*.json produced by run_eval.py")
    ap.add_argument("eval_set", help="the evaluation set to score against")
    ap.add_argument("--against", default=None,
                    help="an older evaluation set, to print the delta")
    args = ap.parse_args()

    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    results = payload["results"] if isinstance(payload, dict) else payload

    new = load_set(Path(args.eval_set))
    h_new, n_new, per_new = tally(results, new)

    if args.against:
        old = load_set(Path(args.against))
        h_old, n_old, per_old = tally(results, old)
        print(f"{'k':<5} {Path(args.against).name:>22} {Path(args.eval_set).name:>22} {'delta':>8}")
        for k in K_VALUES:
            a = h_old[k] / n_old if n_old else 0.0
            b = h_new[k] / n_new if n_new else 0.0
            print(f"{k:<5} {a:>15.3f} ({h_old[k]:>2}) {b:>15.3f} ({h_new[k]:>2}) {b - a:>+8.3f}")
        flips = [i for i in per_new if i in per_old and per_new[i] != per_old[i]]
        print(f"\nitems changing at hit@10: {sorted(flips)}")
    else:
        print(f"{'k':>4}  {'hit rate':>10}  {'count':>8}")
        for k in K_VALUES:
            rate = h_new[k] / n_new if n_new else 0.0
            print(f"{k:>4}  {rate:>10.3f}  {h_new[k]:>4}/{n_new}")

    print(f"\nscored {n_new} items from {Path(args.results).name}")
    print("NOTE: hit@k only. Answer grades live in 'rag_correct' and are unaffected.")


if __name__ == "__main__":
    main()
