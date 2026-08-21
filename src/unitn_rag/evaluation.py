"""Evaluation (Colab cells 9-11), with the hit@k measurement corrected.

Bug in v1: ``retrieve_context`` truncated results to ``max_pages=5`` before the
metrics were computed, so hit@1/@3/@5 were all measured over the same 5-item
list. hit@5 was really "hit@min(5, len(sources))" and could never exceed it.
Here retrieval runs with a wider ``eval_max_pages`` and k is applied afterwards.

Correctness of the generated answer is still judged manually - keep doing that,
it is the honest approach for this dataset. ``export_for_review`` writes a file
you can grade, and ``score_manual_grades`` turns the grades into the paper's
accuracy / accuracy-on-attempted / refusal-rate table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

# Imported lazily: the metric functions below are pure Python and must stay
# usable (and testable) without faiss / torch / llama-index installed.
if TYPE_CHECKING:
    from .pipeline import RagAnswer, RagPipeline


def _url_match(a: str, b: str) -> bool:
    """Compare URLs ignoring trailing slash and scheme differences."""
    def norm(u: str) -> str:
        return (u or "").strip().rstrip("/").replace("https://", "").replace("http://", "")
    return norm(a) == norm(b)


def hit_at_k(sources: list[str], target_url: str, k: int) -> bool:
    return any(_url_match(s, target_url) for s in sources[:k])


def run_evaluation(
    pipeline: "RagPipeline",
    eval_set: list[dict],
    eval_max_pages: int = 10,
    include_baseline: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Run RAG (and optionally the baseline) over the evaluation set."""
    results = []

    for i, item in enumerate(eval_set, 1):
        question = item["question"]
        target_url = item.get("target_url", "")

        if verbose:
            print(f"[eval] {i}/{len(eval_set)}  {question[:70]}")

        rag: "RagAnswer" = pipeline.answer(question, max_pages=eval_max_pages)

        row = {
            "question": question,
            "target_url": target_url,
            "gold_answer": item.get("answer") or item.get("gold_answer", ""),
            "rag_answer": rag.answer,
            "rag_sources": rag.sources,
            "rag_language": rag.language,
            "rag_refused": rag.refused,
            "rag_cited": rag.cited,
            "hit@1": hit_at_k(rag.sources, target_url, 1),
            "hit@3": hit_at_k(rag.sources, target_url, 3),
            "hit@5": hit_at_k(rag.sources, target_url, 5),
            "hit@10": hit_at_k(rag.sources, target_url, 10),
            # filled in during manual review:
            "rag_correct": None,
            "baseline_correct": None,
        }

        if include_baseline:
            base = pipeline.answer_baseline(question)
            row["baseline_answer"] = base.answer
            row["baseline_refused"] = base.refused

        results.append(row)

    return results


def retrieval_metrics(results: list[dict]) -> dict:
    n = len(results) or 1
    return {
        "n": len(results),
        "hit@1": round(sum(r["hit@1"] for r in results) / n, 4),
        "hit@3": round(sum(r["hit@3"] for r in results) / n, 4),
        "hit@5": round(sum(r["hit@5"] for r in results) / n, 4),
        "hit@10": round(sum(r["hit@10"] for r in results) / n, 4),
    }


def generation_metrics(results: list[dict], prefix: str = "rag") -> dict:
    """Requires ``{prefix}_correct`` to be filled in (True/False) by manual review."""
    graded = [r for r in results if r.get(f"{prefix}_correct") is not None]
    if not graded:
        return {"graded": 0, "note": f"no manual grades in '{prefix}_correct' yet"}

    n = len(graded)
    correct = sum(1 for r in graded if r[f"{prefix}_correct"])
    refusals = sum(1 for r in graded if r.get(f"{prefix}_refused"))
    attempted = n - refusals
    wrong_attempts = sum(
        1 for r in graded if not r[f"{prefix}_correct"] and not r.get(f"{prefix}_refused")
    )

    return {
        "graded": n,
        "accuracy": round(correct / n, 4),
        "accuracy_on_attempted": round(correct / attempted, 4) if attempted else None,
        "refusal_rate": round(refusals / n, 4),
        "factual_error_rate": round(wrong_attempts / attempted, 4) if attempted else None,
    }


def retrieved_but_not_answered(results: list[dict]) -> list[dict]:
    """The paper's core failure class: right page retrieved, answer refused.

    Track this number - it is the metric the whole v2 redesign is meant to move.
    """
    return [r for r in results if r["hit@5"] and r.get("rag_refused")]


def summarise(results: list[dict]) -> dict:
    gap = retrieved_but_not_answered(results)
    return {
        "retrieval": retrieval_metrics(results),
        "generation_rag": generation_metrics(results, "rag"),
        "generation_baseline": generation_metrics(results, "baseline"),
        "retrieved_but_refused": {
            "count": len(gap),
            "rate": round(len(gap) / (len(results) or 1), 4),
            "questions": [r["question"] for r in gap],
        },
    }


def run_metadata(cfg) -> dict:
    """What produced these numbers.

    Without this, an ablation across chunking strategies and models leaves you
    with several result files and no way to tell which is which. Recorded at
    save time rather than reconstructed later from memory.
    """
    from datetime import datetime, timezone

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm_model": cfg.llm.model,
        "llm_temperature": cfg.llm.temperature,
        "llm_max_tokens": cfg.llm.max_tokens,
        "embedding_model": cfg.embedding.model_name,
        "embedding_device": cfg.embedding.device,
        "index_dir": str(cfg.paths.index_dir),
        "chunk_size": cfg.chunking.chunk_size,
        "chunk_overlap": cfg.chunking.chunk_overlap,
        "inject_header": cfg.chunking.inject_header,
        "semantic_min_chars": cfg.chunking.semantic_min_chars,
        "semantic_percentile": cfg.chunking.semantic_percentile,
        "similarity_top_k": cfg.retrieval.similarity_top_k,
        "max_pages": cfg.retrieval.max_pages,
        "dedup_by": cfg.retrieval.dedup_by,
        "corpus": str(cfg.paths.corpus),
    }


def save_results(results: list[dict], path: str | Path, cfg=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict | list
    if cfg is not None:
        payload = {
            "run": run_metadata(cfg),
            "summary": summarise(results),
            "results": results,
        }
    else:
        payload = results

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] saved {len(results)} rows to {path}")


def load_results(path: str | Path) -> list[dict]:
    """Read a results file. Accepts both the bare-list and wrapped formats."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("results", [])
    return data


def export_for_review(results: list[dict], path: str | Path) -> None:
    """Write a CSV for manual grading: fill rag_correct / baseline_correct with 1 or 0."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "question", "gold_answer", "rag_answer", "rag_refused",
        "hit@5", "baseline_answer", "rag_correct", "baseline_correct",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"[eval] review sheet written to {path}")
