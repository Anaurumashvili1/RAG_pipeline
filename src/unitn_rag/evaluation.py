"""Evaluation (Colab cells 9-11), with the hit@k measurement corrected.

Bug in v1: ``retrieve_context`` truncated results to ``max_pages=5`` before the
metrics were computed, so hit@1/@3/@5 were all measured over the same 5-item
list. hit@5 was really "hit@min(5, len(sources))" and could never exceed it.
Here retrieval runs with a wider ``eval_max_pages`` and k is applied afterwards.

Second correction: a target is a *set* of URLs, not one URL. ``select_pages``
collapses an IT/EN pair to a single slot, so whichever sibling ranks higher is
the one reported - and a set holding only the other sibling scored a correct
retrieval as a miss. ``acceptable_urls`` names every address that counts, and
items no retriever can fairly be asked to hit (a fact repeated verbatim across
dozens of boilerplate blocks, or an out-of-scope question) are excluded from
the retrieval denominator instead of silently depressing it.

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


def _norm_url(u: str) -> str:
    """Canonical form for comparison: no scheme, no trailing slash."""
    u = (u or "").strip()
    for prefix in ("https://", "http://"):
        if u.startswith(prefix):
            u = u[len(prefix):]
            break
    return u.rstrip("/")


def _url_match(a: str, b: str) -> bool:
    """Compare URLs ignoring trailing slash and scheme differences."""
    return _norm_url(a) == _norm_url(b)


def _as_url_list(raw) -> list[str]:
    """Accept a list, a pipe-separated string (the worksheet format), or None."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.split("|")
    return [u.strip() for u in raw if u and str(u).strip()]


def targets_of(item: dict) -> list[str]:
    """Every URL that counts as a correct retrieval for this item.

    A fact usually lives at more than one address. An IT/EN translation pair is
    one document to a reader and two URLs to the index; ``select_pages``
    collapses the pair to one slot and reports whichever sibling ranked higher.
    Comparing against a single ``target_url`` therefore records a correct
    retrieval as a miss whenever the other sibling surfaces. The same holds for
    a fact stated on two genuinely different pages.

    ``target_url`` stays authoritative for citation display; it is simply the
    first member of the acceptable set.
    """
    urls = _as_url_list(item.get("acceptable_urls"))
    single = (item.get("target_url") or "").strip()
    if single and not any(_url_match(single, u) for u in urls):
        urls.insert(0, single)
    return urls


def is_retrieval_scored(item: dict) -> bool:
    """False for items ``hit@k`` cannot judge fairly.

    Two kinds, and both must leave the retrieval denominator rather than sit in
    it as guaranteed zeros:

    *answer-only* - the fact appears verbatim in many documents (a boilerplate
    contact block) and the question names nothing that singles out the target,
    so no retriever can be expected to pick it. The generated answer is still
    worth grading.

    *out-of-scope* - there is no acceptable URL at all; the item is scored on
    whether the system refuses.
    """
    if item.get("answer_only") or item.get("out_of_scope"):
        return False
    return bool(targets_of(item))


def hit_at_k(sources: list[str], targets, k: int) -> bool:
    """True if any acceptable target appears in the first ``k`` sources.

    ``targets`` may be one URL, a list of URLs, or a pipe-separated string.
    """
    targets = _as_url_list(targets)
    if not targets:
        return False
    return any(_url_match(s, t) for s in sources[:k] for t in targets)


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
        targets = targets_of(item)
        scored = is_retrieval_scored(item)

        if verbose:
            print(f"[eval] {i}/{len(eval_set)}  {question[:70]}")

        # One flaky call must not destroy a 25-minute run. Two runs were lost
        # this way - a read timeout and an upstream 500 - each after most of
        # the set had already been generated.
        rag = None
        error = None
        try:
            rag = pipeline.answer(question, max_pages=eval_max_pages)
        except Exception as exc:                      # noqa: BLE001 - see above
            error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"[eval] !! item {i} failed, continuing: {error[:150]}")

        # hit@k stays None on failure, so a call that never happened is excluded
        # from the denominator rather than counted as a retrieval miss.
        measurable = scored and rag is not None

        row = {
            "id": item.get("id", i - 1),
            "question": question,
            "target_url": target_url,
            "acceptable_urls": targets,
            "retrieval_scored": scored,
            "objective": item.get("objective", ""),
            "gold_answer": item.get("answer") or item.get("gold_answer", ""),
            "error": error,
            "rag_answer": rag.answer if rag else "",
            "rag_sources": rag.sources if rag else [],
            "rag_language": rag.language if rag else None,
            "rag_refused": rag.refused if rag else None,
            "rag_cited": rag.cited if rag else [],
            # None, not False, when the item is not a retrieval test - a zero
            # here would be indistinguishable from a genuine miss.
            "hit@1": hit_at_k(rag.sources, targets, 1) if measurable else None,
            "hit@3": hit_at_k(rag.sources, targets, 3) if measurable else None,
            "hit@5": hit_at_k(rag.sources, targets, 5) if measurable else None,
            "hit@10": hit_at_k(rag.sources, targets, 10) if measurable else None,
            # filled in during manual review:
            "rag_correct": None,
            "baseline_correct": None,
        }

        if include_baseline:
            try:
                base = pipeline.answer_baseline(question)
                row["baseline_answer"] = base.answer
                row["baseline_refused"] = base.refused
            except Exception as exc:                  # noqa: BLE001
                row["baseline_answer"] = ""
                row["baseline_refused"] = None
                row["baseline_error"] = f"{type(exc).__name__}: {exc}"

        results.append(row)

    return results


def retrieval_metrics(results: list[dict]) -> dict:
    """hit@k over the items that are retrieval tests.

    ``n`` is the scored count, not the size of the set: answer-only and
    out-of-scope items are excluded, and ``n_excluded`` records how many, so the
    denominator is always visible next to the number it produced.
    """
    scored = [r for r in results if r.get("hit@1") is not None]
    n = len(scored)
    # Errors are counted separately: they shrink the denominator, so without
    # this a run where half the calls failed would report a healthy-looking
    # rate over the half that survived.
    n_errors = sum(1 for r in results if r.get("error"))
    out: dict = {"n": n, "n_excluded": len(results) - n, "n_errors": n_errors}
    if not n:
        out["note"] = "no items were scored for retrieval"
        return out
    for k in (1, 3, 5, 10):
        out[f"hit@{k}"] = round(sum(1 for r in scored if r.get(f"hit@{k}")) / n, 4)
    return out


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
    return [r for r in results if r.get("hit@5") and r.get("rag_refused")]


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
    # ``objective`` rides along because a red cell on its own does not say
    # whether retrieval missed, generation flattened a conditional, or the
    # metric was unfair to the item - the objective is what makes that call.
    cols = [
        "question", "objective", "gold_answer", "rag_answer", "rag_refused",
        "hit@5", "retrieval_scored", "baseline_answer",
        "rag_correct", "baseline_correct",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"[eval] review sheet written to {path}")
