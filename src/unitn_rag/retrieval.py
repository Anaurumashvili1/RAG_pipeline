"""Retrieval: fetch chunks, then collapse them to distinct source documents.

Two changes from the notebook version:

1. Dedup key is ``doc_group_id`` rather than ``url``, so the Italian and English
   versions of one page no longer occupy two of the five context slots.
2. When translations tie, the version in the user's language wins - unless the
   other-language sibling is materially fresher, in which case the fresher one
   is used and the LLM answers across languages.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .config import RetrievalCfg
from .text import detect_language_from_text, recency_penalty


@dataclass
class RetrievedPage:
    rank: int
    url: str
    title: str
    lang: str
    effective_year: int | None
    score: float
    text: str

    @property
    def citation(self) -> str:
        return f"[{self.rank}] {self.title or self.url} - {self.url}"


def detect_query_language(question: str) -> str:
    """Language of the incoming question. Short queries default to English."""
    return detect_language_from_text(question, sample_chars=400) or "en"


def _group_key(node_meta: dict, dedup_by: str) -> str:
    if dedup_by == "url":
        return node_meta.get("url", "")
    return node_meta.get("doc_group_id") or node_meta.get("url", "")


def select_pages(
    raw_results,
    cfg: RetrievalCfg,
    query_lang: str = "en",
    current_year: int | None = None,
    apply_recency: bool = True,
    max_pages: int | None = None,
) -> list[RetrievedPage]:
    """Collapse ranked chunks into distinct documents.

    ``raw_results`` is the list of NodeWithScore returned by a LlamaIndex retriever.
    """
    current_year = current_year or date.today().year
    limit = max_pages or cfg.max_pages

    best: dict[str, dict] = {}

    for r in raw_results:
        meta = r.node.metadata or {}
        key = _group_key(meta, cfg.dedup_by)
        if not key:
            continue

        score = float(r.score) if r.score is not None else 0.0
        if apply_recency:
            score *= recency_penalty(meta.get("effective_year"), current_year)
        if cfg.prefer_query_language and meta.get("lang") == query_lang:
            score *= 1.10          # mild tie-break, not an override

        candidate = {
            "score": score,
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "lang": meta.get("lang", ""),
            "effective_year": meta.get("effective_year"),
            "text": r.node.get_content(metadata_mode="none").strip(),
        }

        if key not in best or candidate["score"] > best[key]["score"]:
            best[key] = candidate

    ranked = sorted(best.values(), key=lambda c: c["score"], reverse=True)[:limit]

    return [
        RetrievedPage(
            rank=i,
            url=c["url"],
            title=c["title"],
            lang=c["lang"],
            effective_year=c["effective_year"],
            score=c["score"],
            text=c["text"][: cfg.chunk_char_limit],
        )
        for i, c in enumerate(ranked, 1)
    ]


class Retriever:
    """Thin wrapper binding a LlamaIndex retriever to the project's dedup logic."""

    def __init__(self, index, cfg: RetrievalCfg):
        self.cfg = cfg
        self._retriever = index.as_retriever(similarity_top_k=cfg.similarity_top_k)

    def retrieve(
        self,
        question: str,
        query_lang: str | None = None,
        max_pages: int | None = None,
    ) -> list[RetrievedPage]:
        lang = query_lang or detect_query_language(question)
        raw = self._retriever.retrieve(question)
        return select_pages(raw, self.cfg, query_lang=lang, max_pages=max_pages)


def format_context(pages: list[RetrievedPage]) -> str:
    """Render pages as the numbered context block the prompt expects."""
    blocks = []
    for p in pages:
        head = f"[{p.rank}] SOURCE: {p.url}"
        if p.title:
            head += f"\nTITLE: {p.title}"
        if p.effective_year:
            head += f"\nYEAR: {p.effective_year}"
        blocks.append(f"{head}\n{p.text}")
    return "\n\n".join(blocks)
