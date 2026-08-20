"""End-to-end RAG pipeline (Colab cell 8, cleaned up).

The notebook built the context twice - once in ``retrieve_context`` and again
inside ``answer_with_rag``, with different truncation - so the sources reported
did not always match the text the model actually saw. Context is built once here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Config
from .indexing import load_index
from .llm import ChatClient
from .prompts import (
    baseline_messages,
    intent_messages,
    is_refusal,
    out_of_scope_reply,
    rag_messages,
)
from .retrieval import RetrievedPage, Retriever, detect_query_language, format_context

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class RagAnswer:
    question: str
    answer: str
    language: str
    sources: list[str] = field(default_factory=list)
    pages: list[RetrievedPage] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    refused: bool = False
    blocked: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "language": self.language,
            "sources": self.sources,
            "cited": self.cited,
            "refused": self.refused,
            "blocked": self.blocked,
            "retrieved": [
                {
                    "rank": p.rank,
                    "url": p.url,
                    "title": p.title,
                    "lang": p.lang,
                    "effective_year": p.effective_year,
                    "score": round(p.score, 4),
                }
                for p in self.pages
            ],
        }


def extract_citations(answer: str) -> list[int]:
    return sorted({int(x) for x in _CITATION_RE.findall(answer or "")})


class RagPipeline:
    """Load once, query many times."""

    def __init__(self, cfg: Config, index=None, guardrail: bool = True):
        self.cfg = cfg
        self.guardrail = guardrail
        self._index = index if index is not None else load_index(cfg)
        self.retriever = Retriever(self._index, cfg.retrieval)
        self.client = ChatClient(cfg.llm)

    # -- guardrail -----------------------------------------------------------

    def check_intent(self, question: str) -> bool:
        """Cheap upstream gate. Returns True if the query should proceed.

        Fails open: if the classifier errors, the question is allowed through
        rather than blocking a legitimate user.
        """
        try:
            verdict = self.client.complete(
                intent_messages(question), max_tokens=5, temperature=0.0
            )
        except Exception:  
            return True
        return "BLOCK" not in verdict.upper()

    # -- main entry points ---------------------------------------------------

    def answer(self, question: str, max_pages: int | None = None) -> RagAnswer:
        lang = detect_query_language(question)

        if self.guardrail and not self.check_intent(question):
            return RagAnswer(
                question=question,
                answer=out_of_scope_reply(lang),
                language=lang,
                blocked=True,
            )

        pages = self.retriever.retrieve(question, query_lang=lang, max_pages=max_pages)

        if not pages:
            from .prompts import REFUSAL_EN, REFUSAL_IT

            return RagAnswer(
                question=question,
                answer=REFUSAL_IT if lang == "it" else REFUSAL_EN,
                language=lang,
                refused=True,
            )

        context = format_context(pages)
        answer = self.client.complete(rag_messages(context, question, lang))

        return RagAnswer(
            question=question,
            answer=answer,
            language=lang,
            sources=[p.url for p in pages],
            pages=pages,
            cited=extract_citations(answer),
            refused=is_refusal(answer),
        )

    def answer_baseline(self, question: str) -> RagAnswer:
        """No-retrieval control, for the comparison table in the paper."""
        answer = self.client.complete(baseline_messages(question))
        return RagAnswer(
            question=question,
            answer=answer,
            language=detect_query_language(question),
            refused=is_refusal(answer),
        )
