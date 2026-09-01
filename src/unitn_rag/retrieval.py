"""Retrieval: fetch chunks, then collapse them to distinct source documents.

Two changes from the notebook version:

1. Dedup key is ``doc_group_id`` rather than ``url``, so the Italian and English
   versions of one page no longer occupy two of the five context slots.
2. When translations tie, the version in the user's language wins - unless the
   other-language sibling is materially fresher, in which case the fresher one
   is used and the LLM answers across languages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .config import RetrievalCfg
from .text import (
    _WORD_RE,
    detect_language_from_text,
    recency_penalty,
    resolved_year,
    url_host_terms,
    url_terms,
)


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


_IT_QUERY_WORDS = frozenset("""
quando come quali quale dove chi perche cosa quanto quanta quanti quante che
posso devo serve sono e un una il lo la le gli dei delle degli di da in con su
per tra fra non si ci mi ti al alla allo ai agli alle dal della del nel nella
iscrizione iscrizioni laurea corso corsi esame esami tasse borsa borse studenti
scadenza scadenze domanda requisiti ateneo universita
""".split())

_EN_QUERY_WORDS = frozenset("""
when how what where who why which can could should do does did is are was were
a an the of to in for with on at from my i you it and or if there
enrollment enrolment admission deadline application requirements tuition
scholarship course courses exam exams student students degree
""".split())

# à è é ì í ò ó ù ú - present in Italian, essentially absent from English.
_IT_ACCENT_RE = re.compile(r"[àèéìíòóùú]", re.IGNORECASE)


# A department named in a question rarely matches its own host label: "faculty
# of law" has to reach giurisprudenza.unitn.it. Only unambiguous names are
# listed - 'civil' or 'environmental' would pull DICAM pages into questions
# about the Environmental Engineering *course*, which lives on corsi.unitn.it.
_HOST_ALIASES = {
    "law": ("giurisprudenza",),
    "legal": ("giurisprudenza",),
    "giurisprudenza": ("giurisprudenza",),
    "sociology": ("sociologia",),
    "sociologia": ("sociologia",),
    "economics": ("economia",),
    "economia": ("economia",),
    "physics": ("physics", "fisica"),
    "fisica": ("physics", "fisica"),
    "mathematics": ("maths",),
    "matematica": ("maths",),
    "library": ("biblioteca",),
    "biblioteca": ("biblioteca",),
    "humanities": ("lettere",),
    "lettere": ("lettere",),
}

# The two query-language lists were built to *detect* a language, so they miss
# ordinals, quantifiers and filler verbs. Those matter here: 'third year' in a
# question about Law matched phd.unitn.it/.../third-year-admission-requirements
# twice while the correct Giurisprudenza guide matched once, so an uncurated
# stoplist actively promoted the distractor.
_GENERIC_TERMS = frozenset('''
year years anno anni annual first second third fourth fifth sixth
primo secondo terzo quarto quinto next last previous current
take taken taking have has had need needs needed get got find finds found
know tell give make made want would like use used using onwards onward
enrolled enrolling enrol enroll apply applying applied require required
requirement requirements information info detail details
please thanks thank about after before during still also more most
new old any all some each every other another same different
'''.split())

_QUERY_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def question_terms(question: str) -> frozenset[str]:
    """Content words of a question, for matching against URL host and slug.

    The two query-language word lists are reused as the stoplist: they are
    function words plus generic academic vocabulary ('course', 'deadline',
    'esame'), and a word that appears in every UniTn URL cannot discriminate
    between them.
    """
    if not question:
        return frozenset()
    stop = _IT_QUERY_WORDS | _EN_QUERY_WORDS | _GENERIC_TERMS
    terms = {
        w for w in _QUERY_SPLIT_RE.split(question.lower())
        if len(w) > 2 and not w.isdigit() and w not in stop
    }
    for w in tuple(terms):
        terms.update(_HOST_ALIASES.get(w, ()))
    return frozenset(terms)


def detect_query_language(question: str) -> str:
    """Language of a *question*, which is far shorter than a document.

    The document detector requires 20+ words before it will commit, and a
    question is typically 5-12. It therefore returned None for every query ever
    asked, and the ``or "en"`` fallback meant Italian users received English
    refusal messages and had English pages preferred by the retrieval
    tie-break. This detector is tuned for short text: interrogatives, function
    words, and Italian accented characters.
    """
    if not question:
        return "en"

    q = question.lower()
    words = _WORD_RE.findall(q)

    it = sum(1 for w in words if w in _IT_QUERY_WORDS)
    en = sum(1 for w in words if w in _EN_QUERY_WORDS)

    # Accents are near-decisive on their own: English does not use them, and
    # Italian question words are full of them (perché, può, università).
    if _IT_ACCENT_RE.search(q):
        it += 3

    if it != en:
        return "it" if it > en else "en"

    # Nothing conclusive. The fallback decides only which refusal wording an
    # unidentifiable user sees - the answer language comes from the model
    # following the question (prompts.py rule 6), and is_refusal() matches both
    # languages, so neither retrieval nor the metrics depend on this. Left at
    # 'en' because that was the existing behaviour and there is no evidence
    # either way.
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
    question: str = "",
) -> list[RetrievedPage]:
    """Collapse ranked chunks into distinct documents.

    ``raw_results`` is the list of NodeWithScore returned by a LlamaIndex retriever.
    """
    current_year = current_year or date.today().year
    limit = max_pages or cfg.max_pages
    weight = getattr(cfg, "url_affinity_weight", 0.0)
    q_terms = question_terms(question) if weight else frozenset()

    best: dict[str, dict] = {}

    for r in raw_results:
        meta = r.node.metadata or {}
        key = _group_key(meta, cfg.dedup_by)
        if not key:
            continue

        url = meta.get("url", "")
        year = meta.get("effective_year")
        if getattr(cfg, "resolve_year_from_title", False):
            year = resolved_year(year, meta.get("title"), url, current_year)

        score = float(r.score) if r.score is not None else 0.0
        if apply_recency:
            score *= recency_penalty(year, current_year)
        if cfg.prefer_query_language and meta.get("lang") == query_lang:
            score *= 1.10          # mild tie-break, not an override
        if q_terms:
            # A host match counts full; path/slug matches count half and are
            # capped at three, so 'external-research-period' on the computer
            # science page cannot outrank physics.unitn.it for a physics
            # question - the physics page is /node/433 and has no slug to match.
            host_hit = 1.0 if q_terms & url_host_terms(url) else 0.0
            path_hits = len(q_terms & (url_terms(url) - url_host_terms(url)))
            affinity = host_hit + 0.5 * min(path_hits, 3) / 3.0
            if affinity:
                score *= 1.0 + weight * affinity

        candidate = {
            "score": score,
            "url": url,
            "title": meta.get("title", ""),
            "lang": meta.get("lang", ""),
            "effective_year": year,
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
        return select_pages(
            raw, self.cfg, query_lang=lang, max_pages=max_pages, question=question
        )


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
