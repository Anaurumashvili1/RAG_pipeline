"""Retrieve in both languages, not just the question's own.

The corpus is 36k Italian documents to 25k English, and some facts exist in
only one of them - the eval set names three. BGE-M3 is multilingual and does
bridge the gap partially, but not enough against same-language competition.
Measured target ranks, original query vs a translation:

    #27  complementary exams   >1000  ->    34
    #0   application deadline    317  ->    50
    #3   thesis grading scale    311  ->   152

So: translate the question, retrieve for both, union the candidates, and let
the reranker score the lot. The union matters - a bad translation contributes
candidates that get outranked, it cannot remove what the original already
found. Worst case is wasted compute, not a worse answer.

Two things learned by measuring rather than guessing:

  * The model translates domain vocabulary well ('esami complementari' came
    out exactly right) but not proper nouns - it rendered Environmental
    Engineering as 'Ingegneria Ambientale' where UniTn calls the course
    'Ingegneria per l'ambiente e il territorio', costing 47 ranks.
  * Telling it to "keep course names unchanged" backfired: it left "Laurea
    Magistrale" untranslated inside an English query and the target fell from
    132 to 406 - worse than not translating at all. Acronyms are the only
    thing that should survive; degree types must be translated.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_SYSTEM = """You translate questions for a University of Trento search system.

Translate the question into {target}. Output only the translation, nothing else.

Do not translate these acronyms: CEILS, CFU, ECTS, DR, LM, PhD, ISEE, SPID, C3A,
CIMeC, DISI, DICAM, DII, BUD, FAQ, PEC, Esse3.
Do translate degree types: laurea magistrale = master's degree, laurea triennale
= bachelor's degree, laurea a ciclo unico = single-cycle degree.
Use the vocabulary of Italian university administration: esami complementari,
piano di studio, immatricolato, tirocinio, scadenza, discussione della tesi,
appello, ordinamento, trasferimento, immatricolazione."""

_OTHER = {"it": "en", "en": "it"}
_NAMES = {"it": "Italian", "en": "English"}

# The model sometimes answers with "Translation: ..." or wraps in quotes.
_PREFIX_RE = re.compile(r"^\s*(translation|traduzione)\s*[:\-]\s*", re.IGNORECASE)


class QueryTranslator:
    """Translate a question into the other supported language, with a cache.

    The eval set is 41 fixed questions and gets re-run constantly; caching by
    question hash keeps repeated runs both free and deterministic, which
    matters when comparing arms.
    """

    def __init__(self, client, cache_path: str | Path | None = None,
                 languages: tuple[str, ...] = ("it", "en")):
        self.client = client
        self.languages = languages
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - a corrupt cache must not stop a run
                self._cache = {}

    def _key(self, question: str, target: str) -> str:
        h = hashlib.sha1(question.encode("utf-8")).hexdigest()[:16]
        return f"{target}:{h}"

    def _save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def translate(self, question: str, source_lang: str) -> str | None:
        """Return the question in the other language, or None if unavailable.

        Never raises: a translation failure must degrade to single-language
        retrieval, which is what the pipeline did before this existed.
        """
        target = _OTHER.get(source_lang)
        if target is None or target not in self.languages:
            return None

        key = self._key(question, target)
        if key in self._cache:
            return self._cache[key] or None

        try:
            out = self.client.complete([
                {"role": "system", "content": _SYSTEM.format(target=_NAMES[target])},
                {"role": "user", "content": question},
            ])
        except Exception as exc:  # noqa: BLE001
            print(f"[translate] failed ({type(exc).__name__}), continuing "
                  f"with the original question only")
            return None

        out = _PREFIX_RE.sub("", (out or "").strip()).strip('"“” ')
        # A translation that came back empty, or identical, adds nothing.
        if not out or out.strip().lower() == question.strip().lower():
            out = ""
        self._cache[key] = out
        self._save()
        return out or None


def union_nodes(*runs):
    """Merge candidate sets, keeping the best dense score per chunk.

    Union rather than fusion: the reranker rescores everything downstream, so
    a fused pre-score would only be overwritten. What matters here is that a
    chunk found by either phrasing survives to be scored.
    """
    seen: dict[str, object] = {}
    for run in runs:
        for n in run or []:
            nid = getattr(n.node, "node_id", None) or id(n.node)
            prev = seen.get(nid)
            if prev is None or (n.score or 0.0) > (prev.score or 0.0):
                seen[nid] = n
    return list(seen.values())
