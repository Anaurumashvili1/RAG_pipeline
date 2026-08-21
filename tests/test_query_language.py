"""Query-language detection.

The bug this pins: detect_query_language delegated to the *document* detector,
which refuses to commit below 20 words. Real questions are 5-12 words, so it
returned None for every query ever asked and the `or "en"` fallback fired every
time. Italian users received English refusal messages, and the
prefer_query_language tie-break boosted English pages for Italian queries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.retrieval import detect_query_language  # noqa: E402


ITALIAN = [
    "Quando aprono le iscrizioni a ingegneria ambientale?",
    "Come funziona il rimborso spese per le missioni?",
    "Quali sono i requisiti per l'ammissione a un corso di laurea magistrale?",
    "Dove si trova la biblioteca?",
    "Perche devo pagare le tasse?",          # unaccented, as users often type
    "Perché devo pagare le tasse?",          # accented
    "Quanto costa l'iscrizione?",
    "Che documenti servono per il tirocinio?",
]

ENGLISH = [
    "When do enrollment applications open?",
    "What documents do I need for the internship?",
    "How can I apply for a scholarship?",
    "Where is the library?",
    "Which courses are taught in English?",
    "What is the deadline for the graduation application?",
]


@pytest.mark.parametrize("q", ITALIAN)
def test_italian_questions(q):
    assert detect_query_language(q) == "it"


@pytest.mark.parametrize("q", ENGLISH)
def test_english_questions(q):
    assert detect_query_language(q) == "en"


def test_short_questions_are_detected_at_all():
    """The regression itself: these are all under the old 20-word floor."""
    assert len("Dove si trova la biblioteca?".split()) < 20
    assert detect_query_language("Dove si trova la biblioteca?") == "it"


def test_accents_are_decisive():
    assert detect_query_language("Università contatti") == "it"


def test_empty_and_meaningless_fall_back_without_raising():
    assert detect_query_language("") in ("it", "en")
    assert detect_query_language("xyz qwerty") in ("it", "en")
