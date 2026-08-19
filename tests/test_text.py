"""Unit tests for the pure helpers - no models, no network, run in under a second.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.text import (  # noqa: E402
    canonical_group_url,
    clean_text,
    detect_language,
    doc_group_id,
    extract_effective_year,
    recency_penalty,
)


def test_clean_text_collapses_whitespace_and_nbsp():
    assert clean_text("  hello \n  world  ") == "hello world"
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_language_from_url_path():
    assert detect_language(url="https://www.unitn.it/en/ateneo/123/x") == "en"
    assert detect_language(url="https://www.unitn.it/it/ateneo/123/x") == "it"


def test_language_from_query_param_and_subdomain():
    assert detect_language(url="https://www.unitn.it/page?lang=en") == "en"
    assert detect_language(url="https://international.unitn.it/apply") == "en"


def test_language_declared_wins():
    assert detect_language(url="https://www.unitn.it/it/x", declared="en-GB") == "en"


def test_language_from_text_fallback():
    it_text = (
        "La domanda di iscrizione per il corso di laurea deve essere presentata "
        "presso la segreteria studenti dell ateneo con i documenti richiesti."
    )
    en_text = (
        "The application for the degree course must be submitted to the student "
        "office of the university with all of the required documents and forms."
    )
    assert detect_language(url="https://x.example/page", text=it_text) == "it"
    assert detect_language(url="https://x.example/page", text=en_text) == "en"


def test_translations_share_a_doc_group():
    en = "https://www.unitn.it/en/ateneo/1234/final-exam"
    it = "https://www.unitn.it/it/ateneo/1234/final-exam"
    assert canonical_group_url(en) == canonical_group_url(it)
    assert doc_group_id(en) == doc_group_id(it)


def test_different_pages_do_not_share_a_doc_group():
    a = "https://www.unitn.it/en/ateneo/1234/final-exam"
    b = "https://www.unitn.it/en/ateneo/5678/enrolment"
    assert doc_group_id(a) != doc_group_id(b)


def test_hreflang_group_overrides_url_heuristic():
    a = "https://www.unitn.it/en/some-page"
    b = "https://webmagazine.unitn.it/totally/different/path"
    assert doc_group_id(a, hreflang_group="grp-1") == doc_group_id(b, hreflang_group="grp-1")


def test_effective_year_prefers_text_over_url():
    year = extract_effective_year(
        url="https://www.unitn.it/en/2019/page",
        text="Academic Year 2026/2027 - enrolment information for new students",
    )
    assert year == 2026


def test_effective_year_from_italian_abbreviation():
    assert extract_effective_year(text="A.A. 2025/2026 - iscrizioni aperte") == 2025


def test_effective_year_from_url_when_text_silent():
    assert extract_effective_year(url="https://www.unitn.it/en/2024/news", text="No year here") == 2024


def test_effective_year_none_when_absent():
    assert extract_effective_year(url="https://www.unitn.it/en/page", text="No year here") is None


def test_recency_penalty_curve():
    assert recency_penalty(2026, 2026) == 1.0
    assert round(recency_penalty(2024, 2026), 2) == 0.33
    assert recency_penalty(None, 2026) == 0.5
    assert recency_penalty(2030, 2026) == 1.0  # future-dated docs are not penalised
