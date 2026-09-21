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
    document_family_key,
    extract_effective_year,
    recency_penalty,
    title_edition_year,
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


def test_url_marker_wins_over_declared():
    """Superseded assertion, kept as the record of a deliberate reversal.

    This test used to assert that ``declared`` beat the URL marker. Commit
    07901fd reversed the precedence: Drupal serves unaliased ``/node/N`` under
    the site default, so an English body arrives carrying ``<html lang="it">``
    and the declaration is the least trustworthy signal of the three. The
    aliased ``/it/`` and ``/en/`` prefixes are assigned per translation and are
    reliable, so they now win. See test_text_v2.py for the full precedence.
    """
    assert detect_language(url="https://www.unitn.it/it/x", declared="en-GB") == "it"


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
    assert year == 2027                 # end of the span, and the URL's 2019 loses


def test_effective_year_from_italian_abbreviation():
    """The END of the span: a.a. 2025/26 is the current year through 2026."""
    assert extract_effective_year(text="A.A. 2025/2026 - iscrizioni aperte") == 2026
    assert extract_effective_year(text="A.A. 2025/26 - iscrizioni aperte") == 2026


def test_effective_year_from_url_when_text_silent():
    assert extract_effective_year(url="https://www.unitn.it/en/2024/news", text="No year here") == 2024


def test_effective_year_none_when_absent():
    assert extract_effective_year(url="https://www.unitn.it/en/page", text="No year here") is None


def test_recency_penalty_curve():
    assert recency_penalty(2026, 2026) == 1.0
    assert round(recency_penalty(2024, 2026), 2) == 0.33
    assert recency_penalty(None, 2026) == 0.5
    assert recency_penalty(2030, 2026) == 1.0  # future-dated docs are not penalised


def test_document_family_key_collapses_same_regulation_across_years():
    a = "https://corsi.unitn.it/sites/cds/files/2024-12/regolamento-didattico-lm-human-computer-interaction-2015.pdf"
    b = "https://corsi.unitn.it/sites/cds/files/2024-12/regolamento-didattico-lm-human-computer-interaction-2017.pdf"
    c = "https://corsi.unitn.it/sites/cds/files/2024-12/regolamento-didattico-lm-human-computer-interaction-2018.pdf"
    assert document_family_key(a) == document_family_key(b) == document_family_key(c)


def test_document_family_key_keeps_different_regulations_apart():
    """Same course, different regulation - must not collapse together."""
    teaching = "https://corsi.unitn.it/sites/cds/files/x/regolamento-didattico-lm-human-computer-interaction-2018.pdf"
    completion = "https://corsi.unitn.it/sites/cds/files/x/regolamento-conseguimento-titolo-lm-hci-2017.pdf"
    assert document_family_key(teaching) != document_family_key(completion)


def test_document_family_key_none_without_edition_year():
    assert document_family_key("https://corsi.unitn.it/en/human-computer-interaction/graduation/final-exam") is None
    assert document_family_key(None) is None
    assert document_family_key("") is None


def test_document_family_key_collapses_cineca_course_and_its_modules():
    course = "https://unitn.coursecatalogue.cineca.it/corsi/2026/10859?lang=it"
    module_a = "https://unitn.coursecatalogue.cineca.it/corsi/2026/10859/insegnamenti/2026/51297_658844_93999/2026/51297?coorte=2026&schemaid=9588&lang=it"
    module_b = "https://unitn.coursecatalogue.cineca.it/corsi/2026/10859/insegnamenti/2026/51297_658852_95450/2026/51297?coorte=2026&schemaid=9588&lang=en"
    assert document_family_key(course) == document_family_key(module_a) == document_family_key(module_b)


def test_document_family_key_keeps_different_cineca_courses_apart():
    course_a = "https://unitn.coursecatalogue.cineca.it/corsi/2026/10859?lang=it"
    course_b = "https://unitn.coursecatalogue.cineca.it/corsi/2026/99999?lang=it"
    assert document_family_key(course_a) != document_family_key(course_b)


def test_title_edition_year_reads_filename_marker():
    assert title_edition_year(
        None,
        "https://corsi.unitn.it/sites/cds/files/2024-12/regolamento-didattico-lm-human-computer-interaction-2017.pdf",
        current_year=2026,
    ) == 2017


def test_title_edition_year_ignores_metadata_only_years():
    # None of these carry a filename/title edition marker - a body-text
    # mention or URL date segment must not leak through as a penalty signal,
    # even though resolved_year() may legitimately report a year for display.
    assert title_edition_year(None, "https://www.giurisprudenza.unitn.it/node/1025", current_year=2026) is None
    assert title_edition_year(
        None,
        "https://corsi.unitn.it/sites/cds/files/2025-02/guidelines_trasfer_b-comparative-european_internation_legal_studies.pdf",
        current_year=2026,
    ) is None
    assert title_edition_year(None, "https://www.soi.unitn.it/guidelines", current_year=2026) is None


def test_title_edition_year_none_without_url_or_title():
    assert title_edition_year(None, None) is None
    assert title_edition_year("", "") is None
