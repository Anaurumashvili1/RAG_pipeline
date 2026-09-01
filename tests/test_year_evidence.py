"""How a document's year is obtained decides how far it is trusted.

Four confirmed cases where the year in the Drupal upload path is not the year
the document is about. All four share one shape - an old document re-uploaded,
dated by the re-upload - and all four state their real date in their own text.
The strings below are copied from the corpus, not invented.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.text import (                    # noqa: E402
    academic_year_end,
    recency_penalty,
    resolve_effective_year,
    upload_path_year,
    year_from_document,
)

CY = 2026

CEILS = "https://corsi.unitn.it/sites/cds/files/2025-02/guidelines_trasfer_b.pdf"
MISSIONI = "https://www.unitn.it/sites/default/files/2024-11/Regolamento_missioni.pdf"


# --- academic years resolve to their end ---------------------------------

def test_span_resolves_to_its_end():
    assert academic_year_end(2025, 2026) == 2026
    assert academic_year_end(2025, 26) == 2026          # '2025/26' shorthand


def test_span_of_more_than_one_year_is_rejected():
    """'2016/2067' and '2018/2022' are both in the crawl."""
    assert academic_year_end(2016, 2067) is None
    assert academic_year_end(2018, 2022) is None
    assert academic_year_end(2025, None) is None


def test_current_guide_is_not_aged_by_the_convention():
    """The point of the end-year convention, stated as a ranking fact.

    a.a. 2025/26 is the current guide in 2026. Under the start-year convention
    it scored 0.5 against a page dated 2026 from its upload path - the correctly
    tagged document ranked below the carelessly dated one.
    """
    guide = resolve_effective_year({"academic_year": "2025/2026"}, CY)
    assert guide == 2026
    assert recency_penalty(guide, CY) == 1.0


# --- upload paths --------------------------------------------------------

def test_upload_path_year_is_recognised():
    assert upload_path_year(CEILS) == 2025
    assert upload_path_year("https://x.unitn.it/sites/default/files/2024/11/a.pdf") == 2024


def test_a_year_in_an_ordinary_path_is_not_an_upload_stamp():
    """Only the month-stamped upload directory counts; /2021/news is a real path."""
    assert upload_path_year("https://x.unitn.it/2021/news") is None
    assert upload_path_year("https://x.unitn.it/en/node/1603") is None
    assert upload_path_year(None) is None


def test_document_evidence_beats_the_upload_path():
    """The CEILS transfer guidelines: a 2022 document uploaded in February 2025."""
    raw = {
        "url": CEILS,
        "effective_year": 2025,
        "doc_type": "pdf",
        "text": "Guidelines for transfer\n... 4 Last updated on 22nd December 2022 ...",
    }
    assert resolve_effective_year(raw, CY) == 2022


def test_a_crawler_year_that_is_not_path_derived_is_still_trusted():
    """The demotion is specific to upload paths - re-deriving everything would
    leave half the corpus with no freshness signal at all."""
    raw = {"url": "https://x.unitn.it/node/9", "effective_year": 2019,
           "doc_type": "page", "text": "Last updated on 3 March 2024"}
    assert resolve_effective_year(raw, CY) == 2019


# --- emanation decrees ---------------------------------------------------

HEADER = ("Universita degli Studi di Trento Emanato con DR n. 480 del 29 luglio 2015 "
          "Pagina %d di 11 REGOLAMENTO PER LE MISSIONI ")


def test_running_header_dates_the_regolamento():
    """The travel regulation: emanated 2015, re-uploaded November 2024.

    The date is spelled out in Italian - 'del 29 luglio 2015', not '29/07/2015'
    - which is why a numeric-only pattern missed it.
    """
    raw = {"url": MISSIONI, "effective_year": 2024, "doc_type": "pdf",
           "text": (HEADER % 1) + "Art. 1 " + (HEADER % 2) + "Art. 2 " + (HEADER % 3)}
    assert resolve_effective_year(raw, CY) == 2015


def test_numeric_decree_dates_also_parse():
    text = ("Universita degli Studi di Trento Emanato con DR n. 637 del 13/07/2026 "
            "Pagina 1 di 18 " * 3)
    assert year_from_document(text, CY) == 2026


def test_a_cited_decree_is_not_this_document_s_date():
    """Every UniTn decreto opens by citing other decrees in identical words.

    A commissioni decree issued in July 2026 cites the Statute of 2024 and the
    Regolamento didattico of 2012. Reading either as its own date backdates a
    current document by up to fourteen years.
    """
    preamble = (
        "IL PRESIDENTE Visto lo Statuto dell'Universita degli Studi di Trento "
        "emanato con D.R. n. 5 di data 8 gennaio 2024; Visto il Regolamento "
        "Generale di Ateneo, emanato con D.R. n. 421 del 1 ottobre 2012, e da "
        "ultimo modificato con D.R. n. 606 del 29 maggio 2024; "
    )
    assert year_from_document(preamble, CY) is None


def test_a_decree_named_once_is_a_citation_not_a_header():
    """A running header repeats verbatim on every page; a preamble names each
    decree once. Repetition is the test."""
    once = "Regolamento adottato, emanato con D.R. n. 975 del 4 ottobre 2022, per il corso."
    assert year_from_document(once, CY) is None
    assert year_from_document(once + " " + once, CY) == 2022


def test_the_decree_rule_does_not_apply_to_html():
    """An HTML page has no running header, so a repeated decree there is prose.

    Measured: exactly one page in the corpus, disi/node/1603, which cites the
    Conto Terzi regolamento of 2015 and would otherwise be dated eleven years
    stale."""
    prose = ("previste dal Regolamento per attivita Conto Terzi emanato con D.R. "
             "n. 599 del 29 settembre 2015, e seguita dagli uffici. ") * 2
    assert year_from_document(prose, CY, is_pdf=True) == 2015
    assert year_from_document(prose, CY, is_pdf=False) is None


# --- filename edition labels --------------------------------------------

def test_filename_edition_beats_the_upload_path():
    """Three editions of the energy regolamento sat in one December 2024 upload
    directory and were all dated 2024, collapsing 2016, 2023 and 2024 into one."""
    raw = {"url": "https://corsi.unitn.it/sites/cds/files/2024-12/reg-energetica-2023.pdf",
           "effective_year": 2024, "doc_type": "pdf",
           "title": "regolamento-didattico-lm-ingegneria-energetica-2023.pdf", "text": ""}
    assert resolve_effective_year(raw, CY) == 2023
