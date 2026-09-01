"""Tests for the text/data changes made during the v2 corpus work.

Every one of these covers a bug that was found in production data and that
produced *plausible output* rather than an error - which is why they are worth
having. Runs in under a second with no models, no network, no corpus.

    python3 -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.text import (                       # noqa: E402
    clean_text,
    looks_letter_spaced,
    repair_letter_spacing,
    detect_language,
    is_junk_url,
    is_latin_script,
    max_plausible_year,
    parse_academic_year,
    recency_penalty,
    resolve_effective_year,
    year_from_title,
)

CY = 2026


# ---------------------------------------------------------------------------
# clean_text: paragraph structure
# ---------------------------------------------------------------------------

def test_clean_text_default_still_collapses_everything():
    assert clean_text("  hello \n  world  ") == "hello world"
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_clean_text_keep_breaks_preserves_paragraphs():
    """The bug: collapsing newlines made SentenceSplitter's paragraph_separator
    unmatchable, so chunk boundaries ignored document structure entirely."""
    raw = "First paragraph here.\n\n## A heading\n\nSecond paragraph here."
    out = clean_text(raw, keep_breaks=True)
    assert "\n\n" in out
    assert "## A heading" in out


def test_clean_text_keep_breaks_still_collapses_spaces_and_runs():
    out = clean_text("a    b\t\tc\n\n\n\n\nd", keep_breaks=True)
    assert "a b c" in out
    assert "\n\n\n" not in out          # 3+ newlines collapse to one blank line


# ---------------------------------------------------------------------------
# Academic years
# ---------------------------------------------------------------------------

def test_parse_academic_year_returns_the_end_of_the_span():
    """2025/2026 -> 2026, not 2025.

    A guide for a.a. 2025/26 is the current guide through the whole of 2026.
    Dating it 2025 makes 1/(1+age) treat it as a year old on publication, and
    ages every correctly tagged document by one against every document dated
    from a calendar-year upload path.
    """
    assert parse_academic_year("2025/2026", CY) == 2026
    assert parse_academic_year("2025/26", CY) == 2026


def test_parse_academic_year_rejects_malformed_spans():
    """'2016/2067' and '2018/2022' are in the crawl. A span must be one year."""
    assert parse_academic_year("2016/2067", CY) is None
    assert parse_academic_year("2018/2022", CY) is None
    assert parse_academic_year("garbage", CY) is None
    assert parse_academic_year(None, CY) is None


def test_year_ceiling_is_next_academic_year_not_2100():
    """A 2099 date under 1/(1+age) scores maximum freshness. The old ceiling
    was a hardcoded 2100, so parse errors outranked everything on the site."""
    assert max_plausible_year(CY) == 2027
    assert parse_academic_year("2098/2099", CY) is None


# ---------------------------------------------------------------------------
# resolve_effective_year: trust the crawler, fall back deliberately
# ---------------------------------------------------------------------------

def test_effective_year_prefers_academic_year():
    raw = {"effective_year": 2026, "academic_year": "2022/2023"}
    assert resolve_effective_year(raw, CY) == 2023


def test_effective_year_uses_crawler_value():
    assert resolve_effective_year({"effective_year": 2024}, CY) == 2024


def test_effective_year_clamps_implausible_crawler_value():
    assert resolve_effective_year({"effective_year": 2099}, CY) is None


def test_effective_year_falls_back_to_url():
    raw = {"effective_year": None, "url": "https://x.unitn.it/2021/news", "text": ""}
    assert resolve_effective_year(raw, CY) == 2021


def test_effective_year_uses_filename_when_crawler_defaulted_to_now():
    """Alfresco PDFs have UUID URLs and often no year in their first 500 chars,
    so the crawler returned current_year. Six Faculty of Law handbooks from
    2007-2010 were therefore dated 2026 and ranked as freshly published."""
    raw = {"effective_year": CY, "title": "02_Guida Magistrale 2007-08.pdf"}
    assert resolve_effective_year(raw, CY) == 2008


def test_effective_year_leaves_genuinely_current_pages_alone():
    raw = {"effective_year": CY, "title": "Ammissioni 2026 | UniTrento"}
    assert resolve_effective_year(raw, CY) == CY


def test_effective_year_ignores_title_when_crawler_had_a_real_signal():
    raw = {"effective_year": 2019, "title": "Guida Facolta 2007-2008.pdf"}
    assert resolve_effective_year(raw, CY) == 2019


def test_year_from_title_ignores_stray_years():
    """A year loose in the middle of a title is part of a sentence."""
    assert year_from_title("Premio 2019 assegnato", CY) is None
    assert year_from_title("News | JobGuidance", CY) is None
    assert year_from_title("09_Guida Facolta 2012-2013.pdf", CY) == 2013


def test_year_from_title_reads_a_terminal_edition_year():
    """'regolamento-...-2023.pdf' is an edition label. Position is what makes it
    safe: at the end of the stem it names the file, anywhere else it is prose.

    Returned as-is, not as a span: the filename claims 2023, and reading it as
    a.a. 2023/24 would invent a claim it does not make - which would land the
    file back on the same year as the upload path already known to be wrong."""
    assert year_from_title("regolamento-didattico-lm-ingegneria-energetica-2023.pdf", CY) == 2023
    assert year_from_title("Consorzi_2020.pdf", CY) == 2020
    assert year_from_title("Premio 2019 assegnato ai vincitori", CY) is None


def test_unknown_year_is_penalised_not_rewarded():
    assert recency_penalty(CY, CY) == 1.0
    assert recency_penalty(None, CY) == 0.5
    assert recency_penalty(2007, CY) < 0.1


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

def test_url_marker_beats_declared_language():
    """A site-wide <html lang> is often contradicted by the page's own URL."""
    assert detect_language(url="https://x.unitn.it/en/1/a",
                           text="Il corso di laurea prevede un test",
                           declared="it") == "en"


def test_english_body_overrides_declared_italian_on_unaliased_node_url():
    """Drupal serves /node/N under the site default language. 1,565 documents
    were English text filed as Italian."""
    text = ("The Nanoscience Laboratory focuses on the physical phenomena of photons "
            "and their propagation in integrated structures. The research of the group "
            "is devoted to the study of silicon and to the development of devices.")
    assert detect_language(url="https://www.physics.unitn.it/node/867",
                           text=text, declared="it") == "en"


def test_italian_page_quoting_english_is_not_flipped():
    text = ("Il corso di laurea magistrale in Physics of Data and the Department of "
            "Physics offrono agli studenti una didattica che prevede corsi presso il "
            "dipartimento con una prova finale")
    assert detect_language(url="https://www.physics.unitn.it/node/13",
                           text=text, declared="it") == "it"


def test_language_defaults_to_italian_with_no_signal():
    assert detect_language(url="https://x.unitn.it/node/9", text="", declared=None) == "it"


def test_latin_script_guard():
    assert is_latin_script("Il corso di laurea prevede un test di ammissione obbligatorio")
    assert not is_latin_script("奖学金 2026 申请截止日期是三月三十一日请查看官方网站获取更多信息")
    # An Italian page naming a Chinese author must not be excluded.
    assert is_latin_script("Il professore Zhang 张伟 terra una lezione sulla linguistica")
    assert is_latin_script("")


# ---------------------------------------------------------------------------
# AppleDouble stubs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Letter-spaced PDFs
# ---------------------------------------------------------------------------

SPACED = ("W e l c o m e  t o  t h e  I n t e r n s h i p\n"
          "H e r e  y o u  w i l l  f i n d  t h e  a n s w e r s  t o  t h e  "
          "m o s t  f r e q u e n t l y  a s k e d  q u e s t i o n s\n")


def test_letter_spacing_is_detected_and_repaired():
    """Design-tool PDFs position each glyph, so pypdf emits a space between
    every character. Single space = padding, double space = word boundary."""
    assert looks_letter_spaced(SPACED)
    fixed = repair_letter_spacing(SPACED)
    assert "Welcome to the Internship" in fixed
    assert "frequently asked questions" in fixed


def test_markdown_tables_are_not_mistaken_for_letter_spacing():
    """Tables are full of one-character tokens too - but '|' and '-', not
    letters. Repairing them would destroy real content."""
    table = "| Language | Coordinator |\n|---|---|\n| English | Roger Smith |\n" * 20
    assert not looks_letter_spaced(table)
    assert repair_letter_spacing(table) == table


def test_ordinary_prose_is_untouched():
    prose = ("Il corso di laurea in ingegneria ambientale prevede un test di "
             "ammissione obbligatorio per tutti gli studenti che intendono "
             "iscriversi al primo anno accademico presso il dipartimento.")
    assert repair_letter_spacing(prose) == prose
    assert repair_letter_spacing(None) is None


def test_partly_spaced_document_repairs_only_the_spaced_lines():
    mixed = SPACED + "This paragraph is perfectly normal and must survive intact.\n"
    fixed = repair_letter_spacing(mixed)
    assert "This paragraph is perfectly normal and must survive intact." in fixed


# ---------------------------------------------------------------------------
# duplicate_of is a canonical-URL claim, not an observed duplicate
# ---------------------------------------------------------------------------

def _write(tmp, rows):
    import json as _json
    p = tmp / "corpus.jsonl"
    p.write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_duplicate_of_is_kept_when_its_canonical_was_never_fetched(tmp_path=None):
    """The corpus has 2,426 records carrying duplicate_of. None shares
    content_sha256 with another document, and 87% name a canonical that was
    never crawled - so dropping on the flag deleted 2,102 documents whose text
    exists exactly once."""
    import tempfile
    from pathlib import Path as _P
    from unitn_rag.data import load_documents

    tmp = _P(tempfile.mkdtemp())
    body = "Il bando di selezione prevede requisiti specifici per i candidati. " * 8
    rows = [
        {"url": "https://x.unitn.it/node/1", "text": body, "title": "A",
         "duplicate_of": "https://x.unitn.it/en/1/pretty-alias"},   # never fetched
        {"url": "https://x.unitn.it/node/2", "text": body + " due", "title": "B",
         "duplicate_of": "https://x.unitn.it/node/3"},              # present below
        {"url": "https://x.unitn.it/node/3", "text": body + " tre", "title": "C"},
    ]
    docs = load_documents(_write(tmp, rows), min_chars=150, current_year=2026)
    urls = {d.url for d in docs}

    assert "https://x.unitn.it/node/1" in urls, "canonical absent - must be kept"
    assert "https://x.unitn.it/node/2" not in urls, "canonical present - safe to drop"
    assert "https://x.unitn.it/node/3" in urls


def test_appledouble_stubs_are_junk():
    assert is_junk_url("https://disi.unitn.it/x/._assignment.pdf")
    assert is_junk_url("https://disi.unitn.it/x/._qnets-Mair.pdf")
    assert not is_junk_url("https://disi.unitn.it/x/assignment.pdf")
    assert not is_junk_url("https://event.unitn.it/tbs-cnw/map-bus.pdf")
    assert not is_junk_url("")
