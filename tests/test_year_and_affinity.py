"""Regression tests for the two retrieval-time corrections.

Both were found by grading the 41-question eval set answer by answer, and both
are measured against real URLs from dataset.v2.jsonl.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.retrieval import question_terms                    # noqa: E402
from unitn_rag.text import (                                       # noqa: E402
    recency_penalty,
    resolved_year,
    url_host_terms,
    url_terms,
)

ALF = "https://www.{host}.unitn.it/alfresco/download/workspace/SpacesStore/x/{name}"
GUIDA_2025 = ALF.format(host="giurisprudenza", name="GUIDA%20GIURISPRUDENZA%202025-26_PORTALE.pdf")
GUIDE_2002 = ALF.format(host="sociologia", name="4-2002_2003_student_guide_soc.pdf")
SYLL_2008 = ALF.format(host="sociologia", name="2008_2009_syllabus_lm_srs.pdf")


def test_filename_year_beats_crawl_fallback():
    """The crawl stored 2026 for every undated Alfresco PDF; filenames disagree."""
    assert resolved_year(2026, None, GUIDE_2002) == 2003
    assert resolved_year(2026, None, SYLL_2008) == 2009
    # a.a. 2025/26 resolves to its end year, per academic_year_end
    assert resolved_year(2025, None, GUIDA_2025) == 2026


def test_recency_ranking_is_no_longer_inverted():
    """Before the fix a 2002 guide scored 1.0 and the current guide scored 0.5."""
    current = recency_penalty(resolved_year(2025, None, GUIDA_2025), 2026)
    obsolete = recency_penalty(resolved_year(2026, None, GUIDE_2002), 2026)
    assert current > obsolete
    assert current == 1.0


def test_upload_path_year_is_treated_as_unknown():
    """A stored year that merely echoes the upload path is not evidence."""
    url = "https://corsi.unitn.it/sites/cds/files/2025-02/linee_guida.pdf"
    assert resolved_year(2025, None, url) is None


def test_stored_year_kept_when_nothing_better():
    assert resolved_year(2019, None, "https://corsi.unitn.it/en/some-page") == 2019


def _affinity(question: str, url: str, weight: float = 0.25) -> float:
    q = question_terms(question)
    host = 1.0 if q & url_host_terms(url) else 0.0
    path = len(q & (url_terms(url) - url_host_terms(url)))
    return 1.0 + weight * (host + 0.5 * min(path, 3) / 3.0)


def test_host_match_outranks_slug_match():
    """#38: physics is /node/433 with no slug; the CS page's slug matches three
    query words. Without host weighting the wrong department wins."""
    q = ("What documents are required for a research period in an external "
         "institution for masters students in physics?")
    assert _affinity(q, "https://www.physics.unitn.it/node/433") > _affinity(
        q, "https://corsi.unitn.it/en/computer-science-master/graduation/external-research-period"
    )


def test_course_slug_separates_sibling_calendars():
    """#18: six near-identical graduation calendars, one per course."""
    q = ("When is the deadline to upload a final thesis to your graduation "
         "application in Esse3 for Human Computer Interaction?")
    base = "https://corsi.unitn.it/en/{}/graduation/graduation-calendar"
    assert _affinity(q, base.format("human-computer-interaction")) > _affinity(
        q, base.format("computer-science-master")
    )


def test_department_alias_reaches_italian_host():
    """#23: 'faculty of law' must reach giurisprudenza and not DII, whose
    calendar states a different second-semester start for the same year."""
    q = "When does the classes of second semester start for the faculity of law?"
    assert _affinity(q, "https://www.giurisprudenza.unitn.it/node/997") > _affinity(
        q, "https://www.dii.unitn.it/alfresco/download/workspace/SpacesStore/x/Calendario.pdf"
    )


def test_ordinals_do_not_promote_distractors():
    """#27: 'third year' matched phd.../third-year-admission-requirements."""
    q = ("I enrolled in Law at Trento in 2019. How many complementary exams do "
         "I have to take from the third year onwards?")
    terms = question_terms(q)
    assert "third" not in terms and "year" not in terms
    assert "giurisprudenza" in terms  # alias of 'law'
    assert _affinity(q, GUIDA_2025) > _affinity(
        q, "https://phd.unitn.it/drsgce/en/541/third-year-admission-requirements"
    )
