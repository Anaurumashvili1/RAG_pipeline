"""Retrieval dedup / ranking tests using fake nodes - no index required."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.config import RetrievalCfg          # noqa: E402
from unitn_rag.evaluation import hit_at_k          # noqa: E402
from unitn_rag.prompts import is_refusal           # noqa: E402
from unitn_rag.retrieval import select_pages       # noqa: E402


def fake(score, url, group, lang="en", year=2026, text="body text"):
    node = SimpleNamespace(
        metadata={
            "url": url,
            "doc_group_id": group,
            "lang": lang,
            "title": f"title of {url}",
            "effective_year": year,
        },
        get_content=lambda metadata_mode=None: text,
    )
    return SimpleNamespace(node=node, score=score)


CFG = RetrievalCfg(similarity_top_k=20, max_pages=5, dedup_by="doc_group")


def test_translations_collapse_to_one_slot():
    results = [
        fake(0.90, "https://unitn.it/en/a", "grp-a", lang="en"),
        fake(0.88, "https://unitn.it/it/a", "grp-a", lang="it"),
        fake(0.70, "https://unitn.it/en/b", "grp-b", lang="en"),
    ]
    pages = select_pages(results, CFG, query_lang="en", current_year=2026)
    assert len(pages) == 2
    assert {p.url for p in pages} == {"https://unitn.it/en/a", "https://unitn.it/en/b"}


def test_query_language_breaks_ties():
    results = [
        fake(0.80, "https://unitn.it/it/a", "grp-a", lang="it"),
        fake(0.79, "https://unitn.it/en/a", "grp-a", lang="en"),
    ]
    assert select_pages(results, CFG, query_lang="en", current_year=2026)[0].lang == "en"
    assert select_pages(results, CFG, query_lang="it", current_year=2026)[0].lang == "it"


def test_fresher_sibling_beats_language_preference():
    """A 2026 Italian edition should outrank a 2020 English one for an English
    query - but only because the year is a genuine filename edition marker.
    A bare metadata year on a non-descriptive URL must NOT drive this (see
    test_metadata_only_year_does_not_drive_recency below) - that was the
    2026-09-10 regression: correct-but-undated documents lost to irrelevant
    pages that merely carried a fresher crawl-metadata year.
    """
    results = [
        fake(0.80, "https://unitn.it/it/guide-2026.pdf", "grp-a", lang="it", year=2026),
        fake(0.80, "https://unitn.it/en/guide-2020.pdf", "grp-a", lang="en", year=2020),
    ]
    assert select_pages(results, CFG, query_lang="en", current_year=2026)[0].lang == "it"


def test_stale_edition_is_demoted():
    """A filename-edition year is the one case recency should still act on -
    verified against question #3's stale HCI regulation PDFs."""
    results = [
        fake(0.90, "https://unitn.it/files/regolamento-2018.pdf", "grp-old", year=2018),
        fake(0.70, "https://unitn.it/files/regolamento-2026.pdf", "grp-new", year=2026),
    ]
    pages = select_pages(results, CFG, query_lang="en", current_year=2026)
    assert pages[0].url.endswith("2026.pdf")


def test_metadata_only_year_does_not_drive_recency():
    """The 2026-09-10 regression, reproduced directly: a correct document with
    only a crawl-metadata year (no filename edition marker) must not be
    demoted below an irrelevant page that merely has a fresher metadata year.
    Mirrors question #4 (CEILS guidelines, content-dated 2022) losing to
    unrelated pages crawl-dated 2026.
    """
    results = [
        fake(0.90, "https://corsi.unitn.it/en/ceils/guidelines", "grp-target", year=2022),
        fake(0.85, "https://corsi.unitn.it/en/unrelated-course/enrollment", "grp-other", year=2026),
    ]
    pages = select_pages(results, CFG, query_lang="en", current_year=2026)
    assert pages[0].url.endswith("/guidelines")


def test_max_pages_is_respected():
    results = [fake(0.9 - i / 100, f"https://unitn.it/en/{i}", f"grp-{i}") for i in range(20)]
    assert len(select_pages(results, CFG, query_lang="en", current_year=2026)) == 5
    assert len(select_pages(results, CFG, query_lang="en", current_year=2026, max_pages=10)) == 10


def test_dedup_by_url_keeps_translations_separate():
    cfg = RetrievalCfg(dedup_by="url", max_pages=5)
    results = [
        fake(0.90, "https://unitn.it/en/a", "grp-a", lang="en"),
        fake(0.88, "https://unitn.it/it/a", "grp-a", lang="it"),
    ]
    assert len(select_pages(results, cfg, query_lang="en", current_year=2026)) == 2


def test_hit_at_k_is_not_capped_by_truncation():
    """The v1 bug: hit@5 could never exceed the truncated 5-item source list."""
    sources = [f"https://unitn.it/en/{i}" for i in range(10)]
    target = "https://unitn.it/en/7"
    assert hit_at_k(sources, target, 1) is False
    assert hit_at_k(sources, target, 5) is False
    assert hit_at_k(sources, target, 10) is True


def test_hit_at_k_ignores_trailing_slash():
    assert hit_at_k(["https://unitn.it/en/a/"], "https://unitn.it/en/a", 1) is True


def test_refusal_detection():
    assert is_refusal("I don't know based on the provided documents.")
    assert is_refusal("Non lo so sulla base dei documenti forniti.")
    assert is_refusal("")
    assert not is_refusal("Enrolment opens in July [1].")


def test_family_cap_limits_near_duplicate_editions():
    """Three editions of one regulation should not fill 3 of 3 selected slots,
    as long as enough other-family candidates exist to fill the remaining ones."""
    cfg = RetrievalCfg(max_pages=3, dedup_by="doc_group", family_cap=1)
    results = [
        fake(0.95, "https://x.unitn.it/f/reg-hci-2015.pdf", "grp-2015"),
        fake(0.94, "https://x.unitn.it/f/reg-hci-2017.pdf", "grp-2017"),
        fake(0.93, "https://x.unitn.it/f/reg-hci-2018.pdf", "grp-2018"),
        fake(0.80, "https://corsi.unitn.it/en/hci/graduation/final-exam", "grp-current"),
        fake(0.75, "https://corsi.unitn.it/en/hci/other-page", "grp-other"),
    ]
    pages = select_pages(results, cfg, query_lang="en", current_year=2026)
    assert len(pages) == 3
    assert any("final-exam" in p.url for p in pages)
    # only one of the three same-family editions should have made it in
    same_family = [p for p in pages if "reg-hci-" in p.url]
    assert len(same_family) == 1


def test_family_cap_backfills_when_not_enough_other_candidates():
    """The cap must never return fewer than max_pages just to enforce itself."""
    cfg = RetrievalCfg(max_pages=3, dedup_by="doc_group", family_cap=1)
    results = [
        fake(0.95, "https://x.unitn.it/f/reg-hci-2015.pdf", "grp-2015"),
        fake(0.94, "https://x.unitn.it/f/reg-hci-2017.pdf", "grp-2017"),
        fake(0.93, "https://x.unitn.it/f/reg-hci-2018.pdf", "grp-2018"),
    ]
    pages = select_pages(results, cfg, query_lang="en", current_year=2026)
    assert len(pages) == 3


def test_family_cap_disabled_by_default():
    """family_cap defaults to 0 - existing behaviour is unchanged unless set."""
    cfg = RetrievalCfg(max_pages=3, dedup_by="doc_group")
    results = [
        fake(0.95, "https://x.unitn.it/f/reg-hci-2015.pdf", "grp-2015"),
        fake(0.94, "https://x.unitn.it/f/reg-hci-2017.pdf", "grp-2017"),
        fake(0.93, "https://x.unitn.it/f/reg-hci-2018.pdf", "grp-2018"),
    ]
    pages = select_pages(results, cfg, query_lang="en", current_year=2026)
    assert len(pages) == 3
