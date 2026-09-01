"""hit@k over a set of acceptable URLs, and items that are not retrieval tests.

The bug these cover: ``select_pages`` collapses an IT/EN pair to one slot and
reports whichever sibling ranked higher, but ``hit_at_k`` compared against a
single ``target_url`` - so a correct retrieval that surfaced the sibling was
recorded as a miss. Separately, items no retriever can fairly hit were sitting
in the denominator as guaranteed zeros.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.evaluation import (              # noqa: E402
    hit_at_k,
    is_retrieval_scored,
    retrieval_metrics,
    targets_of,
)

IT = "https://www.unitn.it/it/rit"
EN = "https://www.unitn.it/en/international/coming-unitrento/rit-raise-your-international-talent"


# --- hit_at_k ------------------------------------------------------------

def test_single_url_still_works():
    """Back-compat: the old call signature, a bare string."""
    assert hit_at_k([IT], IT, 1) is True
    assert hit_at_k(["https://unitn.it/en/other"], IT, 1) is False


def test_sibling_counts_as_a_hit():
    """The false negative: retrieval surfaced EN, the set names IT."""
    assert hit_at_k([EN], [IT, EN], 1) is True
    assert hit_at_k([EN], IT, 1) is False          # what it used to do


def test_pipe_separated_string_is_accepted():
    """The candidate worksheet emits acceptable_urls as one pipe-joined cell."""
    assert hit_at_k([EN], f"{IT}|{EN}", 1) is True


def test_k_truncates_the_source_list_not_the_targets():
    sources = ["https://a", "https://b", "https://c", "https://d", EN]
    assert hit_at_k(sources, [IT, EN], 4) is False
    assert hit_at_k(sources, [IT, EN], 5) is True


def test_normalisation_of_scheme_and_trailing_slash():
    assert hit_at_k(["http://www.unitn.it/it/rit/"], IT, 1) is True


def test_empty_target_never_hits():
    assert hit_at_k([IT], [], 1) is False
    assert hit_at_k([IT], None, 1) is False


# --- targets_of ----------------------------------------------------------

def test_target_url_leads_the_acceptable_set():
    item = {"target_url": IT, "acceptable_urls": [EN]}
    assert targets_of(item) == [IT, EN]


def test_target_url_is_not_duplicated_when_already_listed():
    item = {"target_url": IT, "acceptable_urls": [IT + "/", EN]}
    assert targets_of(item) == [IT + "/", EN]


# --- items that are not retrieval tests ----------------------------------

def test_answer_only_item_is_not_scored():
    """Item 12: a contact block repeated in 53 documents, question names no course."""
    assert is_retrieval_scored({"target_url": IT, "answer_only": True}) is False


def test_out_of_scope_item_is_not_scored():
    assert is_retrieval_scored({"acceptable_urls": [], "out_of_scope": True}) is False


def test_item_with_no_target_is_not_scored():
    assert is_retrieval_scored({"question": "?"}) is False


def test_ordinary_item_is_scored():
    assert is_retrieval_scored({"target_url": IT}) is True


# --- retrieval_metrics ---------------------------------------------------

def _row(**kw):
    row = {f"hit@{k}": False for k in (1, 3, 5, 10)}
    row.update(kw)
    return row


def test_excluded_items_leave_the_denominator():
    results = [
        _row(**{"hit@1": True, "hit@3": True, "hit@5": True, "hit@10": True}),
        _row(),
        _row(**{f"hit@{k}": None for k in (1, 3, 5, 10)}),   # answer-only
    ]
    m = retrieval_metrics(results)
    assert m["n"] == 2 and m["n_excluded"] == 1
    assert m["hit@1"] == 0.5          # not 0.3333


def test_all_excluded_reports_a_note_not_a_zero():
    results = [_row(**{f"hit@{k}": None for k in (1, 3, 5, 10)})]
    m = retrieval_metrics(results)
    assert m["n"] == 0 and "note" in m
    assert "hit@1" not in m
