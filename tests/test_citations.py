"""Unit tests for extract_citations() - covers the multi-source-bracket bug
fixed 2026-09-10 (the old regex only matched a single digit per bracket, so
"[2, 6]" or "[1,3,6,8]" style citations were silently dropped entirely).

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.pipeline import extract_citations  # noqa: E402


def test_single_citation():
    assert extract_citations("This is stated [3].") == [3]


def test_multi_citation_comma_space():
    assert extract_citations("Supported by two sources [2, 6].") == [2, 6]


def test_multi_citation_no_space():
    assert extract_citations("See [1,3,6,8] for details.") == [1, 3, 6, 8]


def test_multiple_separate_brackets():
    assert extract_citations("First point [2]. Second point [6].") == [2, 6]


def test_no_citations():
    assert extract_citations("No brackets here at all.") == []


def test_empty_and_none():
    assert extract_citations("") == []
    assert extract_citations(None) == []


def test_duplicate_ids_deduped_and_sorted():
    assert extract_citations("[6, 2] and again [2].") == [2, 6]
