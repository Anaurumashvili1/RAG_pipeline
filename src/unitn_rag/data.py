"""Corpus loading and cleaning (Colab cells 0-1)."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from .text import (
    clean_text,
    detect_language,
    doc_group_id,
    doc_id_from_url,
    resolve_effective_year,
)


@dataclass
class Doc:
    doc_id: str
    url: str
    title: str
    text: str
    lang: str
    doc_group_id: str
    effective_year: int | None = None
    doc_type: str | None = None
    department: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[data] skipping malformed JSON on line {line_no}")


def load_documents(
    path: str | Path,
    min_chars: int = 150,
    max_docs: int | None = None,
    drop_duplicates: bool = True,
    drop_low_content: bool = True,
    drop_boilerplate: bool = True,
    keep_languages: tuple[str, ...] | list[str] | None = ("it", "en"),
    current_year: int | None = None,
) -> list[Doc]:
    """Load, clean, filter and enrich the crawl output.

    The v2 crawl already carries ``lang``, ``effective_year`` and per-document
    quality flags. This trusts those and only falls back to deriving values
    itself where the crawl left a gap - re-deriving everything from scratch
    discarded work the crawler had already done correctly.
    """
    docs: list[Doc] = []
    seen_ids: set[str] = set()
    skipped_lang: dict[str, int] = {}

    for raw in iter_jsonl(path):
        url = (raw.get("url") or "").strip()
        title = clean_text(raw.get("title"))
        # keep_breaks: paragraph structure is what SentenceSplitter splits on.
        text = clean_text(raw.get("text"), keep_breaks=True)

        if not url or len(text) < min_chars:
            continue

        # Quality flags decided during crawling. Cheaper and more accurate than
        # re-deciding here, since the crawler saw the raw HTML and we do not.
        if drop_duplicates and raw.get("duplicate_of"):
            continue
        if drop_low_content and raw.get("low_content"):
            continue
        if drop_boilerplate and raw.get("boilerplate"):
            continue

        # Language scope. The crawler's `lang` is authoritative - it read
        # <html lang> - and it is the only place zh is recorded correctly:
        # detect_language() knows only it/en, so a Chinese page silently
        # resolves to 'en' and then competes for English queries.
        # Only drop when the crawler actually declared something; a null lang
        # falls through to detection as before.
        if keep_languages:
            declared = (raw.get("lang") or "").strip().lower().split("-")[0]
            if declared and declared not in keep_languages:
                skipped_lang[declared] = skipped_lang.get(declared, 0) + 1
                continue

        did = doc_id_from_url(url)
        if did in seen_ids:          # same URL crawled twice
            continue
        seen_ids.add(did)

        docs.append(
            Doc(
                doc_id=did,
                url=url,
                title=title,
                text=text,
                lang=detect_language(url=url, text=text, declared=raw.get("lang")),
                doc_group_id=doc_group_id(url, hreflang_group=raw.get("hreflang_group")),
                effective_year=resolve_effective_year(raw, current_year=current_year),
                doc_type=raw.get("doc_type"),
                department=raw.get("department"),
            )
        )

        if max_docs and len(docs) >= max_docs:
            break

    if skipped_lang:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(skipped_lang.items()))
        print(f"[data] skipped out-of-scope languages: {summary}")

    return docs


def corpus_stats(docs: list[Doc]) -> dict:
    """Quick sanity numbers - run this before every index build."""
    langs: dict[str, int] = {}
    years: dict[str, int] = {}
    for d in docs:
        langs[d.lang] = langs.get(d.lang, 0) + 1
        key = str(d.effective_year) if d.effective_year else "unknown"
        years[key] = years.get(key, 0) + 1

    groups = {d.doc_group_id for d in docs}
    return {
        "documents": len(docs),
        "doc_groups": len(groups),
        "translated_pairs": len(docs) - len(groups),
        "by_language": dict(sorted(langs.items())),
        "by_year": dict(sorted(years.items())),
    }
