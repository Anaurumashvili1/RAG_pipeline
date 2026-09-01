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
    is_junk_url,
    is_latin_script,
    repair_letter_spacing,
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
    skipped_junk = 0
    repaired_spacing = 0
    canonical_of: dict[str, str] = {}      # doc_id -> declared canonical URL

    # Normalise once: YAML may give ["IT", "en-GB"], and a raw list membership
    # test against that silently drops the entire corpus.
    allowed = (
        {str(x).strip().lower().split("-")[0] for x in keep_languages}
        if keep_languages
        else None
    )

    for raw in iter_jsonl(path):
        url = (raw.get("url") or "").strip()
        title = clean_text(raw.get("title"))
        # Repair before cleaning: letter-spaced PDFs encode word boundaries as
        # double spaces, and clean_text collapses runs of spaces.
        body = repair_letter_spacing(raw.get("text"))
        if body is not raw.get("text"):
            repaired_spacing += 1
        # keep_breaks: paragraph structure is what SentenceSplitter splits on.
        text = clean_text(body, keep_breaks=True)

        if not url:
            continue

        # AppleDouble stubs: not documents at all. Checked before the length
        # filter because it is a cheap URL test, and because counting them
        # separately from "too short" is what makes the log honest.
        if is_junk_url(url):
            skipped_junk += 1
            continue

        if len(text) < min_chars:
            continue

        # Quality flags decided during crawling. Cheaper and more accurate than
        # re-deciding here, since the crawler saw the raw HTML and we do not.
        #
        # duplicate_of is the exception, and it is a trap. It records a
        # *canonical URL declaration* - from <link rel="canonical"> or a
        # normalisation rule - not an observed content duplicate. Measured on
        # this corpus: 2,426 records carry it, NONE shares content_sha256 with
        # any other document, and 2,102 (87%) name a canonical that was never
        # fetched. Treating it as "a copy exists elsewhere" silently deleted
        # 2,102 documents whose text is in the corpus exactly once.
        #
        # So it is deferred: the decision needs the full URL set, and is made
        # after the loop.
        canonical = raw.get("duplicate_of") if drop_duplicates else None
        if drop_low_content and raw.get("low_content"):
            continue
        if drop_boilerplate and raw.get("boilerplate"):
            continue

        # Language scope. The crawler's `lang` is authoritative - it read
        # <html lang> - and it is the only place a non-it/en language is
        # recorded at all: detect_language() knows only it/en and defaults to
        # 'it', so an out-of-scope page silently becomes Italian and then
        # competes for Italian queries.
        if allowed is not None:
            declared = (raw.get("lang") or "").strip().lower().split("-")[0]
            if declared:
                if declared not in allowed:
                    skipped_lang[declared] = skipped_lang.get(declared, 0) + 1
                    continue
            elif not is_latin_script(text):
                # No declared language and the text is not Latin script, so
                # detect_language() would default it to 'it'. This is the case
                # that let Chinese flyer PDFs into the corpus as Italian.
                skipped_lang["non-latin"] = skipped_lang.get("non-latin", 0) + 1
                continue

        did = doc_id_from_url(url)
        if did in seen_ids:          # same URL crawled twice
            continue
        seen_ids.add(did)

        if canonical:
            canonical_of[did] = canonical

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

    # Now that every surviving URL is known, resolve the canonical claims. Drop
    # a document only when the page it points at is actually in the corpus -
    # otherwise the flag would delete content that exists exactly once.
    if canonical_of:
        def _norm(u: str) -> str:
            return (u or "").strip().rstrip("/").replace("https://", "").replace("http://", "")

        present = {_norm(d.url) for d in docs}
        redundant = {did for did, can in canonical_of.items() if _norm(can) in present}
        rescued = len(canonical_of) - len(redundant)
        if redundant:
            docs = [d for d in docs if d.doc_id not in redundant]
        print(f"[data] duplicate_of: dropped {len(redundant)} whose canonical is "
              f"present, kept {rescued} whose canonical was never fetched")

    if repaired_spacing:
        print(f"[data] repaired letter-spacing in {repaired_spacing} PDFs")
    if skipped_junk:
        print(f"[data] skipped {skipped_junk} AppleDouble '._' stubs")
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
