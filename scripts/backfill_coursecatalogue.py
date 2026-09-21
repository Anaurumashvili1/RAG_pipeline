#!/usr/bin/env python3
"""Backfill thin/empty 2026/27 course-catalogue syllabi with 2025/26 content.

Context (2026-09-08): scrape_coursecatalogue.py --anno 2026 was run at the
very start of the 2026/27 academic year. Second-semester courses (Feb-Jun
2027) frequently don't have syllabus text published yet -- the unit exists
in the teaching plan (corso-offerta) but /insegnamento's testiTotali is
empty. That's a real, temporary gap, not a scraper bug.

This is a deliberate, known-risky stopgap: showing a student last year's
syllabus can be actively wrong (different professor, different exam
format, different content). Two things keep that risk contained:

1. Matching is exact and conservative: (corso_cod, normalized title, lang).
   No fuzzy matching, no falling back to "close enough". An unmatched
   record is left thin and reported -- never guessed at.

2. A backfilled record's `academic_year` is overwritten to "2025/2026"
   (the content's true origin), not left as "2026/2027". That is the same
   principle as claude/eval-set-document-selection's sibling doc on
   effective-year evidence: "a year is only as good as where it came
   from." unitn_rag.text.resolve_effective_year() will end-year that to
   2026 same as native 2026/27 content would resolve to 2027 -- i.e. the
   backfilled record ranks one year staler via the 1/(1+age) decay, so it
   can still surface but never outranks genuinely current text. A `note`
   flag marks it for later auditing/replacement once the real 2026/27
   text is published and this script is re-run.

Records whose emptiness is structural rather than "not written yet" are
skipped outright, not backfilled: an aggregate insegnamento that has
frazioni (split-section partitions) is *expected* to have empty
testiTotali -- see fetch_fraction_syllabus() in scrape_coursecatalogue.py.
Backfilling it from last year would resurrect the exact misdiagnosis
already made and corrected once this month for "Istituzioni di diritto
romano" / "Sistemi giuridici comparati" (they looked content-edited/
removed; they were actually frazioni-split all along).

Never edits either input in place.

    python3 scripts/backfill_coursecatalogue.py            # report only
    python3 scripts/backfill_coursecatalogue.py --write    # produce coursecatalogue_backfilled.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_CURRENT = Path("coursecatalogue.jsonl")        # anno=2026 output
DEFAULT_PRIOR = Path("coursecatalogue_2025.jsonl")      # anno=2025 output
DEFAULT_OUT = Path("coursecatalogue_backfilled.jsonl")
THIN_THRESHOLD = 300  # text_len below this counts as "no real content yet"
PRIOR_ACADEMIC_YEAR = "2025/2026"
BACKFILL_NOTE = "coursecatalogue_backfill_prior_year"


def normalize_title(title: str) -> str:
    """Course title without the trailing ' - PROGRAMME (Faculty)' suffix.

    'Istituzioni di diritto romano/1 (Iniziali cognome A-E) - GIURISPRUDENZA
    (Facolta di Giurisprudenza)' -> 'istituzioni di diritto romano/1
    (iniziali cognome a-e)'

    The suffix (programme name + faculty) can legitimately differ between
    the two years for the same course (e.g. a programme renamed, or the
    course moved faculty) while the course itself is unchanged, so it is
    stripped before comparing rather than matched.
    """
    head = title.split(" - ")[0]
    return re.sub(r"\s+", " ", head).strip().lower()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def has_frazioni(unit_cod: str, records: list[dict]) -> bool:
    """True if some record in *records* is a fraction of *unit_cod*."""
    return any(r.get("coursecatalogue_parent_unit_cod") == unit_cod for r in records)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    ap.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--thin-threshold", type=int, default=THIN_THRESHOLD)
    ap.add_argument("--write", action="store_true",
                     help="actually write the output; without this it only reports")
    args = ap.parse_args()

    current = load_jsonl(args.current)
    prior = load_jsonl(args.prior)

    # Index last year's syllabus records by (corso_cod, normalized title, lang).
    prior_index: dict[tuple, dict] = {}
    for r in prior:
        if r.get("doc_type") != "syllabus":
            continue
        key = (r.get("coursecatalogue_corso_cod"), normalize_title(r.get("title", "")), r.get("lang"))
        prior_index[key] = r

    n_thin = 0
    n_structural_skip = 0
    n_backfilled = 0
    n_unmatched = 0
    out_records = []

    for r in current:
        if r.get("doc_type") != "syllabus" or (r.get("text_len") or 0) >= args.thin_threshold:
            out_records.append(r)
            continue

        n_thin += 1
        unit_cod = r.get("coursecatalogue_unit_cod")
        if not r.get("coursecatalogue_partition") and has_frazioni(unit_cod, current):
            # Aggregate record for a split course -- empty by design, not a gap.
            n_structural_skip += 1
            out_records.append(r)
            continue

        key = (r.get("coursecatalogue_corso_cod"), normalize_title(r.get("title", "")), r.get("lang"))
        match = prior_index.get(key)
        if match is None or (match.get("text_len") or 0) < args.thin_threshold:
            n_unmatched += 1
            out_records.append(r)
            continue

        patched = dict(r)
        patched["title"] = match["title"]
        patched["text"] = match["text"]
        patched["text_len"] = match["text_len"]
        patched["text_sha256"] = match["text_sha256"]
        patched["content_sha256"] = match["content_sha256"]
        patched["academic_year"] = PRIOR_ACADEMIC_YEAR
        patched["note"] = BACKFILL_NOTE
        out_records.append(patched)
        n_backfilled += 1

    print(f"[backfill] {len(current):,} current records, {n_thin} thin (< {args.thin_threshold} chars)")
    print(f"[backfill]   {n_structural_skip} skipped (frazioni-split, empty by design)")
    print(f"[backfill]   {n_backfilled} backfilled from {PRIOR_ACADEMIC_YEAR}")
    print(f"[backfill]   {n_unmatched} still thin (no 2025/26 match, or 2025/26 was thin too)")

    if not args.write:
        print("[backfill] dry run - nothing written. Re-run with --write.")
        return 0

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[backfill] wrote {args.out} ({len(out_records):,} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
