#!/usr/bin/env python3
"""Fold scrape_coursecatalogue.py's output into the corpus.

Unlike merge_ocr.py this never touches an existing record - every URL out of
the course catalogue is new to the corpus (there is nothing there today for
unitn.coursecatalogue.cineca.it besides the single 4-character SPA-shell
record), so this is a concatenation with a URL-collision safety check, not a
field-by-field merge.

    python3 scripts/merge_coursecatalogue.py             # report only, writes nothing
    python3 scripts/merge_coursecatalogue.py --write     # produce dataset.v3.jsonl

Never edits either input in place - a bad merge costs a rerun, not the corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_CORPUS = Path("dataset.v2.jsonl")
DEFAULT_NEW = Path("coursecatalogue.jsonl")
DEFAULT_OUT = Path("dataset.v3.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--new", type=Path, default=DEFAULT_NEW)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--write", action="store_true",
                    help="actually write the output; without this it only reports")
    args = ap.parse_args()

    existing_urls: set[str] = set()
    n_corpus = 0
    with open(args.corpus, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_corpus += 1
            existing_urls.add(json.loads(line).get("url"))

    new_records: list[dict] = []
    n_new_total = 0
    n_empty_text = 0
    n_collisions = 0
    doc_types: dict[str, int] = {}
    with open(args.new, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_new_total += 1
            r = json.loads(line)
            if not (r.get("text") or "").strip():
                n_empty_text += 1
                continue
            if r.get("url") in existing_urls:
                n_collisions += 1
                print(f"[merge] WARNING url already in corpus, skipping: {r['url']}")
                continue
            doc_types[r.get("doc_type")] = doc_types.get(r.get("doc_type"), 0) + 1
            new_records.append(r)

    print(f"[merge] corpus:   {n_corpus:,} existing records")
    print(f"[merge] new file: {n_new_total:,} records, {n_empty_text} empty (skipped), "
          f"{n_collisions} URL collisions (skipped), {len(new_records):,} to add")
    for dt, n in sorted(doc_types.items(), key=lambda kv: -kv[1]):
        print(f"[merge]    {dt}: {n:,}")

    if not args.write:
        print("[merge] dry run - nothing written. Re-run with --write.")
        return 0

    with open(args.out, "w", encoding="utf-8") as out_f, \
            open(args.corpus, encoding="utf-8") as corpus_f:
        for line in corpus_f:
            out_f.write(line if line.endswith("\n") else line + "\n")
        for r in new_records:
            out_f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[merge] wrote {args.out} ({n_corpus + len(new_records):,} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
