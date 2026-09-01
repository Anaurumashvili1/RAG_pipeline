#!/usr/bin/env python3
"""Resolve stale target_urls in the evaluation set against the current corpus.

The old evaluation set records URLs from an earlier version of unitn.it. Since
then Drupal aliases changed and programme slugs were renamed:

    /en/867/nanoscience                     ->  /node/867
    /en/computer-science/graduation/...     ->  /en/computer-science-master/...
    webmagazine.unitn.it/en/evento/...      ->  eventi.unitn.it/it/...

`hit_at_k` compares URLs exactly, so a *correct* retrieval scores 0 against a
stale target. 22 of 25 apparently-missing gold documents are in the corpus
under a different URL.

    python3 scripts/fix_eval_urls.py --corpus dataset.v2.jsonl        # report
    python3 scripts/fix_eval_urls.py --corpus dataset.v2.jsonl --write

Matching is by exact normalised title, and only when the title maps to exactly
one corpus document. Ambiguous and unmatched entries are reported for a human
to decide - a generic title like "Admission and enrollment for Europeans and
equivalents" exists on every degree programme, and guessing there would
silently point a question at the wrong programme.

Output adds three fields per entry, and never deletes the original:

    target_url            resolved, or unchanged
    original_target_url   what the set said before
    answerable            is the gold document in the corpus at all
    url_status            matched | already-present | ambiguous | absent
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from urllib.parse import urlsplit


def norm_url(u: str) -> str:
    return (u or "").strip().rstrip("/").replace("https://", "").replace("http://", "")


def norm_title(t: str) -> str:
    """Title before the site-name suffix, whitespace-collapsed, lowercased.

    UniTn titles are 'Page name | Site name', and the site name differs between
    the Italian and English renderings of one page.
    """
    head = (t or "").split("|")[0]
    return " ".join(head.lower().split())


def load_corpus(path: Path):
    by_url: set[str] = set()
    by_title: dict[str, list[dict]] = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_url.add(norm_url(r.get("url", "")))
            t = norm_title(r.get("title"))
            if t:
                by_title[t].append(r)
    return by_url, by_title


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=Path("data/evaluation_set.json"))
    ap.add_argument("--corpus", type=Path, default=Path("dataset.v2.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/evaluation_set.resolved.json"))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    by_url, by_title = load_corpus(args.corpus)
    ev = json.loads(args.eval.read_text(encoding="utf-8"))
    print(f"[eval-urls] {len(ev)} questions · {len(by_url):,} corpus urls")
    print()

    counts = collections.Counter()
    out = []

    for i, e in enumerate(ev):
        entry = dict(e)
        entry["original_target_url"] = e.get("target_url", "")
        gold = norm_url(e.get("target_url", ""))

        if gold in by_url:
            entry["url_status"] = "already-present"
            entry["answerable"] = True
        else:
            matches = by_title.get(norm_title(e.get("title")), [])
            urls = sorted({m["url"] for m in matches})

            # Prefer candidates on the same host as the original URL. Page
            # titles repeat across the university - every degree programme has
            # an "Internship" page - but the host is a strong discriminator
            # that survives Drupal alias changes and slug renames.
            if len(urls) > 1:
                host = urlsplit(e.get("target_url", "")).netloc.lower().lstrip("www.")
                same_host = [u for u in urls
                             if urlsplit(u).netloc.lower().lstrip("www.") == host]
                if len(same_host) == 1:
                    urls = same_host
                    entry["resolved_by"] = "title+host"
                elif same_host:
                    urls = same_host          # narrowed, still ambiguous

            if len(urls) == 1:
                entry["target_url"] = urls[0]
                entry["url_status"] = "matched"
                entry["answerable"] = True
                print(f"[{i:>2}] MATCHED   {(e.get('title') or '')[:44]}")
                print(f"       was: {e.get('target_url','')[:96]}")
                print(f"       now: {urls[0][:96]}")
            elif len(urls) > 1:
                # Ambiguous is NOT unanswerable: the document is in the corpus,
                # we just cannot tell which candidate is the right one. Left
                # answerable=None so it is counted as "needs a human" rather
                # than silently inflating the abstention set.
                entry["url_status"] = "ambiguous"
                entry["answerable"] = None
                entry["candidate_urls"] = urls[:8]
                print(f"[{i:>2}] AMBIGUOUS {(e.get('title') or '')[:44]}  ({len(urls)} candidates)")
                for u in urls[:4]:
                    print(f"           {u[:92]}")
            else:
                entry["url_status"] = "absent"
                entry["answerable"] = False
                print(f"[{i:>2}] ABSENT    {(e.get('title') or '')[:44]}")
                print(f"       {e.get('target_url','')[:96]}")

        counts[entry["url_status"]] += 1
        out.append(entry)

    print()
    for k in ("already-present", "matched", "ambiguous", "absent"):
        print(f"  {k:<17} {counts[k]}")
    answerable = counts["already-present"] + counts["matched"]
    print()
    print(f"  {'-> answerable now':<21} {answerable}")
    print(f"  {'-> needs a human':<21} {counts['ambiguous']}  (in the corpus, pick the right URL)")
    print(f"  {'-> abstention set':<21} {counts['absent']}  (genuinely not in the corpus)")
    print(f"  {'-> potential total':<21} {answerable + counts['ambiguous']} of {len(ev)}")

    if args.write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[eval-urls] wrote {args.out}")
    else:
        print("\n[eval-urls] dry run - nothing written. Re-run with --write.")

    if counts["ambiguous"]:
        print("\nAMBIGUOUS entries need a human: the title exists on several pages")
        print("(every degree programme has an 'Admission and enrollment' page).")
        print("Pick the right URL by hand, or drop the question.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
