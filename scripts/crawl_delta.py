#!/usr/bin/env python3
"""Crawl only the hosts that were missing from the main crawl.

Copy this next to scraper.py and run it from there:

    cd ~/scraper
    python3 crawl_delta.py                 # the four missing hosts
    python3 crawl_delta.py --hosts www.dicam.unitn.it

It imports the existing spider and overrides four module globals rather than
editing scraper.py, so the main crawl configuration is untouched and this file
can be deleted afterwards.

Isolation matters here:
  * a separate JOBDIR   - reusing unitn_jobdir makes Scrapy resume a *finished*
                          crawl and immediately exit
  * a separate state DB - the delta cannot corrupt the state of a 201k-page run
  * a separate output   - unitn_crawl_delta.jsonl, never appended to
                          unitn_crawl.jsonl

The trade-off of a fresh state DB: cross-URL duplicate detection
(``first_url_for_text``) only sees pages from this run, so ``duplicate_of``
will not catch a delta page duplicating something in the main corpus.
clean_corpus.py deduplicates at merge time, so this costs nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import scraper


# Missing from ALLOWED_HOSTS_RAW in the main crawl. Found by diffing the corpus
# against an older crawl: 29 of 42 evaluation gold URLs were never fetched.
MISSING_HOSTS = [
    "www.giurisprudenza.unitn.it",   # Faculty of Law
    "www.dicam.unitn.it",            # Civil, Environmental and Mechanical Eng
    "www.dii.unitn.it",              # Industrial Engineering
    "nse.physics.unitn.it",          # Nonlinear Systems and Electronics lab
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", nargs="+", default=MISSING_HOSTS)
    ap.add_argument("--out", default="unitn_crawl_delta.jsonl")
    ap.add_argument("--jobdir", default="unitn_jobdir_delta")
    ap.add_argument("--state", default="unitn_state_delta.sqlite3")
    ap.add_argument("--max-pages", type=int, default=50_000)
    args = ap.parse_args()

    for p in (args.jobdir, args.state):
        if Path(p).exists():
            print(f"[delta] {p} already exists - the crawl will RESUME, not restart.")
            print(f"[delta] Delete it first for a clean run:  rm -rf {p}")

    # normalize_host() strips a leading 'www.', and ALLOWED_HOSTS holds the
    # normalized form. is_valid_target() normalizes the candidate too, so both
    # www and non-www resolve correctly against this set.
    hosts = {scraper.normalize_host(h) for h in args.hosts}
    scraper.ALLOWED_HOSTS = frozenset(hosts)

    # VersioningPipeline reads this global in open_spider().
    scraper.OUTPUT_JSONL = args.out

    # JOBDIR lives in custom_settings, evaluated at class-definition time.
    scraper.UnitnSpider.custom_settings = {
        **scraper.UnitnSpider.custom_settings,
        "JOBDIR": args.jobdir,
        "CLOSESPIDER_PAGECOUNT": args.max_pages,
    }

    # build_seeds() derives one seed per allowed host, so the four hosts above
    # become the four seeds. Both language trees are seeded explicitly: these
    # are Drupal sites where /en/ is a separate tree, and English departmental
    # pages are exactly what went missing last time.
    # build_seeds() emits the *normalized* host, i.e. www-stripped, and several
    # of these Drupal sites only answer on www. Seed both forms, plus the /en
    # tree explicitly - English departmental pages are precisely what went
    # missing last time, and on these sites /en/ is a separate tree.
    extra = []
    for h in args.hosts:
        extra += [f"https://{h}/", f"https://{h}/en", f"https://{h}/en/"]
    seeds = scraper.build_seeds(extra=extra)

    print(f"[delta] hosts  : {sorted(hosts)}")
    print(f"[delta] seeds  : {len(seeds)}")
    for s in seeds:
        print(f"          {s}")
    print(f"[delta] output : {args.out}")
    print(f"[delta] jobdir : {args.jobdir}")
    print(f"[delta] state  : {args.state}")
    print()

    from scrapy.crawler import CrawlerProcess
    from scrapy.settings import Settings

    settings = Settings()
    settings.set("TWISTED_REACTOR",
                 "twisted.internet.asyncioreactor.AsyncioSelectorReactor")
    process = CrawlerProcess(settings=settings)
    process.crawl(scraper.UnitnSpider, seeds=seeds, state_path=args.state)
    process.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
