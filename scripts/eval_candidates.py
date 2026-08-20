#!/usr/bin/env python3
"""Build a worksheet of candidate documents to write evaluation questions about.

This script never writes a question. It selects *what you write about*, which is
where code actually helps: left to intuition you will write about whatever you
happened to look at, and the set will be unrepresentative in ways you cannot
see.

    python3 scripts/eval_candidates.py --corpus dataset.ocr.jsonl --out eval_candidates

Produces two files:

    <out>_faq.csv          real questions harvested from FAQ pages, with source
    <out>_targets.csv      stratified documents to write questions about

Categories in the targets file map onto specific design decisions, so the
finished eval set tests the architecture rather than producing a generic score:

    bilingual   - page exists in IT and EN -> tests doc_group dedup
    temporal    - same page group across years -> tests the freshness decay
    ocr         - text exists only because of OCR -> tests the recovery stage
    admissions  - high-traffic topic, precise answers
    course      - programme detail
    scholarship - deadline- and eligibility-heavy, so staleness matters
    trap        - lexical near-neighbours (e.g. MASTER-M3 vs master's degree)
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
from pathlib import Path
from urllib.parse import urlsplit

# A FAQ heading: short, ends in a question mark, not a whole paragraph.
_Q_RE = re.compile(r"^[#\s*]*(.{10,140}\?)\s*$", re.MULTILINE)

# Terms whose lexical neighbours mean something else entirely. MASTER-M3 is a
# metamaterials project, not a master's degree; "dottorato" vs "dottore";
# "magistrale" vs "magistrale a ciclo unico".
_TRAP_TERMS = ("master", "magistrale", "dottorato", "tirocinio", "borsa")


# Pages that exist but cannot answer a question: staff directories, publication
# lists, person records. They have text and metadata and pass every length
# filter, which is exactly why they have to be excluded explicitly.
_DEAD_PATHS = ("/du/", "/persona/", "/persone/", "/strutturaaccademica/",
               "/pubblicazioni/", "/publications/", "/rubrica")
_DEAD_TYPES = ("people",)


def is_answerable(r: dict) -> bool:
    """Could a human write a factual question this page answers?"""
    if r.get("doc_type") in _DEAD_TYPES:
        return False
    path = urlsplit(r.get("url", "")).path.lower()
    if any(d in path for d in _DEAD_PATHS):
        return False
    return len(r.get("text") or "") > 800


def norm_group(url: str) -> str:
    """Language-stripped URL key, for finding translation pairs."""
    p = urlsplit(url)
    host = p.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    path = re.sub(r"/(en|it)(/|$)", "/", p.path)
    path = re.sub(r"[._-](en|it)(\.\w+)$", r"\2", path)
    return f"{host}{path.rstrip('/')}"


def load(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def harvest_faq(rows: list[dict]) -> list[dict]:
    """Real questions written by UniTn staff, with the page that answers them."""
    out = []
    for r in rows:
        url = r.get("url", "")
        title = r.get("title") or ""
        if "faq" not in url.lower() and "faq" not in title.lower():
            continue
        text = r.get("text") or ""
        seen = set()
        for m in _Q_RE.finditer(text):
            q = " ".join(m.group(1).split())
            if q.lower() in seen:
                continue
            seen.add(q.lower())
            out.append({
                "harvested_question": q,
                "rewrite_as_a_student_would_ask": "",
                "expected_url": url,
                "lang": r.get("lang") or "",
                "page_title": title,
                "keep": "",
            })
    return out


def pick_targets(rows: list[dict], per_category: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_url = {r.get("url", ""): r for r in rows}
    out: list[dict] = []

    def emit(category: str, picks: list[dict], note: str = "") -> None:
        for r in picks:
            out.append({
                "category": category,
                "question": "",
                "expected_url": r.get("url", ""),
                "lang": r.get("lang") or "",
                "year": r.get("effective_year") or "",
                "doc_type": r.get("doc_type") or "",
                "page_title": (r.get("title") or "")[:90],
                "note": note,
            })

    def sample(pool: list[dict], n: int) -> list[dict]:
        return rng.sample(pool, min(n, len(pool)))

    # --- bilingual: the same page in both languages -----------------------
    groups: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for r in rows:
        lang = (r.get("lang") or "").lower()
        if lang in ("it", "en"):
            groups[norm_group(r.get("url", ""))][lang] = r
    pairs = [g for g in groups.values() if len(g) == 2]
    emit("bilingual", [g["it"] for g in sample(pairs, per_category)],
         "write the question in BOTH languages; both versions should not occupy two slots")

    # --- temporal: one page group appearing across several years ----------
    years: dict[str, set] = collections.defaultdict(set)
    holder: dict[str, dict] = {}
    for r in rows:
        y = r.get("effective_year")
        if not isinstance(y, int):
            continue
        k = re.sub(r"20\d{2}", "YYYY", norm_group(r.get("url", "")))
        years[k].add(y)
        if y == max(years[k]):
            holder[k] = r
    multi = [holder[k] for k, ys in years.items() if len(ys) >= 3 and k in holder]
    emit("temporal", sample(multi, per_category),
         "an outdated sibling exists; the current one must win")

    # --- ocr: text that exists only because of the OCR stage --------------
    ocr = [r for r in rows if r.get("extractor") == "tesseract"
           and not r.get("ocr_suspect") and len(r.get("text") or "") > 3000]
    emit("ocr", sample(ocr, per_category),
         "answer exists ONLY in recovered text - proves OCR changed retrieval")

    # --- topical strata ---------------------------------------------------
    for dt in ("admissions", "course", "scholarship"):
        pool = [r for r in rows if r.get("doc_type") == dt
                and len(r.get("text") or "") > 800]
        emit(dt, sample(pool, per_category))

    # --- traps: lexical near-neighbours -----------------------------------
    traps = [r for r in rows
             if any(t in (r.get("title") or "").lower() for t in _TRAP_TERMS)
             and r.get("doc_type") not in ("admissions", "course")]
    emit("trap", sample(traps, per_category),
         "term collides with a different concept; a confident wrong answer is the risk")

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("dataset.ocr.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("eval_candidates"))
    ap.add_argument("--per-category", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260820,
                    help="fixed so the selection is reproducible and citable")
    args = ap.parse_args()

    rows = load(args.corpus)
    print(f"[eval] corpus: {len(rows):,} documents")

    faq = harvest_faq(rows)
    faq_path = args.out.with_name(args.out.name + "_faq.csv")
    with open(faq_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(faq[0].keys()) if faq else
                           ["harvested_question", "rewrite_as_a_student_would_ask",
                            "expected_url", "lang", "page_title", "keep"])
        w.writeheader()
        w.writerows(faq)
    pages = len({r["expected_url"] for r in faq})
    print(f"[eval] harvested {len(faq)} real questions from {pages} FAQ pages -> {faq_path}")

    targets = pick_targets(rows, args.per_category, args.seed)
    tgt_path = args.out.with_name(args.out.name + "_targets.csv")
    with open(tgt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(targets[0].keys()))
        w.writeheader()
        w.writerows(targets)

    counts = collections.Counter(t["category"] for t in targets)
    print(f"[eval] {len(targets)} target documents -> {tgt_path}")
    for k, v in sorted(counts.items()):
        print(f"         {k:<12} {v}")

    print()
    print("Next: fill the blank question columns by hand. Rewrite harvested FAQ")
    print("questions in your own words - verbatim ones share vocabulary with the")
    print("page and measure paraphrase matching rather than retrieval.")
    print("Add ~10 out-of-domain questions the system should refuse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
