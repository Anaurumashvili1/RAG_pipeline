#!/usr/bin/env python3
"""OCR the scanned PDFs the crawler could not extract text from.

The crawl kept extracted text and discarded the PDF bytes, so this runs in two
phases:

    fetch   re-download each PDF from unitn.it, politely, into a local cache
    ocr     rasterise with pdftoppm, run tesseract, write the text back out

Both phases are resumable: rerunning skips anything already done, so an
interrupted job costs only the document it was working on.

    python scripts/ocr_pending.py --limit 20          # sample first, always
    python scripts/ocr_pending.py                     # full 873
    python scripts/ocr_pending.py --fetch-only        # download overnight, OCR later
    python scripts/ocr_pending.py --ocr-only --workers 8

Requires: tesseract with the 'ita' language pack, and pdftoppm (poppler-utils).
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------
# Defaults. Overridable from the command line.
# --------------------------------------------------------------------------

PENDING = Path.home() / "scraper" / "pending_ocr.jsonl"
CACHE = Path.home() / "scraper" / "pdf_cache"
OUTPUT = Path.home() / "scraper" / "ocr_output.jsonl"

USER_AGENT = "UniTn-RAG-research/1.0 (+anaurumashvili@gmail.com)"
FETCH_DELAY = 1.0        # seconds between downloads - be a good citizen
FETCH_TIMEOUT = 120
DPI = 300                # 300 is the sweet spot for tesseract; 600 is slower, not better
OCR_LANG = "ita+eng"     # corpus is Italian with English pages mixed in

# Below this, the "text" is almost certainly noise from a map or a photograph
# rather than a failed scan of real prose.
MIN_CHARS_PER_PAGE = 100


def log(msg: str) -> None:
    print(f"[ocr] {msg}", flush=True)


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------


def cache_path(url: str, cache: Path | None = None) -> Path:
    """Stable filename per URL. Content-addressing would need the bytes first."""
    return (cache or CACHE) / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.pdf"


def fetch_pdf(url: str, dest: Path, timeout: int = FETCH_TIMEOUT) -> tuple[bool, str]:
    """Download one PDF. Returns (ok, message)."""
    if dest.exists() and dest.stat().st_size > 0:
        return True, "cached"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ctype = (r.headers.get("Content-Type") or "").lower()
            data = r.read()
    except urllib.error.HTTPError as e:
        return False, f"http {e.code}"
    except Exception as e:  # noqa: BLE001
        return False, f"error {type(e).__name__}: {e}"

    if not data:
        return False, "empty response"
    if not data.startswith(b"%PDF") and "pdf" not in ctype:
        return False, f"not a pdf (content-type {ctype!r})"

    tmp.write_bytes(data)
    tmp.rename(dest)                      # atomic: a .pdf on disk is always complete
    return True, f"{len(data) / 1024:.0f} KB"


def verify_sha256(path: Path, expected: str | None) -> bool | None:
    """True/False if we can check, None if the crawl recorded no hash."""
    if not expected or expected == "None":
        return None
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h == expected


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------


def ocr_pdf(pdf: Path, dpi: int = DPI, lang: str = OCR_LANG) -> tuple[str, int]:
    """Rasterise then OCR, one page at a time. Returns (text, pages_done).

    Page-at-a-time keeps peak disk use to a single PNG. A 112-page document at
    300 DPI would otherwise put ~2 GB of images in the temp directory at once.
    """
    pages: list[str] = []

    with tempfile.TemporaryDirectory(prefix="ocr_") as td:
        tmp = Path(td)
        n = page_count(pdf)
        if n == 0:
            return "", 0

        for i in range(1, n + 1):
            stem = tmp / f"p{i}"
            r = subprocess.run(
                ["pdftoppm", "-f", str(i), "-l", str(i),
                 "-r", str(dpi), "-png", "-singlefile", str(pdf), str(stem)],
                capture_output=True, timeout=300,
            )
            png = stem.with_suffix(".png")
            if r.returncode != 0 or not png.exists():
                pages.append("")
                continue

            r = subprocess.run(
                ["tesseract", str(png), "stdout", "-l", lang, "--psm", "1"],
                capture_output=True, timeout=600,
            )
            pages.append(r.stdout.decode("utf-8", errors="replace") if r.returncode == 0 else "")
            png.unlink(missing_ok=True)

    return "\n\n".join(p.strip() for p in pages if p.strip()), len(pages)


def page_count(pdf: Path) -> int:
    """Page count via pdfinfo, falling back to a byte scan if poppler is partial."""
    try:
        r = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, timeout=60)
        for line in r.stdout.decode("utf-8", errors="replace").splitlines():
            if line.lower().startswith("pages:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:  # noqa: BLE001
        pass
    try:
        return max(1, pdf.read_bytes().count(b"/Type /Page") - 1)
    except Exception:  # noqa: BLE001
        return 0


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def load_pending(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # Smallest first: fastest feedback, and a crash costs the least work.
    rows.sort(key=lambda r: int(r.get("page_count") or 0))
    return rows[:limit] if limit else rows


def done_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.add(json.loads(line)["url"])
                except Exception:  # noqa: BLE001
                    continue
    return out


def process(row: dict, dpi: int, lang: str, cache: Path) -> dict:
    """OCR one already-cached document. Runs in a worker process.

    The cache directory is passed in rather than read from the module global:
    a spawned worker re-imports this file and would see the default, not the
    value main() set from --cache.
    """
    url = row["url"]
    pdf = cache_path(url, cache)
    t0 = time.time()

    if not pdf.exists():
        return {"url": url, "ok": False, "error": "not fetched"}

    try:
        text, pages = ocr_pdf(pdf, dpi=dpi, lang=lang)
    except subprocess.TimeoutExpired:
        return {"url": url, "ok": False, "error": "timeout"}
    except Exception as e:  # noqa: BLE001
        return {"url": url, "ok": False, "error": f"{type(e).__name__}: {e}"}

    per_page = len(text) / pages if pages else 0
    return {
        "url": url,
        "ok": True,
        "title": row.get("title"),
        "ocr_text": text,
        "ocr_chars": len(text),
        "ocr_pages": pages,
        "chars_per_page": round(per_page, 1),
        # Flag rather than drop: a bus map legitimately has almost no text, and
        # that is a fact about the document, not a failure of the OCR.
        "ocr_suspect": per_page < MIN_CHARS_PER_PAGE,
        "ocr_engine": "tesseract",
        "ocr_lang": lang,
        "ocr_dpi": dpi,
        "ocr_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    global CACHE

    ap = argparse.ArgumentParser()
    ap.add_argument("--pending", type=Path, default=PENDING)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--limit", type=int, help="process only the first N (smallest first)")
    ap.add_argument("--workers", type=int, default=4, help="parallel OCR processes")
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--lang", default=OCR_LANG)
    ap.add_argument("--delay", type=float, default=FETCH_DELAY)
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--ocr-only", action="store_true")
    args = ap.parse_args()

    CACHE = args.cache
    CACHE.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for tool in ("pdftoppm", "tesseract"):
        if not shutil.which(tool):
            log(f"FAIL {tool} not found on PATH")
            return 1

    rows = load_pending(args.pending, args.limit)
    log(f"{len(rows)} documents pending, {sum(int(r.get('page_count') or 0) for r in rows)} pages")

    # ---------------- fetch ----------------
    if not args.ocr_only:
        ok = failed = cached = mismatched = 0
        for i, row in enumerate(rows, 1):
            url = row["url"]
            dest = cache_path(url)
            was_cached = dest.exists()
            good, msg = fetch_pdf(url, dest)

            if good:
                if was_cached:
                    cached += 1
                else:
                    ok += 1
                    if verify_sha256(dest, row.get("content_sha256")) is False:
                        mismatched += 1
                        log(f"  sha mismatch (page changed since crawl): {url}")
                    time.sleep(args.delay)
            else:
                failed += 1
                log(f"  fetch failed [{msg}]: {url}")

            if i % 50 == 0:
                log(f"fetch {i}/{len(rows)}  new={ok} cached={cached} failed={failed}")

        log(f"fetch done: {ok} new, {cached} cached, {failed} failed, {mismatched} changed since crawl")
        if args.fetch_only:
            return 0

    # ---------------- ocr ----------------
    already = done_urls(args.output)
    todo = [r for r in rows if r["url"] not in already and cache_path(r["url"], CACHE).exists()]
    log(f"{len(already)} already OCR'd, {len(todo)} to do, {args.workers} workers")

    t0 = time.time()
    done = suspect = failed = 0

    with open(args.output, "a", encoding="utf-8") as out:
        with futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(process, r, args.dpi, args.lang, CACHE): r for r in todo}
            for n, fut in enumerate(futures.as_completed(jobs), 1):
                res = fut.result()
                if res.get("ok"):
                    done += 1
                    suspect += bool(res.get("ocr_suspect"))
                    out.write(json.dumps(res, ensure_ascii=False) + "\n")
                    out.flush()          # survive a kill without losing the batch
                else:
                    failed += 1
                    log(f"  {res.get('error')}: {res['url']}")

                if n % 20 == 0:
                    rate = n / max(1e-9, time.time() - t0)
                    left = (len(todo) - n) / rate / 60
                    log(f"ocr {n}/{len(todo)}  ok={done} suspect={suspect} failed={failed}"
                        f"  ~{left:.0f} min left")

    log(f"done: {done} OCR'd ({suspect} suspect, under {MIN_CHARS_PER_PAGE} chars/page), "
        f"{failed} failed, {(time.time() - t0) / 60:.0f} min")
    log(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
