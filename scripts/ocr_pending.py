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
import re
import ssl
import urllib.error
import urllib.parse
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

# Quality thresholds. Chars-per-page alone is not enough: OCR run over a graphic
# poster hallucinates letterforms and yields *more* characters than real prose
# ("Cee] = ni Tri + d a rai = E n ki i m n n a i |g" scored 2,720 chars/page).
# Measuring whether the output behaves like language catches that; counting it
# does not.
# Share of tokens that are 4+ characters. Measured on the first sample:
#   real prose            83%      OCR'd event poster    53-59%
#   hallucinated noise    14%
# A stopword-frequency check was tried first and failed - noise contains enough
# stray 'a', 'i', 'e', 'di' to score 0.121, indistinguishable from real text.
# Word *length* separates cleanly because noise fragments into single letters.
MIN_LONG_TOKEN_RATIO = 0.35

_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")


def text_quality(text: str) -> dict:
    """Does this read like language, or like OCR noise over a graphic?

    Tesseract run across a design-heavy poster invents letterforms and emits
    more characters than a page of real prose, so volume cannot be the test.
    Token length can: real words are long, hallucinated ones are debris.
    """
    empty = {"long_token_ratio": 0.0, "one_char_ratio": 0.0,
             "mean_token_len": 0.0, "tokens": 0}
    if not text:
        return empty

    tokens = _TOKEN_RE.findall(text)
    n = len(tokens)
    if n == 0:
        return empty

    return {
        "long_token_ratio": round(sum(1 for t in tokens if len(t) >= 4) / n, 4),
        "one_char_ratio": round(sum(1 for t in tokens if len(t) == 1) / n, 4),
        "mean_token_len": round(sum(len(t) for t in tokens) / n, 2),
        "tokens": n,
    }


def log(msg: str) -> None:
    print(f"[ocr] {msg}", flush=True)


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------


def cache_path(url: str, cache: Path | None = None) -> Path:
    """Stable filename per URL. Content-addressing would need the bytes first."""
    return (cache or CACHE) / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.pdf"


def is_junk_url(url: str) -> bool:
    """AppleDouble resource forks: '._name.pdf' files a Mac left on a web server.

    They carry Content-Type: application/pdf but contain no document - just
    finder metadata. Fetching and OCRing them wastes time and pollutes the
    corpus with empty records.
    """
    name = urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]
    return name.startswith("._")


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        # Some unitn hosts return an empty body without a browser-ish Accept.
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Accept-Language": "it,en;q=0.8",
    })


def fetch_pdf(
    url: str,
    dest: Path,
    timeout: int = FETCH_TIMEOUT,
    allow_insecure: bool = False,
    expected_sha256: str | None = None,
) -> tuple[bool, str]:
    """Download one PDF. Returns (ok, message).

    Some unitn subdomains (disi.unitn.it) present a chain this server cannot
    verify. With --insecure we retry without verification, but *only* accept the
    result if its SHA-256 matches what the crawler recorded - so the content is
    still proven identical to what was originally fetched, even though the
    transport was not authenticated.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return True, "cached"

    data: bytes | None = None
    note = ""

    try:
        with urllib.request.urlopen(_request(url), timeout=timeout) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        return False, f"http {e.code}"
    except urllib.error.URLError as e:
        if not (allow_insecure and isinstance(e.reason, ssl.SSLCertVerificationError)):
            return False, f"error {type(e).__name__}: {e}"
        if not expected_sha256 or expected_sha256 == "None":
            return False, "ssl verify failed, and no recorded hash to fall back on"
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            with urllib.request.urlopen(_request(url), timeout=timeout, context=ctx) as r:
                data = r.read()
        except Exception as e2:  # noqa: BLE001
            return False, f"insecure retry failed: {type(e2).__name__}"
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            return False, "insecure fetch, hash does not match the crawl - rejected"
        note = " (unverified TLS, hash matched)"
    except Exception as e:  # noqa: BLE001
        return False, f"error {type(e).__name__}: {e}"

    if not data:
        return False, "empty response"

    # Magic bytes only. Content-Type lies: AppleDouble stubs are served as
    # application/pdf, and that is what let nine non-documents through.
    if not data.startswith(b"%PDF"):
        return False, f"not a pdf (starts {data[:8]!r})"

    tmp = dest.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(dest)                      # atomic: a .pdf on disk is always complete
    return True, f"{len(data) / 1024:.0f} KB{note}"


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


def load_pending(
    path: Path,
    limit: int | None,
    keep_junk: bool = False,
    min_pages: int = 0,
) -> list[dict]:
    """Read the pending list, drop non-documents, order for useful sampling."""
    rows, junk = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not keep_junk and is_junk_url(r.get("url", "")):
                junk += 1
                continue
            rows.append(r)
    if junk:
        log(f"skipped {junk} AppleDouble '._' stubs - not documents")

    def pages(r: dict) -> int:
        try:
            return int(r.get("page_count") or 0)
        except (TypeError, ValueError):
            return 0

    if min_pages:
        before = len(rows)
        rows = [r for r in rows if pages(r) >= min_pages]
        log(f"{len(rows)} of {before} have {min_pages}+ pages")

    # Smallest first, but documents with a *known* page count come before
    # unknowns. Sorting purely by page count floats every count-less record to
    # the top, so a --limit sample sees only the records we know least about.
    rows.sort(key=lambda r: (pages(r) == 0, pages(r)))
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
    q = text_quality(text)

    # Two distinct failure modes, worth naming separately rather than collapsing
    # into one boolean:
    #   empty  - almost nothing came out (a map, a photograph, a blank scan)
    #   noise  - plenty came out, but it does not read like language
    empty = per_page < MIN_CHARS_PER_PAGE
    noise = (
        not empty
        and q["tokens"] >= 20
        and q["long_token_ratio"] < MIN_LONG_TOKEN_RATIO
    )

    return {
        "url": url,
        "ok": True,
        "title": row.get("title"),
        "ocr_text": text,
        "ocr_chars": len(text),
        "ocr_pages": pages,
        "chars_per_page": round(per_page, 1),
        **q,
        # Flag rather than drop: a bus map legitimately has almost no text, and
        # that is a fact about the document, not a failure of the OCR.
        "ocr_empty": empty,
        "ocr_noise": noise,
        "ocr_suspect": empty or noise,
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
    ap.add_argument("--insecure", action="store_true",
                    help="on SSL verify failure, retry unverified and accept only "
                         "if the SHA-256 matches the crawl")
    ap.add_argument("--keep-junk", action="store_true",
                    help="do not skip AppleDouble '._' stubs")
    ap.add_argument("--min-pages", type=int, default=0,
                    help="only documents with at least N pages - use to sample "
                         "substantial documents rather than one-page posters")
    args = ap.parse_args()

    CACHE = args.cache
    CACHE.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for tool in ("pdftoppm", "tesseract"):
        if not shutil.which(tool):
            log(f"FAIL {tool} not found on PATH")
            return 1

    rows = load_pending(args.pending, args.limit,
                        keep_junk=args.keep_junk, min_pages=args.min_pages)
    log(f"{len(rows)} documents pending, {sum(int(r.get('page_count') or 0) for r in rows)} pages")

    # ---------------- fetch ----------------
    if not args.ocr_only:
        ok = failed = cached = mismatched = 0
        for i, row in enumerate(rows, 1):
            url = row["url"]
            dest = cache_path(url)
            was_cached = dest.exists()
            good, msg = fetch_pdf(
                url, dest,
                allow_insecure=args.insecure,
                expected_sha256=row.get("content_sha256"),
            )

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
