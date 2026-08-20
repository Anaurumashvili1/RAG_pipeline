"""Text normalisation and metadata extraction.

Pure functions, no I/O, no model loading - so they are cheap to unit-test.
This is where the three fixes from the paper's error analysis live:
  - language detection      (IT/EN corpus)
  - doc_group_id            (collapses translations of the same page)
  - effective_year          (freshness signal for time-decay ranking)
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import urlsplit, urlunsplit, parse_qsl

# --------------------------------------------------------------------------
# Basic normalisation
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
# Whitespace that is *not* a newline. Used when paragraph structure must survive.
_INLINE_WS_RE = re.compile(r"[^\S\n]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_text(t: str | None, keep_breaks: bool = False) -> str:
    """Collapse whitespace and strip non-breaking spaces.

    ``keep_breaks`` preserves line and paragraph breaks. This matters more than
    it looks: the crawler emits markdown (trafilatura), so headings and
    paragraph boundaries are present in the source. Collapsing all whitespace
    flattened them away, which silently disabled ``SentenceSplitter``'s
    ``paragraph_separator="\\n\\n"`` - it could never match, so chunk boundaries
    ignored document structure entirely.

    Default stays False so titles and short fields collapse to one line.
    """
    if not t:
        return ""
    t = t.replace("\u00a0", " ").replace("\u200b", "")

    if not keep_breaks:
        return _WS_RE.sub(" ", t).strip()

    t = _INLINE_WS_RE.sub(" ", t)          # spaces and tabs, not newlines
    t = _BLANK_LINES_RE.sub("\n\n", t)     # at most one blank line
    return "\n".join(line.strip() for line in t.split("\n")).strip()


def doc_id_from_url(url: str) -> str:
    """Stable 16-hex identifier for a single page."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Language
# --------------------------------------------------------------------------

# UniTn expresses language in the path (/en/, /it/), in a query param (?lang=en),
# in a filename suffix (page.en.html) or via a dedicated subdomain.
_LANG_PATH_RE = re.compile(r"/(en|eng|english|it|ita|italiano)(?:/|$)", re.IGNORECASE)
_LANG_SUFFIX_RE = re.compile(r"[._-](en|it)\.(?:html?|php|aspx?)$", re.IGNORECASE)
_EN_SUBDOMAINS = ("international.", "en.")

_LANG_CANON = {
    "en": "en", "eng": "en", "english": "en",
    "it": "it", "ita": "it", "italiano": "it",
}

# Function words that are frequent, short and near-exclusive to one language.
_IT_MARKERS = {
    "di", "il", "la", "le", "gli", "che", "per", "con", "una", "del", "della",
    "degli", "delle", "sono", "anche", "presso", "corso", "corsi", "iscrizione",
    "domanda", "studenti", "ateneo",
}
_EN_MARKERS = {
    "the", "of", "and", "for", "with", "are", "this", "that", "from", "you",
    "your", "students", "course", "courses", "application", "enrolment",
    "enrollment", "university",
}

_WORD_RE = re.compile(r"[a-zàèéìòù]+", re.IGNORECASE)


# Scripts that rule out Italian or English outright. Checked only when the
# crawler declared no language, since detect_language() defaults to 'it' and
# would otherwise file a Chinese PDF as Italian.
_NON_LATIN_RE = re.compile(
    r"[一-鿿"      # CJK
    r"぀-ヿ"       # kana
    r"가-힯"       # hangul
    r"Ѐ-ӿ"       # Cyrillic
    r"֐-׿"       # Hebrew
    r"؀-ۿ"       # Arabic
    r"]"
)


def is_latin_script(text: str, sample_chars: int = 2000, threshold: float = 0.10) -> bool:
    """False when a meaningful share of the sample is non-Latin script.

    Deliberately tolerant: a single Chinese character in an otherwise Italian
    page (a name, a quotation) should not disqualify it. The threshold asks
    whether the *document* is non-Latin, not whether it contains any.
    """
    if not text:
        return True
    sample = text[:sample_chars]
    letters = [c for c in sample if c.isalpha()]
    if len(letters) < 20:
        return True
    non_latin = sum(1 for c in letters if _NON_LATIN_RE.match(c))
    return (non_latin / len(letters)) < threshold


def detect_language_from_url(url: str) -> str | None:
    """Language from URL structure. Returns 'en', 'it' or None."""
    if not url:
        return None
    parts = urlsplit(url)

    m = _LANG_SUFFIX_RE.search(parts.path)
    if m:
        return _LANG_CANON[m.group(1).lower()]

    m = _LANG_PATH_RE.search(parts.path)
    if m:
        return _LANG_CANON[m.group(1).lower()]

    for key, value in parse_qsl(parts.query):
        if key.lower() in {"lang", "language", "locale"}:
            v = value.lower().split("-")[0]
            if v in _LANG_CANON:
                return _LANG_CANON[v]

    host = parts.netloc.lower()
    if any(host.startswith(s) for s in _EN_SUBDOMAINS):
        return "en"

    return None


def detect_language_from_text(text: str, sample_chars: int = 1500) -> str | None:
    """Fallback heuristic: count language-exclusive function words."""
    if not text:
        return None
    words = [w.lower() for w in _WORD_RE.findall(text[:sample_chars])]
    if len(words) < 20:
        return None
    it = sum(1 for w in words if w in _IT_MARKERS)
    en = sum(1 for w in words if w in _EN_MARKERS)
    if it == en:
        return None
    return "it" if it > en else "en"


def detect_language(url: str = "", text: str = "", declared: str | None = None) -> str:
    """Resolve a document's language.

    Priority: value declared by the scraper (<html lang>) > URL structure > text heuristic.
    Defaults to 'it', since the Italian side of unitn.it is the larger, authoritative corpus.
    """
    if declared:
        v = declared.strip().lower().split("-")[0]
        if v in _LANG_CANON:
            return _LANG_CANON[v]
    return detect_language_from_url(url) or detect_language_from_text(text) or "it"


# --------------------------------------------------------------------------
# Translation grouping
# --------------------------------------------------------------------------


def canonical_group_url(url: str) -> str:
    """Strip language markers from a URL so translations collapse to one key.

    https://www.unitn.it/en/ateneo/123/page  ->  https://www.unitn.it/ateneo/123/page
    https://www.unitn.it/it/ateneo/123/page  ->  https://www.unitn.it/ateneo/123/page
    """
    if not url:
        return ""
    parts = urlsplit(url)

    host = parts.netloc.lower()
    for prefix in _EN_SUBDOMAINS:
        if host.startswith(prefix):
            host = host[len(prefix):]
            break
    host = host[4:] if host.startswith("www.") else host

    path = _LANG_SUFFIX_RE.sub(lambda m: "." + m.group(0).rsplit(".", 1)[1], parts.path)
    path = _LANG_PATH_RE.sub("/", path)
    path = re.sub(r"//+", "/", path).rstrip("/")

    query = "&".join(
        f"{k}={v}"
        for k, v in parse_qsl(parts.query)
        if k.lower() not in {"lang", "language", "locale"}
    )

    return urlunsplit((parts.scheme or "https", host, path, query, ""))


def doc_group_id(url: str, hreflang_group: str | None = None) -> str:
    """Identifier shared by every translation of the same page.

    Pass ``hreflang_group`` once the new scraper captures
    <link rel="alternate" hreflang="..."> - that mapping is authoritative and
    should win over the URL heuristic.
    """
    key = hreflang_group or canonical_group_url(url)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Freshness
# --------------------------------------------------------------------------

_YEAR_URL_RE = re.compile(r"/((?:19|20)\d{2})(?:[-/_]|$)")
_ACADEMIC_YEAR_RE = re.compile(
    r"(?:a\.?\s*a\.?|anno accademico|academic year)\s*[:\-]?\s*((?:19|20)\d{2})",
    re.IGNORECASE,
)
_YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\s*[/\-]\s*(?:(?:19|20)?\d{2})\b")

# A year beyond next academic year is a parse error, not a fresh document, and
# under the 1/(1+age) decay it scores maximum freshness. The old ceiling of 2100
# let "A.A. 2016/2067" rank above everything on the site.
_MIN_YEAR = 1990


def max_plausible_year(current_year: int | None = None) -> int:
    """Newest year a document may legitimately claim: next academic year."""
    if current_year is None:
        from datetime import date

        current_year = date.today().year
    return current_year + 1


def _valid(year: int | None, current_year: int | None = None) -> int | None:
    if not year:
        return None
    return year if _MIN_YEAR <= year <= max_plausible_year(current_year) else None


def parse_academic_year(value: str | None, current_year: int | None = None) -> int | None:
    """Start year of an 'A.A. 2025/2026' string, if the range is sane.

    The crawl emits some malformed ranges - '2016/2067', '2018/2022' - where the
    second half is a typo or an unrelated number. A real academic year spans
    exactly one calendar year, so anything else is rejected rather than trusted.
    """
    if not value:
        return None
    m = re.match(r"\s*((?:19|20)\d{2})\s*/\s*((?:19|20)?\d{2})\s*$", str(value))
    if not m:
        return None
    start = int(m.group(1))
    end = int(m.group(2))
    if end < 100:                       # '2025/26' shorthand
        end += (start // 100) * 100
    if end - start != 1:
        return None
    return _valid(start, current_year)


def extract_effective_year(
    url: str = "",
    text: str = "",
    last_modified: str | None = None,
    sample_chars: int = 500,
    current_year: int | None = None,
) -> int | None:
    """Best available "what year does this document describe" signal.

    Order matters: an explicit academic year in the text beats a year in the URL
    path, which beats the server's Last-Modified header.
    """
    head = text[:sample_chars] if text else ""

    m = _ACADEMIC_YEAR_RE.search(head)
    if m and _valid(int(m.group(1)), current_year):
        return int(m.group(1))

    m = _YEAR_RANGE_RE.search(head)
    if m and _valid(int(m.group(1)), current_year):
        return int(m.group(1))

    m = _YEAR_URL_RE.search(urlsplit(url).path if url else "")
    if m and _valid(int(m.group(1)), current_year):
        return int(m.group(1))

    if last_modified:
        m = re.search(r"((?:19|20)\d{2})", last_modified)
        if m and _valid(int(m.group(1)), current_year):
            return int(m.group(1))

    return None


def resolve_effective_year(raw: dict, current_year: int | None = None) -> int | None:
    """Effective year for one crawl record, trusting the crawler first.

    The v2 crawl computes ``effective_year`` for every document. Re-deriving it
    from scratch throws that away and leaves ~50% of the corpus with no freshness
    signal at all. Priority:

      1. ``academic_year`` ('2025/2026') - the most specific claim available
      2. ``effective_year`` from the crawl, clamped to a plausible range
      3. the regex fallback, for records the crawl left empty
    """
    year = parse_academic_year(raw.get("academic_year"), current_year)
    if year:
        return year

    crawled = raw.get("effective_year")
    if isinstance(crawled, str) and crawled.isdigit():
        crawled = int(crawled)
    if isinstance(crawled, int):
        year = _valid(crawled, current_year)
        if year:
            return year

    return extract_effective_year(
        url=raw.get("url") or "",
        text=raw.get("text") or "",
        last_modified=raw.get("last_modified"),
        current_year=current_year,
    )


def recency_penalty(year: int | None, current_year: int) -> float:
    """1 / (1 + age) multiplier from your Timestamps note. Unknown year -> mild penalty."""
    if year is None:
        return 0.5
    age = max(0, current_year - year)
    return 1.0 / (1.0 + age)
