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
from urllib.parse import urlsplit, urlunsplit, parse_qsl, unquote

# --------------------------------------------------------------------------
# Basic normalisation
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
# Whitespace that is *not* a newline. Used when paragraph structure must survive.
_INLINE_WS_RE = re.compile(r"[^\S\n]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _letter_spaced_ratio(text: str, sample_chars: int = 3000,
                         min_tokens: int = 40) -> float:
    """Share of tokens that are a single *alphabetic* character.

    Alphabetic specifically: markdown tables produce runs of one-character
    tokens too, but they are '|' and '-', and repairing those would destroy
    real content.

    ``min_tokens`` guards against deciding from too little evidence. It must be
    high for a whole document and low for a single line - a spaced heading like
    'I N D U S T R I A L' is only ten tokens, and a document-level floor of 40
    silently skipped every line worth repairing.
    """
    toks = (text or "")[:sample_chars].split()
    if len(toks) < min_tokens:
        return 0.0
    return sum(1 for x in toks if len(x) == 1 and x.isalpha()) / len(toks)


def looks_letter_spaced(text: str, threshold: float = 0.4) -> bool:
    return _letter_spaced_ratio(text) > threshold


def repair_letter_spacing(text: str | None) -> str | None:
    """Undo per-glyph spacing in PDFs that position each character separately.

    Design tools (posters, brochures, the jobguidance FAQ PDFs) apply letter
    tracking, and pypdf then emits a space between every character:

        'W e l c o m e  t o  t h e  I n t e r n s h i p'

    Single space = inter-letter padding, double space = a real word boundary,
    so it is deterministically reversible. Measured on one FAQ PDF, this takes
    long_token_ratio from 0.000 to 0.584 - the difference between an
    unembeddable document and a usable one.

    250 documents in the corpus are affected, all pypdf-extracted.
    """
    if not text or not looks_letter_spaced(text):
        return text

    out = []
    for line in text.split("\n"):
        # Per line, because a document is often only partly affected - a poster
        # may have a spaced title over an ordinary paragraph.
        if _letter_spaced_ratio(line, sample_chars=10 ** 9, min_tokens=4) > 0.4:
            line = line.replace("  ", "\x00").replace(" ", "").replace("\x00", " ")
        out.append(line)
    return "\n".join(out)


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


def is_junk_url(url: str) -> bool:
    """AppleDouble resource forks: '._name.pdf' left on a web server by a Mac.

    They are served as application/pdf and contain Finder metadata, not a
    document. Twenty are in the corpus, all under disi.unitn.it/locigno, as
    empty records that would chunk and embed into nothing. ocr_pending.py
    already skips them; the corpus loader did not.
    """
    if not url:
        return False
    name = urlsplit(url).path.rsplit("/", 1)[-1]
    return name.startswith("._")


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


def detect_language_from_text(
    text: str,
    sample_chars: int = 1500,
    min_ratio: float = 1.0,
    min_hits: int = 3,
) -> str | None:
    """Count language-exclusive function words.

    ``min_ratio`` demands the winner beat the loser by that factor, and
    ``min_hits`` demands enough evidence to be worth acting on. Both default to
    permissive values for the plain fallback case; raise them when the result
    is going to override a declared language.
    """
    if not text:
        return None
    words = [w.lower() for w in _WORD_RE.findall(text[:sample_chars])]
    if len(words) < 20:
        return None
    it = sum(1 for w in words if w in _IT_MARKERS)
    en = sum(1 for w in words if w in _EN_MARKERS)
    hi, lo = max(it, en), min(it, en)
    if hi < min_hits or hi == lo:
        return None
    if lo and hi / lo < min_ratio:
        return None
    return "it" if it > en else "en"


def detect_language(url: str = "", text: str = "", declared: str | None = None) -> str:
    """Resolve a document's language.

    Priority: URL structure > declared (<html lang>) > text heuristic, with one
    correction - see below. Defaults to 'it', since the Italian side of unitn.it
    is the larger, authoritative corpus.

    Why the declared value is not simply trusted: Drupal serves *unaliased*
    ``/node/N`` URLs under the site default language, so ``<html lang="it">``
    appears on pages whose body is the English translation. 1,565 documents
    (2.5% of the corpus) were English text filed as Italian - concentrated on
    departmental sites, where it broke language preference in retrieval and
    wrote ``LANGUAGE: it`` into the embedded text of English chunks.

    A URL language marker (/en/, ?lang=, .en.html) is structural and still
    wins outright. Only where the URL says nothing does the body get to
    overrule the declaration, and then only on a clear margin - a stray English
    quotation in an Italian page must not flip it.
    """
    url_lang = detect_language_from_url(url)
    if url_lang:
        return url_lang

    dec: str | None = None
    if declared:
        v = declared.strip().lower().split("-")[0]
        dec = _LANG_CANON.get(v)

    if dec:
        # Demand a 2:1 margin and real evidence before contradicting <html lang>.
        strong = detect_language_from_text(text, min_ratio=2.0, min_hits=5)
        if strong and strong != dec:
            return strong
        return dec

    return detect_language_from_text(text) or "it"


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
# Document families (same document, different edition year)
# --------------------------------------------------------------------------


# unitn.coursecatalogue.cineca.it serves one page per course
# ('/corsi/<year>/<course_id>') and one page per exam/module within that
# course ('/corsi/<year>/<course_id>/insegnamenti/...'). All of them repeat
# most of the same course-level boilerplate, so for a broad question they
# score near-identically to a reranker - confirmed 2026-09-10: after the
# edition-year cap below stopped the HCI regulation editions from crowding
# the top 5, five of these cineca pages (one course page, four insegnamenti)
# immediately filled the freed slots instead, still keeping the correct
# target page out.
_CINECA_COURSE_RE = re.compile(r"^/corsi/(\d{4})/(\d+)")


def document_family_key(url: str | None) -> str | None:
    """Group pages that are near-duplicates of each other under one key.

    Distinct from ``doc_group_id``: that collapses language translations of
    one page. This collapses same-language pages that are structural
    siblings of one another - either yearly re-editions of the same document,
    or different exam-module pages under the same course catalogue entry -
    which doc_group_id treats as unrelated because their URLs differ.

    Found via the eval_v3 run of 2026-09-10: with reranking on and
    recency_weight at 0, three old editions of the HCI teaching regulation
    filled 3 of the reranker's top 10 slots, pushing the correct (undated,
    current) page to rank 24 - outside max_pages. Capping the regulation
    editions alone was not enough: five cineca course-catalogue pages for the
    same course then filled the freed slots instead. Fixing recency scoring
    alone does not address either case, because the crowding is about *count*
    of near-duplicate slots, not their individual scores.

    Two patterns recognised, in order:

    1. A cineca course-catalogue page - grouped by ``(academic year, course
       id)``, so the course page and every one of its insegnamenti children
       collapse together regardless of which specific module they describe.
    2. A bare edition year at the tail of the filename, before the extension
       - the shape ``year_from_title`` already recognises as an edition
       label, e.g. 'regolamento-didattico-lm-hci-2015.pdf'.

    Returns None for anything matching neither, so a document is never
    grouped with another by guesswork.
    """
    if not url:
        return None

    parts = urlsplit(url)
    if parts.netloc.lower().endswith("coursecatalogue.cineca.it"):
        m = _CINECA_COURSE_RE.match(parts.path)
        if m:
            return f"cineca-course:{m.group(1)}:{m.group(2)}"

    stem = _dashed(unquote(parts.path.rsplit("/", 1)[-1]))
    m = _TITLE_TAIL_YEAR_RE.search(stem)
    if not m:
        return None
    family = stem[: m.start()].rstrip("-_ ").lower()
    return family or None


# --------------------------------------------------------------------------
# Freshness
# --------------------------------------------------------------------------

_YEAR_URL_RE = re.compile(r"/((?:19|20)\d{2})(?:[-/_]|$)")
_ACADEMIC_YEAR_RE = re.compile(
    r"(?:a\.?\s*a\.?|anno accademico|academic year)\s*[:\-]?\s*"
    r"((?:19|20)\d{2})(?:\s*[/\-]\s*((?:19|20)?\d{2}))?",
    re.IGNORECASE,
)
_YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\s*[/\-]\s*((?:19|20)?\d{2})\b")

# Drupal serves uploads from a path stamped with the month the file was put
# there: /sites/cds/files/2025-02/guidelines.pdf. That is when someone uploaded
# it, not what it is about, and the crawler used it as effective_year for all
# 5,179 documents that have such a path - every one of them, without exception.
_UPLOAD_PATH_RE = re.compile(r"/((?:19|20)\d{2})[-/](?:0[1-9]|1[0-2])/")

# A regolamento states its own emanation in a running page header: "Emanato con
# DR n. 788 del 28/07/2025". Every UniTn decree ALSO opens with a preamble that
# cites other decrees in the same words - "Visto lo Statuto ... emanato con D.R.
# n. 5 di data 8 gennaio 2024; Visto il Regolamento Generale di Ateneo, emanato
# con D.R. n. 421 del 1 ottobre 2012" - so the phrase alone identifies the wrong
# document. Two things separate them, and both are checked below: a running
# header repeats the *same* decree verbatim on every page, while a preamble
# cites a different one each time; and a citation is introduced by Visto /
# di cui / ai sensi.
_MONTHS_IT = ("gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|"
              "settembre|ottobre|novembre|dicembre")
_DECREE_RE = re.compile(
    r"emanat[oa]\s+con\s+(?:D\.?\s*R\.?|decreto\s+rettorale)\s*n\.?\s*\d+\s*"
    r"(?:del|di\s+data)\s+"
    r"(?:\d{1,2}[/.\-]\d{1,2}[/.\-]((?:19|20)\d{2})"
    r"|\d{1,2}\s*°?\s+(?:" + _MONTHS_IT + r")\s+((?:19|20)\d{2}))",
    re.IGNORECASE,
)
_CITATION_FRAME_RE = re.compile(
    r"\b(?:vist[oa]|viste|di\s+cui|ai\s+sensi|modificat[oa])\b", re.IGNORECASE
)
_DECREE_HEAD_CHARS = 2500      # the page-1/2 header zone
_CITATION_LOOKBACK = 250
_LAST_UPDATED_RE = re.compile(
    r"(?:last\s+updated?|ultimo\s+aggiornamento|aggiornat[oa]\s+al)"
    r"[^\n]{0,40}?((?:19|20)\d{2})",
    re.IGNORECASE,
)

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


def academic_year_end(start: int, end: int | str | None) -> int | None:
    """End year of a one-year span, or None if it is not one.

    **The end year, not the start.** A guide for a.a. 2025/26 is the current
    guide for the whole of 2026; calling it 2025 makes ``1/(1+age)`` treat it as
    a year old on the day it is published, and - because a year taken from an
    upload path is already a calendar year - systematically ages every correctly
    tagged document by one against every path-dated one.

    A real academic year spans exactly one calendar year. The crawl emits some
    malformed ranges - '2016/2067', '2018/2022' - where the second half is a
    typo or an unrelated number; those are rejected rather than trusted.
    """
    if end is None:
        return None
    end = int(end)
    if end < 100:                       # '2025/26' shorthand
        end += (start // 100) * 100
    return end if end - start == 1 else None


def parse_academic_year(value: str | None, current_year: int | None = None) -> int | None:
    """End year of an 'A.A. 2025/2026' string, if the range is sane. See above."""
    if not value:
        return None
    m = re.match(r"\s*((?:19|20)\d{2})\s*/\s*((?:19|20)?\d{2})\s*$", str(value))
    if not m:
        return None
    return _valid(academic_year_end(int(m.group(1)), m.group(2)), current_year)


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
    if m:
        year = academic_year_end(int(m.group(1)), m.group(2)) or int(m.group(1))
        if _valid(year, current_year):
            return year

    m = _YEAR_RANGE_RE.search(head)
    if m:
        year = academic_year_end(int(m.group(1)), m.group(2))
        if _valid(year, current_year):
            return year

    m = _YEAR_URL_RE.search(urlsplit(url).path if url else "")
    if m and _valid(int(m.group(1)), current_year):
        return int(m.group(1))

    if last_modified:
        m = re.search(r"((?:19|20)\d{2})", last_modified)
        if m and _valid(int(m.group(1)), current_year):
            return int(m.group(1))

    return None


_TITLE_AY_RE = re.compile(
    r"\b((?:19|20)\d{2})\s*[/\-_]\s*((?:19|20)?\d{2})\b"
)
# A year in the last position of a filename stem, before the extension:
# 'regolamento-didattico-lm-ingegneria-energetica-2023.pdf'. Position is what
# makes it safe - it is the edition label, where a year loose in the middle
# ('Premio 2019 assegnato') is part of a sentence.
_TITLE_TAIL_YEAR_RE = re.compile(
    r"[-_ ]((?:19|20)\d{2})\s*(?:\.[a-z0-9]{2,4})?\s*$", re.IGNORECASE
)


def year_from_title(title: str | None, current_year: int | None = None) -> int | None:
    """Year stated in a filename, e.g. '09_Guida Facolta 2012-2013.pdf'.

    Alfresco-hosted PDFs have UUID URLs and often do not repeat the year inside
    the first 500 characters of text, so the filename is the only place it
    appears. Six Faculty of Law handbooks from 2007-2010 were dated to the
    current year for exactly this reason - and under a 1/(1+age) decay that
    made obsolete handbooks rank as freshly published.

    A span resolves to its end year, as everywhere else. A bare terminal year
    does not: 'regolamento-...-2023.pdf' is an edition label, and reading it as
    a.a. 2023/24 would be inventing a span the filename does not claim. The
    asymmetry is deliberate - guessing here would put the file in the same year
    as the upload path that is already known to be wrong.
    """
    if not title:
        return None
    m = _TITLE_AY_RE.search(title)
    if m:
        return _valid(academic_year_end(int(m.group(1)), m.group(2)), current_year)
    m = _TITLE_TAIL_YEAR_RE.search(title.strip())
    if m:
        return _valid(int(m.group(1)), current_year)
    return None


def upload_path_year(url: str | None) -> int | None:
    """Year in a Drupal upload path - '/sites/cds/files/2025-02/guide.pdf'.

    When the file was put on the server, not what it is about. Measured on this
    corpus: 5,179 documents carry such a path and the crawl's ``effective_year``
    equals the path year in **every one of them**, so this is also the test for
    "the crawler had nothing better than the upload date".
    """
    if not url:
        return None
    m = _UPLOAD_PATH_RE.search(urlsplit(url).path)
    return int(m.group(1)) if m else None


# Structural path components - they say where a document is served from, not
# what it is about, so they must never contribute to URL/question affinity.
_URL_STOPWORDS = frozenset("""
www unitn it en node sites default files cds download workspace spacesstore
alfresco system allegati pdf html htm php aspx index home page view print
tiki uploads media documents doc docs public web
""".split())

_URL_SPLIT_RE = re.compile(r"[^a-z0-9]+")


def _dashed(name: str | None) -> str | None:
    """Underscores to hyphens before year parsing.

    ``_TITLE_AY_RE`` ends on ``\b``, and ``_`` is a word character, so
    '2008_2009_syllabus.pdf' never matched while '2008-2009-syllabus.pdf' does.
    Underscore is the dominant separator in this corpus's Alfresco filenames.
    """
    return name.replace("_", "-") if name else name



def resolved_year(
    effective_year: int | None,
    title: str | None = None,
    url: str | None = None,
    current_year: int | None = None,
) -> int | None:
    """The year a document is *about*, correcting the crawl's fallback.

    ``effective_year`` as stored is unreliable in one specific, measurable way:
    when nothing better was available the crawl used the upload or fetch date,
    so a 2002 student guide and a 2008 syllabus both carry the current year.
    Under ``recency_penalty`` that is an inversion, not just noise - an obsolete
    handbook scores age 0 (multiplier 1.0) while ``GUIDA GIURISPRUDENZA
    2025-26`` scores age 1 (multiplier 0.5) and loses to it.

    Order: the filename's edition year wins, because for Alfresco PDFs it is the
    only honest statement of the year. Failing that, a stored year that merely
    echoes the upload path is treated as unknown rather than as fresh.
    """
    from_title = year_from_title(_dashed(title), current_year)
    if from_title is None and url:
        # Alfresco serves PDFs from a UUID path; the filename is the last segment.
        stem = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
        from_title = year_from_title(_dashed(stem), current_year)
    if from_title is not None:
        return from_title

    up = upload_path_year(url)
    if up is not None and effective_year == up:
        return None  # the crawl had nothing better than the upload date
    return effective_year


def title_edition_year(title: str | None, url: str | None = None, current_year: int | None = None) -> int | None:
    """Year from an explicit filename/title edition marker - nothing else.

    ``resolved_year`` answers "what year is this document about", and a year
    it recovers from crawl metadata (a body-text academic-year mention, a URL
    date segment, a Last-Modified header) is a fine answer to that question -
    'academic year 2023/2024' in the text is exactly the right year to surface
    when someone asks about that year. It is a much weaker basis for "is this
    document stale", which is what ``recency_penalty`` uses it for. Measured on
    four regressions from the 2026-09-10 recency rollout: the CEILS transfer
    guidelines (content-dated 2022), a missions-expense regulation (no
    resolvable year at all), and a Giurisprudenza exam-calendar page
    (correctly content-dated to the 2023/2024 academic year the question
    asked about) were all outranked by unrelated pages that merely carried a
    fresher metadata year - sociology internship pages, admission pages for
    other courses, a different Giurisprudenza calendar for the *wrong* year.
    None of the winners were more relevant; all of them just looked newer.

    Only a year baked into the filename itself
    ('regolamento-didattico-...-2017.pdf') is strong, low-noise evidence that
    a document is one of several competing editions - that is the one case
    (stale HCI regulation PDFs, verified against question #3) where the
    penalty measurably helps. Everything else should read as "unknown" for
    penalty purposes, even when ``resolved_year`` can say more for display.
    """
    from_title = year_from_title(_dashed(title), current_year)
    if from_title is not None:
        return from_title
    if url:
        stem = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
        return year_from_title(_dashed(stem), current_year)
    return None


def url_terms(url: str | None) -> frozenset[str]:
    """Content-bearing words in a URL: host labels and path/filename segments.

    ``corsi.unitn.it/en/human-computer-interaction/graduation/graduation-calendar``
    yields {corsi, human, computer, interaction, graduation, calendar}. Overlap
    with the question's words is what separates six near-identical graduation
    calendars, or a Law question from the DII calendar that states a different
    second-semester start date for the same academic year.
    """
    if not url:
        return frozenset()
    parts = urlsplit(url)
    raw = f"{parts.netloc}/{unquote(parts.path)}".lower()
    return _clean_terms(raw)


def url_host_terms(url: str | None) -> frozenset[str]:
    """Just the host labels: 'www.physics.unitn.it' -> {physics}.

    A host match is far stronger evidence than a path match. physics.unitn.it
    answers a physics question by virtue of *being* the physics site, while
    'external-research-period' in a path is a topic word that any department's
    equivalent page also carries - and pages served under /node/<id> have no
    descriptive slug at all, so path matching alone systematically loses them
    to slug-rich siblings from the wrong department.
    """
    if not url:
        return frozenset()
    return _clean_terms(urlsplit(url).netloc.lower())


def _clean_terms(raw: str) -> frozenset[str]:
    return frozenset(
        w for w in _URL_SPLIT_RE.split(raw)
        if len(w) > 2 and not w.isdigit() and w not in _URL_STOPWORDS
    )


def year_from_document(
    text: str | None,
    current_year: int | None = None,
    is_pdf: bool = True,
) -> int | None:
    """A year the document states about itself, not one that merely appears in it.

    Only two forms qualify, both anchored to the phrase that makes them a claim
    about this document:

      "Emanato con DR n. 788 del 28/07/2025"   - a regolamento's own emanation
      "Last updated on 22nd December 2022"     - an explicit revision date

    The anchoring is the whole point. Loose date matching reads the wrong year
    off almost every one of these files: the current Giurisprudenza guide says
    "a partire dall'anno accademico 2011-2012" six thousand characters in, and
    the energy regolamenti cite "ai sensi del D.M. del 16.03.2007". Those are
    history and legal reference, not publication dates.

    A decree year is accepted only when the *same* decree - same number, same
    date - appears at least twice, and its first appearance is not introduced by
    a citation frame. That is what distinguishes a running header from the
    preamble of a decreto, which cites three or four other decrees in identical
    language. Measured on the four regolamenti in hand the header repeats 3+
    times; in the commissioni decrees every cited decree appears once.

    The decree rule is for PDFs only. An HTML page has no running header, so a
    twice-repeated decree there is prose citing a regulation - measured, exactly
    one page in the corpus, disi/node/1603, which cites the Conto Terzi
    regolamento of 2015 and would otherwise be dated eleven years stale.
    """
    if not text:
        return None
    if not is_pdf:
        return _year_from_last_updated(text, current_year)

    counts: dict[str, int] = {}
    first: dict[str, re.Match] = {}
    for m in _DECREE_RE.finditer(text):
        token = "".join(m.group(0).split()).lower()   # 'n. 480' == 'n.480'
        counts[token] = counts.get(token, 0) + 1
        first.setdefault(token, m)
    for token, m in first.items():                    # earliest first
        if counts[token] < 2 or m.start() > _DECREE_HEAD_CHARS:
            continue
        before = text[max(0, m.start() - _CITATION_LOOKBACK):m.start()]
        if _CITATION_FRAME_RE.search(before):
            continue
        year = _valid(int(m.group(1) or m.group(2)), current_year)
        if year:
            return year

    return _year_from_last_updated(text, current_year)


def _year_from_last_updated(text: str, current_year: int | None = None) -> int | None:
    """Latest 'last updated on ...' the document states. Latest, because a page
    that lists several revisions is current as of the most recent one."""
    years = [int(m.group(1)) for m in _LAST_UPDATED_RE.finditer(text)]
    years = [y for y in years if _valid(y, current_year)]
    return max(years) if years else None


def resolve_effective_year(raw: dict, current_year: int | None = None) -> int | None:
    """Effective year for one crawl record, ranked by how the year was obtained.

    The v2 crawl computes ``effective_year`` for every document and re-deriving
    it from scratch would leave ~50% of the corpus with no freshness signal at
    all, so the crawler is still trusted by default. What changed is that
    "trusted by default" is not the same as "trusted unconditionally": for 5,179
    documents the crawl's year is simply the month-stamped Drupal upload path,
    which records when a file was put on the server. Four confirmed cases where
    that is the wrong year - the CEILS transfer guidelines (a 2022 document
    uploaded in 2025), the travel regulation (2015 -> 2026), the energy
    regolamenti (2016, 2023 and 2024 editions all -> 2024) and the a.a. 2007-08
    guides - all share the shape: an old document re-uploaded, dated by the
    re-upload.

    So an upload-path year is the weakest evidence there is, below anything the
    document says about itself. Order:

      1. ``academic_year`` ('2025/2026' -> 2026) - the crawler's own extraction
         from the document, and the most specific claim available
      2. what the document states about itself - its emanation decree or an
         explicit revision date - when the crawl had only the upload path
      3. the filename's edition label, same condition
      4. ``effective_year`` from the crawl, clamped to a plausible range
      5. the regex fallback, for records the crawl left empty
    """
    year = parse_academic_year(raw.get("academic_year"), current_year)
    if year:
        return year

    crawled = raw.get("effective_year")
    if isinstance(crawled, str) and crawled.isdigit():
        crawled = int(crawled)
    if not isinstance(crawled, int):
        crawled = None

    # Two ways the crawler can be holding a year it did not really find:
    # it read the upload path, or it fell back to the current year outright.
    from_path = crawled is not None and crawled == upload_path_year(raw.get("url"))
    from_default = crawled is not None and crawled == max_plausible_year(current_year) - 1
    weak = from_path or from_default or crawled is None

    if weak:
        stated = year_from_document(
            raw.get("text"), current_year, is_pdf=raw.get("doc_type") == "pdf"
        )
        if stated:
            return stated

        # Elsewhere a stray year in a title ("Premio 2019 assegnato") must not
        # override a year the crawler genuinely derived, which is why this is
        # reached only when the crawler's value is known to be weak.
        from_title = year_from_title(raw.get("title"), current_year)
        if from_title and from_title != crawled:
            return from_title

    if crawled is not None:
        year = _valid(crawled, current_year)
        if year:
            return year

    return extract_effective_year(
        url=raw.get("url") or "",
        text=raw.get("text") or "",
        last_modified=raw.get("last_modified"),
        current_year=current_year,
    )


def recency_penalty(
    year: int | None,
    current_year: int,
    weight: float = 1.0,
    unknown: float = 0.5,
) -> float:
    """Freshness multiplier, 1/(1+age), with two knobs measured into existence.

    ``unknown`` - what an undated document is worth. 0.5 treats "no evidence"
    as "old", which penalises the *crawler* rather than the document: the CEILS
    transfer guidelines are current but carry no parseable year, and took the
    same x0.5 as a genuinely stale file. Items 4, 5 and 13 all turn on this.

    ``weight`` - how hard the curve bites. At 1.0 this is the original: three
    years old is x0.25, which no relevance signal can overcome. That was
    tolerable against raw cosine scores, whose spread is wide; it is not
    against reranker scores squashed into (0, 1], where a strong and a
    mediocre match may differ by a few hundredths. Turning recency off
    entirely moved hit@10 from 0.750 to 0.800 but hit@1 from 0.525 to 0.375 -
    it is doing real work and real damage at once, so the useful setting is
    somewhere between veto and silence:

        effective = 1 - weight * (1 - raw)

    weight 1.0 keeps today's behaviour, 0.3 turns a x0.25 into x0.78, 0.0
    disables it.
    """
    raw = unknown if year is None else 1.0 / (1.0 + max(0, current_year - year))
    if weight == 1.0:
        return raw
    return 1.0 - weight * (1.0 - raw)
