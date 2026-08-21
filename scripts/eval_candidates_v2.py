#!/usr/bin/env python3
"""Build a worksheet of candidate documents to write evaluation questions about.

This script never writes a question. It selects *what you write about*, which is
where code actually helps: left to intuition you will write about whatever you
happened to look at, and the set will be unrepresentative in ways you cannot
see.

    python3 scripts/eval_candidates_v2.py --corpus dataset.ocr.jsonl --out evalprep/cand

Produces three files:

    <out>_targets.csv      stratified documents to write questions about
    <out>_faq.csv          real questions harvested from FAQ pages, with source
    <out>_report.txt       what was selected, what was skipped, and why

===========================================================================
WHY THIS IS NOT STRATIFIED ON doc_type
===========================================================================
``doc_type`` conflates three orthogonal axes in a single field:

    format   page, pdf                      how it was served
    genre    event, news, research, people  what kind of thing it is
    topic    admissions, course, scholarship, tuition, housing, service, phd

A document gets exactly one value, so a PDF *about* admissions is typed ``pdf``
and the topic is lost. Measured on this corpus:

    topic         PDFs on topic    docs actually typed as that topic
    admissions        1,136                  321
    course            1,229                  185     <- regolamenti, manifesti
    scholarship       1,624                  780     <- the bandi
    tuition             242                   38
    service             787                  309

Stratifying on ``doc_type`` therefore samples HTML landing pages and silently
excludes the 16,391 PDFs where the precise, citable answers live. It also
misses ``graduation`` (laurea / tesi / esame di laurea / proclamazione)
entirely - 7,028 documents carrying no relevant type at all.

The field also appears to have been assigned from the URL prefix: 37 documents
about accommodation are typed ``tuition`` because they live under
``/studiare/tasse-borse-alloggi/``.

So: topic is derived here from URL + title + opening text, format is a separate
axis with its own quota, and ``doc_type`` is used only to *exclude* genres
(people/news/event/research) that no student question should target.

===========================================================================
OTHER CHANGES FROM v1
===========================================================================
1. ``is_answerable()`` was defined, documented and never called, so ``bilingual``
   returned twelve ``webapps.unitn.it/du/`` person pages. It is now enforced.

2. Selection ran on the raw corpus while ``load_documents()`` drops
   ``duplicate_of`` / ``low_content`` / non-it-en, so 8 of 72 targets were
   absent from the index and scored ``hit@k = 0`` regardless of retrieval
   quality. Eligibility is now enforced with the loader's own rules.

3. ``temporal`` masked any 20xx in the URL, which matched professors'
   course-folder names (``~passerini/teaching/2025-2026/...`` against
   ``fm2024/SLIDES/...``). Those are different courses, not revisions. Personal
   teaching trees are now excluded.

4. Empty categories were silent - ``ocr`` produced zero rows against the
   pre-OCR corpus and the summary just omitted the line. Every category now
   reports filled/requested and the report names the shortfall.

5. ``expected_url`` was a single URL, but ``select_pages()`` dedups by
   ``doc_group_id`` and returns one URL per group, so a correct bilingual
   retrieval could score 0 for surfacing the sibling. The column is now
   ``acceptable_urls`` (pipe-separated) and ``hit_at_k`` must be updated to
   accept a list.

6. ``bilingual`` is built from reciprocal language-switcher links rather than
   URL symmetry. See ``reciprocal_pairs()`` for why that matters.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, unquote

# --------------------------------------------------------------------------
# Topic - derived, because doc_type does not carry it (see module docstring)
# --------------------------------------------------------------------------

# Deliberately procedural rather than lexical. A bare "laurea" matches every
# "corso di laurea" landing page - which is why an earlier pass counted 7,028
# "graduation" documents that were mostly course descriptions. Each pattern
# must name a procedure, a document type or an office, not a subject area.
TOPIC_PATTERNS = {
    "admissions":  r"ammission|immatricolazion|iscrizion|admission|enrol(ment|ling)|call.?for.?application|test.?d.?ingresso|\btolc\b|graduatoria|bando.?di.?ammissione|ranking.?list",
    "tuition":     r"\btasse\b|tuition.?fee|contribuzione.?studentesc|esonero|rimborso.?tass|\bisee\b|rate.?e.?scadenz|tassa.?regional|pagamento.?tass",
    "scholarship": r"borsa.?di.?studio|borse.?di.?studio|scholarship|assegno.?di.?tutorat|premio.?di.?laurea|bando.?borsa|fellowship",
    "housing":     r"alloggi|accommodation|residenza.?universitar|studentat|\bmensa\b|canteen|opera.?universitaria",
    "course":      r"regolamento.?didattico|manifesto.?degli.?studi|piano.?di.?stud|scheda.?del.?corso|study.?plan|course.?catalogue|programme.?structure",
    "graduation":  r"esame.?di.?laurea|proclamazione|domanda.?di.?laurea|seduta.?di.?laurea|graduation.?(session|calendar|deadline)|final.?exam.?regulation|deposito.?tesi|thesis.?submission",
    "mobility":    r"erasmus|mobilità.?internazional|international.?mobility|studiare.?all.?ester|study.?abroad|traineeship.?abroad",
    "internship":  r"tirocini|internship|job.?guidance|placement.?servic|convenzione.?di.?tirocinio",
    "services":    r"bibliotec|library.?servic|centro.?linguistico|language.?centre|test.?center|help.?desk|segreteria.?student|sportello",
}
_TOPIC_RE = {k: re.compile(v, re.IGNORECASE) for k, v in TOPIC_PATTERNS.items()}

# Genres no student question should ever target. Used only as an exclusion -
# never as a stratum.
_DEAD_TYPES = ("people", "research", "event", "news")

# Administrative-transparency and accounting trees. Real documents, indexed
# rightly, but not what a student assistant answers from. They also trip the
# fee patterns: "Trasferimenti_correnti_III_trimestre_2021.pdf" is a quarterly
# accounting return, not a tuition page.
_ADMIN_PATHS = ("amministrazione-trasparente", "/bilanci", "trasferimenti_correnti",
                "anticorruzione", "prevenzione-della-corruzione", "/albo",
                "/gare", "/appalti", "/concorsi", "amministrazione-aperta")

# A title that is a bare UUID or hash carries no human-readable name, so a
# citation to it tells the user nothing. Common on the Alfresco download URLs.
_UUID_TITLE = re.compile(
    r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$|^[0-9a-f]{16,}$",
    re.IGNORECASE)

# Documents describing a partner institution under a double-degree agreement.
# DL_TILBURG_TuitionFees_and_StudyPlan.pdf states *Tilburg's* fees; a question
# written from it would score the bot on another university's policy.
_PARTNER_DOC = re.compile(r"/DL_|_DL_|double.?degree", re.IGNORECASE)


def topics_of(r: dict, body_chars: int = 600) -> list[str]:
    """Every topic this document plausibly belongs to.

    Multi-label on purpose: a bando for a mobility scholarship is both. Quotas
    fill per topic, so overlap costs nothing and reflects reality.

    A match in the title or the *final* path segment is strong; a match in the
    body alone is not enough. Body-only matching is what put an anti-corruption
    report and two quarterly accounting returns into the tuition stratum.

    Only the final segment is read, because unitn.it groups whole sections
    under one compound segment: every fee page lives under
    ``/studiare/tasse-borse-alloggi/``, which contains the trigger word for
    tuition *and* scholarship *and* housing at once, so any ancestor segment
    files a page under all three. That compound segment is also why 37
    accommodation pages carry ``doc_type: tuition``.
    """
    tail = unquote(urlsplit(r.get("url") or "").path).rstrip("/").split("/")[-1]
    head = tail + " " + (r.get("title") or "")
    return [k for k, p in _TOPIC_RE.items() if p.search(head)]


def weak_topics_of(r: dict, body_chars: int = 600) -> list[str]:
    """Topics matched anywhere, including the body. Reported, never sampled."""
    blob = (unquote(r.get("url") or "") + " " + (r.get("title") or "") + " "
            + (r.get("text") or "")[:body_chars])
    return [k for k, p in _TOPIC_RE.items() if p.search(blob)]


def fmt_of(r: dict) -> str:
    ct = (r.get("content_type") or "").split(";")[0].strip().lower()
    if ct == "application/pdf" or (r.get("url") or "").lower().endswith(".pdf"):
        return "pdf"
    return "html"


# --------------------------------------------------------------------------
# What counts as a page a student could ask a question about
# --------------------------------------------------------------------------

# In the index (rightly, as distractors) but never a question target.
_NOISE_HOSTS = {
    "teseo.unitn.it",          # OJS journal platform - 6,187 research articles
    "lavoraconnoi.unitn.it",   # job postings and concorso materials - 5,011
    "eventi.unitn.it", "event.unitn.it",
    "mag.unitn.it", "webmagazine.unitn.it", "pressroom.unitn.it",
    "r1.unitn.it", "r.unitn.it", "projects.unitn.it",
    "nanolab.physics.unitn.it", "grace.unitn.it", "acme.soc.unitn.it",
    "swsp.soc.unitn.it", "cjm.unitn.it", "www-ceel.economia.unitn.it",
    "c2s2.unitn.it", "ppsn2026.disi.unitn.it", "donazioni.unitn.it",
    "apps.cimec.unitn.it", "wiki.cimec.unitn.it", "lims.unitn.it",
    "dol.unitn.it", "alumni.unitn.it",
}

_STUDENT_HOSTS = {
    "www.unitn.it", "corsi.unitn.it", "orienta.unitn.it", "borse.unitn.it",
    "phd.unitn.it", "webapps.unitn.it", "www.cla.unitn.it",
    "www.biblioteca.unitn.it", "www.testcenter.unitn.it",
    "www.jobguidance.unitn.it", "unitrentosport.unitn.it",
    "collegioclesio.unitn.it", "iecs.unitn.it", "iid.unitn.it",
    "www.centro3a.unitn.it", "formazioneinsegnanti.unitn.it",
    "www.sociologia.unitn.it", "www.economia.unitn.it", "www.lettere.unitn.it",
    "www.disi.unitn.it", "disi.unitn.it", "www.maths.unitn.it",
    "www.physics.unitn.it", "www.cibio.unitn.it", "www.sis.unitn.it",
    "www.cimec.unitn.it", "www.cogsci.unitn.it", "www.soi.unitn.it",
    "www.cismed.unitn.it", "www.openscience.unitn.it", "master-m3.unitn.it",
}

# Staff directories, publication lists, person records. They have text and
# metadata and pass every length filter, which is why they need naming.
_DEAD_PATHS = ("/du/", "/persona/", "/persone/", "/strutturaaccademica/",
               "/strutturagestionale/", "/pubblicazioni/", "/publications/",
               "/rubrica", "/prenotazionieventi/", "/albo/",
               "/amministrazioneaperta/", "/propostetesi/")

# Personal teaching trees: real content, but versioned by course edition rather
# than by revision, so useless as freshness or near-duplicate tests.
_PERSONAL_TREE = re.compile(r"/~|/didattica/|/teaching/|/slides?/", re.IGNORECASE)

# Apache autoindex ("Index of /rseba/DIDATTICA/fm2024/EXAMPLE-TESTS").
_AUTOINDEX = re.compile(r"^index of /", re.IGNORECASE)

# Posters, flyers and accounting returns: text-bearing, never question targets.
_POSTER_WORDS = ("locandina", "flyer", "poster", "consuntivo", "bilancio",
                 "verbale", "organigramma")

_Q_RE = re.compile(r"^[#\s*]*(.{10,140}\?)\s*$", re.MULTILINE)

# MASTER-M3 is a metamaterials project, not a master's degree; "assegno di
# ricerca" is not "assegno di tutorato"; "collegio" is a residence and a body.
_TRAP_TERMS = ("master", "magistrale", "dottorato", "tirocinio", "borsa",
               "assegno", "collegio", "residenza")

# Hosted on unitn.it but describing a different institution. DL_CEAS_FAQ.pdf is
# University of Cincinnati and supplied 27 of 191 harvested questions in v1,
# all about grad.uc.edu admissions and US GPA thresholds.
_FOREIGN_MARKERS = ("grad.uc.edu", "college of engineering and applied science",
                    "university of cincinnati", "toefl ibt code", "gre general")


def is_index_eligible(r: dict, min_chars: int = 150) -> tuple[bool, str]:
    """Will this document survive load_documents() and reach the index?

    Must mirror data.load_documents(), or targets are unreachable by
    construction and score hit@k = 0 no matter how good retrieval is.
    """
    if not (r.get("url") or "").strip():
        return False, "no_url"
    if len(r.get("text") or "") < min_chars:
        return False, "short_text"
    if r.get("duplicate_of"):
        return False, "duplicate_of"
    if r.get("low_content"):
        return False, "low_content"
    if r.get("boilerplate"):
        return False, "boilerplate"
    lang = (r.get("lang") or "").strip().lower().split("-")[0]
    if lang and lang not in ("it", "en"):
        return False, f"lang_{lang}"
    return True, ""


def is_answerable(r: dict) -> bool:
    """Could a human write a factual question this page answers?"""
    if r.get("doc_type") in _DEAD_TYPES:
        return False
    parts = urlsplit(r.get("url", ""))
    if parts.netloc.lower() in _NOISE_HOSTS:
        return False
    path = unquote(parts.path).lower()
    if any(d in path for d in _DEAD_PATHS):
        return False
    if any(d in path for d in _ADMIN_PATHS):
        return False
    # Paginated index pages ("...?page=19"): a list of links to the bandi, not
    # a document stating anything. Retrieval surfacing one is not a hit.
    if re.search(r"(^|&)page=\d+", parts.query):
        return False
    title = (r.get("title") or "").strip()
    if not title or _AUTOINDEX.match(title) or _UUID_TITLE.match(title):
        return False
    if any(w in title.lower() for w in _POSTER_WORDS):
        return False
    return len(r.get("text") or "") > 800


def is_student_facing(r: dict) -> bool:
    return urlsplit(r.get("url", "")).netloc.lower() in _STUDENT_HOSTS


# --------------------------------------------------------------------------
# Grouping keys
# --------------------------------------------------------------------------

def norm_group(url: str) -> str:
    p = urlsplit(url)
    host = p.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    path = re.sub(r"/(en|it)(/|$)", "/", p.path)
    path = re.sub(r"[._-](en|it)(\.\w+)$", r"\2", path)
    return f"{host}{path.rstrip('/')}"


def year_group(url: str) -> str:
    k = norm_group(url)
    k = re.sub(r"20\d{2}[-_/]20\d{2}", "YYYY", k)
    k = re.sub(r"20\d{2}[-_]\d{2}", "YYYY", k)
    return re.sub(r"20\d{2}", "YYYY", k)


_VERSION_NOISE = re.compile(
    r"(20\d{2}[-_/.]?\d{0,4})|(\bv\d+\b)|(\bed\b)|(\brev\b)|(\d{1,2}[._-]\d{1,2}[._-]\d{2,4})",
    re.IGNORECASE)


def title_key(r: dict) -> str:
    """Filename/title with dates and version markers stripped.

    Catches clusters year_group() misses because the version is in the filename:
    FAQ tirocinio.pdf / FAQ tirocinio 28.10.25.pdf / 2025-10-28_FAQ TIROCINIO IN
    CONVENZIONE.pdf on economia.unitn.it.
    """
    t = unquote(r.get("title") or "")
    t = re.sub(r"\.(pdf|docx?|xlsx?)$", "", t, flags=re.IGNORECASE)
    t = _VERSION_NOISE.sub(" ", t)
    t = re.sub(r"[^0-9a-zà-ÿ]+", " ", t.lower()).strip()
    return re.sub(r"\s+", " ", t)


def shingle(text: str, n: int = 2000) -> set:
    return set(re.findall(r"[0-9a-zà-ÿ]{4,}", (text or "")[:n].lower()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

_KEEP_KEYS = ("url", "title", "text", "lang", "doc_type", "department",
              "effective_year", "academic_year", "extractor", "content_type",
              "last_modified", "ocr_suspect", "page_count")


def load(path: Path) -> tuple[list[dict], collections.Counter]:
    """Load the corpus, keeping only index-eligible rows.

    sha256 fields are dropped on read. ``out_links`` is kept only for
    student-facing hosts - it is most of the 713MB, and only the language-pair
    reconstruction needs it.
    """
    rows, dropped = [], collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                dropped["malformed_json"] += 1
                continue
            ok, why = is_index_eligible(r)
            if not ok:
                dropped[why] += 1
                continue
            row = {k: r.get(k) for k in _KEEP_KEYS}
            if urlsplit(row["url"] or "").netloc.lower() in _STUDENT_HOSTS:
                row["out_links"] = {x.rstrip("/") for x in (r.get("out_links") or [])}
            rows.append(row)
    return rows, dropped


# --------------------------------------------------------------------------
# Translation pairs
# --------------------------------------------------------------------------

def reciprocal_pairs(rows: list[dict]) -> dict[str, str]:
    """Recover IT/EN siblings from the language-switcher link.

    Why not URL symmetry: unitn.it translates the path segments, not just the
    /it/ and /en/ prefix, so stripping the prefix pairs almost nothing.
    Measured on this corpus, URL symmetry pairs 18 of 893 English pages on
    www.unitn.it (2.0%) and 12 of 641 on corsi.unitn.it (1.9%) - while pairing
    5,773 person records on webapps.unitn.it, which nobody queries. The feature
    it fails on is exactly the one retrieval's doc_group dedup depends on.

    Why not hreflang: the crawler does not emit it yet (PROGRESS §4). But it
    already captured ``out_links``, and the language switcher is an ordinary
    outbound link, so the pairing can be rebuilt from data already in hand -
    no re-crawl.

    Rule: A and B are siblings if A links to B, B links back to A, same host,
    different language. Reciprocity is what makes it safe: over the
    student-facing hosts this yields 991 pairs and *zero* ambiguous cases, and
    recovers what no regex could, e.g.
        https://www.unitn.it/it/rit
        https://www.unitn.it/en/international/coming-unitrento/degree-seeking-student/rit-...
        https://borse.unitn.it/borse  <->  https://borse.unitn.it/en/scolarships
    (the typo is in the real URL).

    Worth feeding to doc_group_id() as ``hreflang_group``: it would switch on
    the IT/EN dedup that select_pages() currently performs almost nowhere.
    """
    lang = {r["url"]: (r.get("lang") or "").lower().split("-")[0]
            for r in rows if "out_links" in r}
    links = {r["url"]: r["out_links"] for r in rows if "out_links" in r}
    known = {u.rstrip("/"): u for u in lang}

    pairs: dict[str, str] = {}
    for u, lg in lang.items():
        if lg not in ("it", "en"):
            continue
        host = urlsplit(u).netloc.lower()
        cands = []
        for tgt in links.get(u, ()):
            real = known.get(tgt)
            if (not real or real == u
                    or urlsplit(real).netloc.lower() != host
                    or lang.get(real) not in ("it", "en")
                    or lang[real] == lg):
                continue
            if u.rstrip("/") in links.get(real, ()):
                cands.append(real)
        if len(cands) == 1:
            pairs[u] = cands[0]
    return pairs


# --------------------------------------------------------------------------
# FAQ harvest
# --------------------------------------------------------------------------

def harvest_faq(rows: list[dict]) -> list[dict]:
    """Real questions written by UniTn staff, with the page that answers them."""
    out = []
    for r in rows:
        url = r.get("url", "")
        title = r.get("title") or ""
        if "faq" not in url.lower() and "faq" not in title.lower():
            continue
        text = r.get("text") or ""
        foreign = any(m in text[:6000].lower() for m in _FOREIGN_MARKERS)
        seen = set()
        for m in _Q_RE.finditer(text):
            q = " ".join(m.group(1).split())
            if q.lower() in seen:
                continue
            seen.add(q.lower())
            out.append({
                "harvested_question": q,
                "rewrite_as_a_student_would_ask": "",
                "acceptable_urls": url,
                "lang": r.get("lang") or "",
                "topic": ",".join(topics_of(r)),
                "page_title": title[:90],
                # the page describes a partner institution, not UniTn: good
                # ambiguity tests, bad plain-fact tests
                "about_other_institution": "YES" if foreign else "",
                "keep": "",
            })
    return out


# --------------------------------------------------------------------------
# Target selection
# --------------------------------------------------------------------------

TOPIC_QUOTAS = {
    "admissions": 8, "tuition": 8, "scholarship": 8, "course": 8,
    "graduation": 8, "mobility": 6, "internship": 6, "housing": 5,
    "services": 6,
}
STRUCTURAL_QUOTAS = {
    "bilingual": 10, "temporal": 10, "nearduplicate": 10, "trap": 10, "ocr": 5,
}
# Share of each topic quota reserved for PDFs. The bandi, regolamenti and
# manifesti are PDFs; sampling only HTML gives you landing pages that link to
# the answer instead of stating it.
PDF_SHARE = 0.4

NOTES = {
    "bilingual": "page exists in IT and EN; ask in BOTH languages. Both URLs count as a hit - doc_group dedup returns only one of them.",
    "temporal": "an outdated sibling is in the index; the current one must win",
    "nearduplicate": "a near-identical sibling exists; tests whether the right version is cited, not just the right topic",
    "ocr": "text exists ONLY because of OCR. Retrieval-completeness test: score whether the page is reachable, not whether a student would ask this.",
    "trap": "term collides with a different concept; a confident wrong answer is the risk",
}


def pick_targets(rows, topic_quotas, structural_quotas, seed):
    rng = random.Random(seed)
    out: list[dict] = []
    filled: dict[str, tuple[int, int]] = {}
    used: set[str] = set()

    answerable = [r for r in rows if is_answerable(r)]
    student = [r for r in answerable if is_student_facing(r)]
    for r in student:
        r["_topics"] = topics_of(r)
        r["_weak"] = weak_topics_of(r)
        r["_fmt"] = fmt_of(r)
        r["_partner"] = bool(_PARTNER_DOC.search(r.get("url") or ""))

    def emit(category, picks, also=None, distract=None):
        for r in picks:
            u = r.get("url", "")
            urls = [u] + [x for x in (also or {}).get(u, []) if x]
            out.append({
                "category": category,
                "question": "",
                "gold_answer": "",
                # every URL that counts as a hit
                "acceptable_urls": "|".join(dict.fromkeys(urls)),
                # documents that must NOT win. Not scored - shown so you can see
                # what the question competes against while writing it.
                "distractor_urls": "|".join((distract or {}).get(u, [])[:3]),
                "format": r.get("_fmt") or fmt_of(r),
                "topics": ",".join(r.get("_topics") or topics_of(r)),
                # describes a partner institution under a double-degree
                # agreement, so its fees/rules are not UniTn's
                "partner_doc": "YES" if r.get("_partner") else "",
                "lang": r.get("lang") or "",
                "year": r.get("effective_year") or "",
                "doc_type_raw": r.get("doc_type") or "",
                "text_len": len(r.get("text") or ""),
                "page_title": (r.get("title") or "")[:90],
                "note": NOTES.get(category, ""),
            })

    def take(pool, n, category, also=None, distract=None):
        pool = [r for r in pool if r.get("url") not in used]
        picks = rng.sample(pool, min(n, len(pool)))
        for p in picks:
            used.add(p.get("url"))
        emit(category, picks, also, distract)
        return picks

    # --- topical strata, balanced across format ---------------------------
    for topic, n in topic_quotas.items():
        if not n:
            continue
        # topic must be evident from the URL or title, and the document must
        # describe UniTn rather than a double-degree partner
        pool = [r for r in student if topic in r["_topics"] and not r["_partner"]]
        want_pdf = int(round(n * PDF_SHARE))
        got = take([r for r in pool if r["_fmt"] == "pdf"], want_pdf, topic)
        got += take([r for r in pool if r["_fmt"] == "html"], n - len(got), topic)
        # if one format is short, top up from the other rather than under-filling
        if len(got) < n:
            got += take(pool, n - len(got), topic)
        filled[topic] = (len(got), n)

    # --- bilingual --------------------------------------------------------
    by_url = {r["url"]: r for r in student}
    seen_pair, bipool, bimap = set(), [], {}
    for a, b in reciprocal_pairs(rows).items():
        key = frozenset((a, b))
        if key in seen_pair or a not in by_url or b not in by_url:
            continue
        seen_pair.add(key)
        it = a if (by_url[a].get("lang") or "").lower() == "it" else b
        en = b if it == a else a
        bipool.append(by_url[it])
        bimap[it] = [en]
    n = structural_quotas.get("bilingual", 0)
    filled["bilingual"] = (len(take(bipool, n, "bilingual", also=bimap)), n)

    # --- temporal ---------------------------------------------------------
    years: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    for r in student:
        y = r.get("effective_year")
        if not isinstance(y, int) or _PERSONAL_TREE.search(urlsplit(r["url"]).path):
            continue
        years[year_group(r["url"])].setdefault(y, r)
    newest, stale = [], {}
    for v in years.values():
        if len(v) < 2:
            continue
        cur = v[max(v)]
        newest.append(cur)
        stale[cur["url"]] = [v[y]["url"] for y in sorted(v, reverse=True)[1:]]
    n = structural_quotas.get("temporal", 0)
    filled["temporal"] = (len(take(newest, n, "temporal", distract=stale)), n)

    # --- nearduplicate ----------------------------------------------------
    by_title: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in student:
        if _PERSONAL_TREE.search(urlsplit(r["url"]).path):
            continue
        tk = title_key(r)
        if len(tk) < 12:
            continue
        by_title[(urlsplit(r["url"]).netloc.lower(), tk, r.get("lang") or "")].append(r)
    clusters = []
    for members in by_title.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda m: (m.get("effective_year") or 0,
                                    len(m.get("text") or "")), reverse=True)
        base = shingle(members[0].get("text", ""))
        sims = [m for m in members[1:]
                if 0.55 < jaccard(base, shingle(m.get("text", ""))) < 0.99]
        if sims:
            clusters.append((members[0], [m["url"] for m in sims]))
    n = structural_quotas.get("nearduplicate", 0)
    filled["nearduplicate"] = (len(take(
        [c[0] for c in clusters], n, "nearduplicate",
        distract={c[0]["url"]: c[1] for c in clusters})), n)

    # --- ocr --------------------------------------------------------------
    # Of the 668 documents the OCR run recovered, only 11 carry a student topic
    # in their title or path (6 admissions, 4 services, 1 course). The rest are
    # governance minutes (107), research quaderni (66), staff/union agreements
    # (51) and job-competition material - lavoraconnoi and teseo are the two
    # largest hosts. An unfiltered sample lands on "Fondo ex Art. 63 - anno
    # 2022.pdf" and CPDS meeting minutes, which no student would ever ask about.
    #
    # So this stratum is deliberately small and is a *retrieval-completeness*
    # test, not a student-QA one: it answers "did the recovered text become
    # reachable at all", which is the claim the OCR chapter actually makes.
    # Topical documents are drawn first, then student-facing ones to fill.
    ocr = [r for r in answerable
           if r.get("extractor") == "tesseract" and not r.get("ocr_suspect")
           and len(r.get("text") or "") > 3000 and is_student_facing(r)]
    for r in ocr:
        r.setdefault("_topics", topics_of(r))
        r.setdefault("_weak", weak_topics_of(r))
        r.setdefault("_fmt", fmt_of(r))
        r.setdefault("_partner", bool(_PARTNER_DOC.search(r.get("url") or "")))
    ocr = [r for r in ocr if not r["_partner"]]
    n = structural_quotas.get("ocr", 0)
    got = take([r for r in ocr if r["_topics"]], n, "ocr")
    got += take([r for r in ocr if not r["_topics"]], n - len(got), "ocr")
    filled["ocr"] = (len(got), n)

    # --- trap -------------------------------------------------------------
    traps = [r for r in student
             if any(t in (r.get("title") or "").lower() for t in _TRAP_TERMS)]
    n = structural_quotas.get("trap", 0)
    filled["trap"] = (len(take(traps, n, "trap")), n)

    return out, filled


# --------------------------------------------------------------------------

SHORTFALL_HINTS = {
    "ocr": "no extractor=tesseract rows - this is the pre-OCR corpus, so the "
           "668 recovered documents cannot be tested",
    "housing": "only ~278 documents mention accommodation at all",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("dataset.ocr.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("evalprep/cand"))
    ap.add_argument("--quota", action="append", default=[], metavar="NAME=N")
    ap.add_argument("--seed", type=int, default=20260820,
                    help="fixed so the selection is reproducible and citable")
    args = ap.parse_args()

    tq, sq = dict(TOPIC_QUOTAS), dict(STRUCTURAL_QUOTAS)
    for q in args.quota:
        k, _, v = q.partition("=")
        k = k.strip()
        (tq if k in tq else sq)[k] = int(v)

    if not args.corpus.exists():
        print(f"[eval] corpus not found: {args.corpus}", file=sys.stderr)
        return 2

    rows, dropped = load(args.corpus)
    print(f"[eval] {args.corpus.name}: {len(rows):,} index-eligible documents "
          f"({sum(dropped.values()):,} dropped)")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    faq = harvest_faq(rows)
    faq_path = args.out.with_name(args.out.name + "_faq.csv")
    faq_cols = ["harvested_question", "rewrite_as_a_student_would_ask",
                "acceptable_urls", "lang", "topic", "page_title",
                "about_other_institution", "keep"]
    with open(faq_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=faq_cols)
        w.writeheader()
        w.writerows(faq)
    foreign = sum(1 for r in faq if r["about_other_institution"])
    print(f"[eval] harvested {len(faq)} questions from "
          f"{len({r['acceptable_urls'] for r in faq})} FAQ pages -> {faq_path}")
    if foreign:
        print(f"         {foreign} come from a partner institution's FAQ, flagged "
              f"'about_other_institution' - not UniTn policy")

    targets, filled = pick_targets(rows, tq, sq, args.seed)
    tgt_path = args.out.with_name(args.out.name + "_targets.csv")
    with open(tgt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(targets[0].keys()))
        w.writeheader()
        w.writerows(targets)

    fmt_mix = collections.Counter(t["format"] for t in targets)
    lang_mix = collections.Counter(t["lang"] for t in targets)

    lines = [f"corpus: {args.corpus}", f"seed: {args.seed}",
             f"index-eligible documents: {len(rows):,}", "",
             "dropped at load (mirrors data.load_documents):"]
    lines += [f"  {v:7,}  {k}" for k, v in dropped.most_common()]
    lines += ["", f"targets written: {len(targets)}",
              f"format mix: {dict(fmt_mix)}",
              f"language mix: {dict(lang_mix)}", "",
              "stratum          filled/requested"]
    short = []
    for k, n in list(tq.items()) + list(sq.items()):
        got, want = filled.get(k, (0, n))
        if got != want:
            short.append((k, got, want))
        lines.append(f"  {k:<16} {got:>3}/{want:<3}" + ("   <-- SHORT" if got != want else ""))
    if short:
        lines += ["", "SHORTFALLS - a stratum that cannot fill is telling you",
                  "something about the corpus, not about the quota:"]
        lines += [f"  {k}: {got} of {want}. {SHORTFALL_HINTS.get(k, '')}"
                  for k, got, want in short]
    rep_path = args.out.with_name(args.out.name + "_report.txt")
    rep_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[eval] {len(targets)} target documents -> {tgt_path}")
    print(f"         format {dict(fmt_mix)}   language {dict(lang_mix)}")
    for k, n in list(tq.items()) + list(sq.items()):
        got, want = filled.get(k, (0, n))
        print(f"         {k:<16} {got:>3}/{want}" + ("   SHORT" if got != want else ""))
    print(f"[eval] report -> {rep_path}")

    print("\nNext:")
    print("  1. Fill the blank question/gold_answer columns by hand. Rewrite")
    print("     harvested FAQ questions in your own words - verbatim ones share")
    print("     vocabulary with the page and measure paraphrase matching rather")
    print("     than retrieval.")
    print("  2. Add out-of-scope questions with an EMPTY acceptable_urls, plus")
    print("     prompt-injection cases. They have no gold document and must be")
    print("     scored on refusal, not on hit@k.")
    print("  3. hit_at_k() takes a single target_url. It must accept the")
    print("     acceptable_urls list, or every bilingual target reports a false")
    print("     failure when dedup surfaces the sibling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
