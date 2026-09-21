#!/usr/bin/env python3
"""Scrape the UniTn Course Catalogue's REST API into crawl-schema JSONL.

https://unitn.coursecatalogue.cineca.it is a client-rendered SPA (Angular):
the crawler's one record for it is HTTP 200 with 4 characters of body, and the
3,576 URLs linked from other pages (mostly `/af/?ad=...` course-unit links)
were never actually fetched with real content. Worse, those linked URLs carry
a different id (`ad=<numeric>`) than the one the SPA's own API expects
(`insegnamento=<pdsId>_<x>_<y>`) - the two do not correspond, so link-following
can never recover this content, no matter how the crawler is fixed.

Enumeration through the API itself does, in five steps, each supplying every
parameter the next one needs:

    gruppi/{anno}                                  12 top-level categories
      -> corsi?anno=&gruppo=&minimal=true           every programme in a group
           -> corso/{anno}/{corso_cod}              programme detail + codicione
                -> corso-offerta/{corso_cod}?...     every course unit's exact id
                     -> insegnamento?...             the syllabus itself

corso-offerta's `cod` field on each course unit *is* the composite id the
insegnamento endpoint calls `insegnamento=` - found by comparing a captured
insegnamento request against this response - so there is no id reconciliation
step and no dependency on out_links at all.

As of 2026-09-04 syllabus content is genuinely populated: confirmed across two
unrelated departments (Giurisprudenza, DISI) with dataModifica timestamps
minutes apart, which reads like a scheduled bulk publish rather than a slow
rollout. That was NOT true a few hours earlier in the same investigation, and
it has not been verified exhaustively here either - the run summary below
reports how many enumerated units still come back with no syllabus text, so a
real run tells you the current coverage rather than assuming it.

Usage:
    python3 scripts/scrape_coursecatalogue.py --only-corso-cod 10818 --limit 5  # smoke test
    python3 scripts/scrape_coursecatalogue.py                                   # full run
    python3 scripts/scrape_coursecatalogue.py --resume                          # continue after Ctrl-C

Output is crawl-schema JSONL - the same fields load_documents() already
expects (url, text, lang, low_content, duplicate_of, academic_year, ...) - so
it drops into the existing pipeline without changes there. It is written
incrementally (flushed after every course unit) so an interrupted run keeps
its progress, and it is never merged into the corpus by this script - see
merge_coursecatalogue.py for that, kept separate deliberately (see its
docstring).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

BASE = "https://unitn.coursecatalogue.cineca.it"
API = f"{BASE}/api/v1"
DEFAULT_OUT = Path("coursecatalogue.jsonl")


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class ApiClient:
    """Thin GET wrapper: retries, a fixed delay between calls, one session.

    A full run makes roughly 2,700 calls (12 groups + ~120 programmes x 2 +
    ~2,500 syllabi) - polite, predictable pacing matters more here than speed.
    """

    def __init__(self, delay: float = 0.2, timeout: float = 20.0, retries: int = 3):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "unitn-rag-corpus-builder/1.0 (student research project; "
                          "scraping the public course catalogue API)",
            "Accept": "application/json",
        })
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self.calls = 0
        self.errors = 0

    def get(self, path: str, params: dict | None = None):
        url = f"{API}{path}"
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                self.calls += 1
                if r.status_code == 200:
                    time.sleep(self.delay)
                    return r.json()
                last_exc = RuntimeError(f"HTTP {r.status_code} for {r.url}")
            except requests.RequestException as exc:
                last_exc = exc
            time.sleep(self.delay * (2 ** attempt))
        self.errors += 1
        print(f"[scrape] giving up on {path} params={params}: {last_exc}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# HTML -> plain text
# --------------------------------------------------------------------------
# testiTotali's rich-text fields (contenuti_it, testi_it, ...) come back as
# HTML fragments (<p>, <ul><li>, <strong>, <a href>). clean_text() downstream
# only collapses whitespace - it does not strip tags - so that has to happen
# here, before the text ever reaches the rest of the pipeline.

_BLOCK_RE = re.compile(r"</?(p|div|li|br|h[1-6]|tr)\b[^>]*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[^\S\n]+")
_BLANK_RE = re.compile(r"\n{3,}")


def html_to_text(s: str | None) -> str:
    if not s:
        return ""
    s = _BLOCK_RE.sub("\n", s)
    s = _TAG_RE.sub("", s)
    s = html.unescape(s)
    s = _WS_RE.sub(" ", s)
    s = _BLANK_RE.sub("\n\n", s)
    return "\n".join(line.strip() for line in s.split("\n")).strip()


# --------------------------------------------------------------------------
# Step 1-2: enumerate programmes
# --------------------------------------------------------------------------

def fetch_groups(api: ApiClient, anno: int) -> list[dict]:
    return api.get(f"/gruppi/{anno}") or []


def fetch_programmes(api: ApiClient, anno: int, gruppo_cod: str) -> list[dict]:
    """Every programme (a cdsSub record) nested under one top-level group.

    The nesting is group -> subgroup (category) -> cds (degree class) ->
    cdsSub (the actual programme/year record, keyed by `cod` == corso_cod).
    Flattened here because nothing downstream needs the category grouping.
    """
    resp = api.get("/corsi", params={"anno": anno, "gruppo": gruppo_cod, "minimal": "true"})
    if not resp:
        return []
    # This endpoint's wrapping is not consistent across callers: browser-side
    # fetch() saw the group object unwrapped ({"cod": ..., "subgroups": [...]}),
    # while a plain `requests` client saw the identical object wrapped in a
    # one-element list ([{"cod": ..., "subgroups": [...]}]) - and a genuinely
    # empty group (Microcredenziali, Scuole di specializzazione area
    # psicologica, etc. all show count 0 on the homepage) is an empty list.
    # Normalising to "a list of group objects" handles all three without
    # needing to know which one caused it.
    items = resp if isinstance(resp, list) else [resp]
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for subgroup in item.get("subgroups") or []:
            for cds in subgroup.get("cds") or []:
                for sub in cds.get("cdsSub") or []:
                    out.append(sub)
    return out


# --------------------------------------------------------------------------
# Step 3-4: per programme -> course units
# --------------------------------------------------------------------------

def fetch_codicione(api: ApiClient, anno: int, corso_cod: str) -> dict | None:
    """Programme detail. Needed for `codicione`, the one field corso-offerta
    requires that isn't already in the minimal programme listing - and, as a
    side benefit, it also carries the programme-level overview text used by
    build_programme_records() below."""
    resp = api.get(f"/corso/{anno}/{corso_cod}")
    if isinstance(resp, list):
        resp = resp[0] if resp else None
    return resp


def fetch_units(api: ApiClient, corso_cod: str, codicione: str,
                 ordinamento_aa: int, sede_cod: str) -> list[dict]:
    """Every course unit taught in one programme/year, deduplicated by `cod`.

    corso-offerta's `cod` field (e.g. "51209_656002_90053") *is* the composite
    id the insegnamento endpoint calls `insegnamento=` - this is what makes
    enumeration possible without the broken ad=/insegnamento= id mapping that
    following out_links would have required.
    """
    resp = api.get(f"/corso-offerta/{corso_cod}",
                    params={"codicione": codicione, "annoOrdinamento": ordinamento_aa,
                            "sede": sede_cod})
    if not resp:
        return []
    # Same wrapping inconsistency as fetch_programmes() - seen wrapping a
    # single-object response in a one-element list from some callers/networks.
    if isinstance(resp, list):
        resp = resp[0] if resp and isinstance(resp[0], dict) else {}
    seen: dict[str, dict] = {}
    for year_block in resp.values():
        if not isinstance(year_block, dict):
            continue
        for period in year_block.values():
            if not isinstance(period, dict):
                continue
            for att in period.get("attivita") or []:
                cod = att.get("cod")
                if cod and cod not in seen:
                    seen[cod] = att
    return list(seen.values())


def fetch_syllabus(api: ApiClient, unit: dict) -> dict | None:
    resp = api.get("/insegnamento", params={
        "anno": unit["aa"],
        "insegnamento": unit["cod"],
        "ordinamento_aa": unit["ordinamento_aa"],
        "af_percorso": unit["af_percorso_id"],
        "corso_cod": unit["corso_cod"],
        "corso_aa": unit["aa"],
    })
    if isinstance(resp, list):
        resp = resp[0] if resp else None
    return resp


def fetch_fraction_syllabus(api: ApiClient, unit: dict, fraction: dict) -> dict | None:
    """The syllabus for one frazione (partition) of a split course.

    A large first-year course is often taught as several parallel sections,
    split by student surname (e.g. "Iniziali cognome A-E" / "F-O" / "P-Z"),
    each with its own professor and its own syllabus text. The *aggregate*
    call (fetch_syllabus, keyed by the composite `cod`) genuinely returns
    `testiTotali: []` for these - confirmed live, not a timing issue - because
    the content was never attached at that level; it lives only under each
    partition's own id (e.g. "25840/1-FO"), fetched here with the same
    /insegnamento endpoint plus `adCodFraz` and the partition's own `cod` in
    place of the unit's composite one. `fraction` is one entry of the
    aggregate response's own `frazioni` array - see build_unit_records().
    """
    resp = api.get("/insegnamento", params={
        "anno": unit["aa"],
        "insegnamento": fraction["cod"],
        "ordinamento_aa": unit["ordinamento_aa"],
        "af_percorso": unit["af_percorso_id"],
        "corso_cod": unit["corso_cod"],
        "corso_aa": unit["aa"],
        "adCodFraz": fraction.get("adCod") or unit.get("adCod"),
    })
    if isinstance(resp, list):
        resp = resp[0] if resp else None
    return resp


# --------------------------------------------------------------------------
# Record assembly - course units
# --------------------------------------------------------------------------

# (testiTotali key prefix, Italian section label, English section label)
_SECTIONS = [
    ("obiettivi_formativi", "Obiettivi formativi", "Learning outcomes"),
    ("prerequisiti", "Prerequisiti", "Prerequisites"),
    ("contenuti", "Programma", "Syllabus"),
    ("metodi_didattici_est", "Metodi didattici", "Teaching methods"),
    ("verifica_apprendimento", "Modalita di verifica dell'apprendimento", "Assessment methods"),
    ("testi", "Testi di riferimento", "Reading list"),
    ("altro", "Altre informazioni", "Other information"),
]


def render_syllabus(detail: dict, lang: str) -> tuple[str, str]:
    """(title, body) for one language from one insegnamento record.

    Falls back to the Italian field when a specific translated field is
    genuinely empty rather than emitting a blank section - some units have
    IT-only text this soon after publish, and des_it/en are frequently
    identical anyway (flagged separately: this makes some EN "translations"
    a de facto cross-language retrieval test, same shape as the Job Guidance
    FAQ finding earlier in this project).
    """
    tt = (detail.get("testiTotali") or [{}])[0] or {}

    def field(base: str) -> str:
        v = tt.get(f"{base}_{lang}")
        if not v and lang != "it":
            v = tt.get(f"{base}_it")
        return html_to_text(v)

    des = detail.get(f"des_{lang}") or detail.get("des_it") or ""
    corso_des = detail.get(f"corso_des_{lang}") or detail.get("corso_des_it") or ""
    dip_des = detail.get(f"dip_des_{lang}") or detail.get("dip_des_it") or ""
    docenti = ", ".join(d.get("des", "") for d in (detail.get("docenti") or []) if d.get("des"))
    tipo_esa = detail.get(f"tipoEsaDes_{lang}") or ""
    valutazione = detail.get(f"valutazione_{lang}") or ""
    periodo = detail.get(f"periodo_didattico_{lang}") or ""
    lingua_des = detail.get(f"lingua_des_{lang}") or ""

    header = f"{des} - {corso_des}" + (f" ({dip_des})" if dip_des else "")
    meta_bits = []
    if detail.get("crediti") is not None:
        meta_bits.append(f"{detail['crediti']} CFU" if lang == "it" else f"{detail['crediti']} ECTS")
    if detail.get("ssd"):
        meta_bits.append(f"SSD {detail['ssd']}")
    if lingua_des:
        meta_bits.append(lingua_des)
    if periodo:
        meta_bits.append(periodo)
    if docenti:
        meta_bits.append(("Docenti: " if lang == "it" else "Teaching staff: ") + docenti)
    if tipo_esa or valutazione:
        meta_bits.append(f"{tipo_esa} ({valutazione})" if valutazione else tipo_esa)

    parts = [header]
    if meta_bits:
        parts.append(" | ".join(meta_bits))
    for key, label_it, label_en in _SECTIONS:
        body = field(key)
        if body:
            parts.append(f"{label_it if lang == 'it' else label_en}\n{body}")

    return header, "\n\n".join(p for p in parts if p).strip()


def build_detail_url(anno: int, corso_cod: str, cod: str, aa: str, ordinamento_aa: int,
                      af_percorso_id: str, schema_id, lang: str,
                      ad_cod_fraz: str | None = None) -> str:
    """The real deep-link route - confirmed by navigating to it directly (a
    fresh page load, not just in-app SPA state), not guessed from the app's
    JS bundle. Shared by both the aggregate unit id ("51209_656019_90054")
    and a frazione's own id ("25840/1-FO") - the latter contains a literal
    "/" that the app itself percent-encodes in the path segment (confirmed
    from a URL captured directly out of the browser), hence `quote(..., safe="")`
    rather than the un-encoded string. `adCodFraz` is only present, and only
    meaningful, for a frazione URL. `lang` is a query param added purely for
    our own dedup / language-detection purposes (detect_language_from_url()
    in text.py already recognises lang=/language=/locale= and doc_group_id()
    already strips it when grouping translations) - the live page ignores it
    and shows whichever language its own UI toggle was last set to, but the
    route and content underneath are real and clickable either way.
    """
    encoded_cod = quote(str(cod), safe="")
    url = (f"{BASE}/corsi/{anno}/{corso_cod}/insegnamenti/{aa}/{encoded_cod}/"
           f"{ordinamento_aa}/{af_percorso_id}?coorte={anno}&schemaid={schema_id}")
    if ad_cod_fraz:
        url += f"&adCodFraz={ad_cod_fraz}"
    url += f"&lang={lang}"
    return url


def base_record(url: str, title: str, text: str, lang: str, doc_type: str,
                 department: str | None, academic_year: str, last_modified: str | None,
                 fetched_at: str, extra: dict) -> dict:
    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    rec = {
        "url": url,
        "fetched_at": fetched_at,
        "http_status": 200,
        "content_type": "application/json",
        "etag": None,
        "last_modified": last_modified,
        # Left unset deliberately: resolve_effective_year() checks academic_year
        # first and will derive the correct year from it. Setting a redundant
        # effective_year here would just be one more place for it to drift out
        # of sync with academic_year later.
        "effective_year": None,
        "academic_year": academic_year,
        "lang": lang,
        "lang_source": "api_field",
        "doc_type": doc_type,
        "department": department,
        "extractor": "coursecatalogue_api",
        "title": title,
        "text": text,
        "text_len": len(text),
        "text_sha256": text_sha,
        "content_sha256": text_sha,
        "raw_html_sha256": None,
        "out_links": [],
        "low_content": len(text) < 150,
        "nav_ratio": 0.0,
        "boilerplate": False,
        "duplicate_of": None,
        "changed": True,
        "note": "coursecatalogue_api_scrape",
    }
    rec.update(extra)
    return rec


def build_unit_records(anno: int, corso_cod: str, unit: dict, detail: dict,
                        fetched_at: str) -> list[dict]:
    records = []
    academic_year = f"{unit['aa']}/{int(unit['aa']) + 1}"
    for lang in ("it", "en"):
        title, text = render_syllabus(detail, lang)
        if not text:
            continue
        records.append(base_record(
            url=build_detail_url(anno, corso_cod, unit["cod"], unit["aa"], unit["ordinamento_aa"],
                                  unit["af_percorso_id"], unit.get("schemaId", ""), lang),
            title=title, text=text, lang=lang, doc_type="syllabus",
            department=detail.get(f"dip_des_{lang}") or detail.get("dip_des_it"),
            academic_year=academic_year, last_modified=detail.get("dataModifica"),
            fetched_at=fetched_at,
            extra={
                "coursecatalogue_unit_cod": unit["cod"],
                "coursecatalogue_corso_cod": corso_cod,
            },
        ))
    return records


def build_fraction_records(anno: int, corso_cod: str, unit: dict, fraction: dict,
                            detail: dict, fetched_at: str) -> list[dict]:
    """Same shape as build_unit_records(), for one frazione (partition) of a
    split course. `detail` here is the fraction's OWN /insegnamento response
    (from fetch_fraction_syllabus), which carries its own des_it/en, docenti,
    testiTotali etc. - not the parent aggregate's - so this is not a thin
    wrapper around build_unit_records(): the URL needs the fraction's own
    `cod` and `adCod` while every other positional piece (aa, ordinamento_aa,
    af_percorso_id, schemaId) comes from the parent unit, since a frazione
    entry does not carry those itself (see fetch_syllabus's docstring).
    """
    records = []
    academic_year = f"{unit['aa']}/{int(unit['aa']) + 1}"
    ad_cod_fraz = fraction.get("adCod") or unit.get("adCod")
    for lang in ("it", "en"):
        title, text = render_syllabus(detail, lang)
        if not text:
            continue
        records.append(base_record(
            url=build_detail_url(anno, corso_cod, fraction["cod"], unit["aa"], unit["ordinamento_aa"],
                                  unit["af_percorso_id"], unit.get("schemaId", ""), lang,
                                  ad_cod_fraz=ad_cod_fraz),
            title=title, text=text, lang=lang, doc_type="syllabus",
            department=detail.get(f"dip_des_{lang}") or detail.get("dip_des_it"),
            academic_year=academic_year, last_modified=detail.get("dataModifica"),
            fetched_at=fetched_at,
            extra={
                "coursecatalogue_unit_cod": fraction["cod"],
                "coursecatalogue_corso_cod": corso_cod,
                "coursecatalogue_parent_unit_cod": unit["cod"],
                "coursecatalogue_partition": fraction.get("domPartCod"),
            },
        ))
    return records


# --------------------------------------------------------------------------
# Record assembly - programme overview ("Scheda del corso")
# --------------------------------------------------------------------------
# Fetched for free alongside `codicione` in fetch_codicione(), and it is
# genuinely new content: this QUADRO_A text (admission requirements, the
# programme's own stated learning objectives) does not exist anywhere in the
# marketing-style corsi.unitn.it pages already in the corpus. --skip-programmes
# turns this off if it turns out not to be wanted.

def render_programme(detail: dict, lang: str) -> tuple[str, str]:
    """(title, body) for one language from one corso record.

    Italian drives the section list, not the requested language: the EN array
    (`programma_testi_obiettivi_en`) can exist as a non-empty list whose items
    simply have an empty `carattTesto` - a present-but-blank translation - and
    testing the array's truthiness (as an earlier version of this function
    did) treats that as "EN has this section" and never falls back, producing
    a near-empty English "Scheda del corso" next to a ~20,000-character
    Italian one. Matching EN to IT section-by-section via `tipoCarattCod` and
    falling back to the Italian body per section, rather than for the array as
    a whole, is what actually catches a blank translation.
    """
    des = detail.get(f"des_{lang}") or detail.get("des_it") or ""
    tipo_corso = detail.get(f"tipo_corso_des_{lang}") or ""
    dur = detail.get(f"durata_{lang}") or ""
    classe = detail.get(f"classe_{lang}") or ""
    header = f"{des} - {tipo_corso}" if tipo_corso else des
    meta = " | ".join(x for x in (dur, classe) if x)

    it_sections = sorted(detail.get("programma_testi_obiettivi_it") or [],
                          key=lambda t: t.get("ordine", 0))
    en_by_key = {t.get("tipoCarattCod"): t for t in (detail.get("programma_testi_obiettivi_en") or [])}

    parts = [header]
    if meta:
        parts.append(meta)
    for sec in it_sections:
        if lang == "it":
            title = sec.get("carattTitolo") or ""
            body = html_to_text(sec.get("carattTesto") or "")
        else:
            en_sec = en_by_key.get(sec.get("tipoCarattCod")) or {}
            body = html_to_text(en_sec.get("carattTesto") or "") or html_to_text(sec.get("carattTesto") or "")
            title = en_sec.get("carattTitolo") or sec.get("carattTitolo") or ""
        if body:
            parts.append(f"{title}\n{body}" if title else body)

    return header, "\n\n".join(p for p in parts if p).strip()


def programme_url(anno: int, corso_cod: str, lang: str) -> str:
    return f"{BASE}/corsi/{anno}/{corso_cod}?lang={lang}"


def build_programme_records(anno: int, corso_cod: str, detail: dict, fetched_at: str) -> list[dict]:
    records = []
    academic_year = f"{anno}/{anno + 1}"
    for lang in ("it", "en"):
        title, text = render_programme(detail, lang)
        if not text:
            continue
        records.append(base_record(
            url=programme_url(anno, corso_cod, lang),
            title=title, text=text, lang=lang, doc_type="course",
            department=None, academic_year=academic_year, last_modified=None,
            fetched_at=fetched_at,
            extra={"coursecatalogue_corso_cod": corso_cod},
        ))
    return records


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anno", type=int, default=2026,
                    help="cohort year - the catalogue currently offers only 2026/2027")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--delay", type=float, default=0.2, help="seconds between API calls")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after enumerating this many course units (smoke testing)")
    ap.add_argument("--only-corso-cod", default=None,
                    help="scrape a single programme by its corso_cod (smoke testing)")
    ap.add_argument("--skip-programmes", action="store_true",
                    help="skip the programme-level 'Scheda del corso' documents")
    ap.add_argument("--resume", action="store_true",
                    help="skip course units already present in --out and append to it")
    args = ap.parse_args()

    if args.out.exists() and not args.resume:
        print(f"[scrape] {args.out} already exists and --resume was not given - refusing to "
              f"overwrite. Pass --resume to continue an interrupted run, or move/remove the "
              f"file to start over.", file=sys.stderr)
        return 1

    done_units: set[str] = set()
    mode = "w"
    if args.resume and args.out.exists():
        mode = "a"
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                u = r.get("coursecatalogue_unit_cod")
                if u:
                    done_units.add(u)
        print(f"[scrape] resuming: {len(done_units)} course units already in {args.out}")

    api = ApiClient(delay=args.delay)
    fetched_at = datetime.now(timezone.utc).isoformat()

    print(f"[scrape] anno={args.anno}")
    groups = fetch_groups(api, args.anno)
    print(f"[scrape] {len(groups)} top-level groups")

    programmes: dict[str, dict] = {}
    for g in groups:
        for p in fetch_programmes(api, args.anno, g["cod"]):
            programmes[p["cod"]] = p
    if args.only_corso_cod:
        programmes = {k: v for k, v in programmes.items() if k == args.only_corso_cod}
    print(f"[scrape] {len(programmes)} programmes to process")

    n_units = n_written = n_empty = n_programme_docs = n_fractions = 0
    out_f = open(args.out, mode, encoding="utf-8")
    try:
        for i, (corso_cod, prog) in enumerate(programmes.items(), 1):
            detail = fetch_codicione(api, args.anno, corso_cod)
            if not detail or not detail.get("codicione"):
                print(f"[scrape] [{i}/{len(programmes)}] {corso_cod}: no codicione, skipping",
                      file=sys.stderr)
                continue

            if not args.skip_programmes:
                for rec in build_programme_records(args.anno, corso_cod, detail, fetched_at):
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_programme_docs += 1

            sede_cod = prog.get("sede_cod")
            ordinamento_aa = prog.get("ordinamento_aa", args.anno)
            units = fetch_units(api, corso_cod, detail["codicione"], ordinamento_aa, sede_cod)
            label = prog.get("des_it") or corso_cod
            print(f"[scrape] [{i}/{len(programmes)}] {label}: {len(units)} course units")

            for unit in units:
                if args.limit and n_units >= args.limit:
                    break
                n_units += 1

                # Always fetched, even on --resume: it is the only way to
                # discover a unit's frazioni (see fetch_syllabus's docstring),
                # and a split course's aggregate id is never in done_units
                # anyway (build_unit_records() only writes it when it has
                # text, and the aggregate testiTotali for a split course is
                # always empty). The done_units checks below still make the
                # per-record writes idempotent under --resume.
                syllabus = fetch_syllabus(api, unit)
                if not syllabus:
                    continue

                if unit["cod"] not in done_units:
                    tt = (syllabus.get("testiTotali") or [{}])[0]
                    if not (tt.get("contenuti_it") or tt.get("obiettivi_formativi_it")):
                        n_empty += 1
                    for rec in build_unit_records(args.anno, corso_cod, unit, syllabus, fetched_at):
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_written += 1

                for fraction in syllabus.get("frazioni") or []:
                    fcod = fraction.get("cod")
                    if not fcod or fcod in done_units:
                        continue
                    n_fractions += 1
                    fdetail = fetch_fraction_syllabus(api, unit, fraction)
                    if not fdetail:
                        continue
                    ftt = (fdetail.get("testiTotali") or [{}])[0]
                    if not (ftt.get("contenuti_it") or ftt.get("obiettivi_formativi_it")):
                        n_empty += 1
                    for rec in build_fraction_records(args.anno, corso_cod, unit, fraction,
                                                       fdetail, fetched_at):
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_written += 1

                out_f.flush()

            if args.limit and n_units >= args.limit:
                break
    finally:
        out_f.close()

    print(f"[scrape] {n_units} course units enumerated, {n_fractions} frazioni (split-section "
          f"partitions) followed, {n_empty} still have no syllabus text")
    print(f"[scrape] {n_written} syllabus documents written ({n_programme_docs} programme-level)")
    print(f"[scrape] {api.calls} API calls, {api.errors} failed")
    print(f"[scrape] wrote {args.out}")
    print("[scrape] this is a standalone file - merge it with scripts/merge_coursecatalogue.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
