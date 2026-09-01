"""Where do these documents state their own date, and what does the crawl say?

Four confirmed cases where the year in the upload path is not the year the
document is about. Before changing the resolution order, look at exactly what
evidence each one carries and where in the text it sits - a rule that only reads
the first 500 characters will miss a footer.
"""
import json, re, sys
from urllib.parse import unquote, urlsplit

CORPUS = "dataset.v2.jsonl"

WANT = [
    ("energy regolamenti", re.compile(r"regolamento-didattico-lm-(energy-engineering|ingegneria-energetica)", re.I)),
    ("CEILS guidelines",   re.compile(r"guidelines_trasfer_b-comparative", re.I)),
    ("travel regulation",  re.compile(r"missioni", re.I)),
    ("a.a. 2007-2008",     re.compile(r"guida.?facolt|2007.?2008", re.I)),
]

PATTERNS = {
    "DR/decree date  'del 13/07/2026'": re.compile(r"\bdel\s+\d{1,2}[/.]\d{1,2}[/.]((?:19|20)\d{2})"),
    "'Anno Accademico 2007-2008'":      re.compile(r"[Aa]nno\s+[Aa]ccademico\s+((?:19|20)\d{2})\s*[/-]\s*((?:19|20)?\d{2})"),
    "'last updated ... 2022'":          re.compile(r"(?:last\s+updat\w*|ultimo\s+aggiornamento)[^.\n]{0,40}?((?:19|20)\d{2})", re.I),
    "upload path /files/YYYY-MM/":      re.compile(r"/(?:files|sites)/[^?]*?/((?:19|20)\d{2})-\d{2}/"),
}

rows = []
for line in open(CORPUS, encoding="utf-8"):
    r = json.loads(line)
    u = unquote(r.get("url") or "")
    t = r.get("text") or ""
    ti = r.get("title") or ""
    for label, rx in WANT:
        if rx.search(u) or rx.search(ti):
            rows.append((label, r))
            break

print("matched %d records\n" % len(rows))
for label, r in sorted(rows, key=lambda x: (x[0], x[1].get("url") or "")):
    u = unquote(r.get("url") or "")
    t = r.get("text") or ""
    if len(t) < 400:
        continue
    print("=" * 78)
    print("%s  |  crawl effective_year=%s  academic_year=%s  len=%d"
          % (label, r.get("effective_year"), r.get("academic_year"), len(t)))
    print("  %s" % u[-108:])
    print("  title: %s" % (r.get("title") or "")[:100])
    print("  last_modified: %s" % r.get("last_modified"))
    for name, rx in PATTERNS.items():
        hay = u if name.startswith("upload") else t
        m = rx.search(hay)
        if m:
            pos = m.start()
            where = ("char %d of %d (%.0f%% in)" % (pos, len(hay), 100 * pos / max(1, len(hay)))
                     if not name.startswith("upload") else "in URL")
            print("    %-34s -> %-9s  %s" % (name, m.group(1), where))
            if not name.startswith("upload"):
                print("        ...%s..." % " ".join(hay[max(0, pos-60):pos+90].split()))
