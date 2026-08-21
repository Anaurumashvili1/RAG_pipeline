"""Is doc_type a coherent axis? No - it conflates format, genre and topic."""
import json, collections, re
from urllib.parse import urlsplit, unquote

TOPIC = {
 "admissions": r"ammission|immatricol|iscriz|admission|enrol|bando.?di.?ammissione|call.?for.?application|test.?d.?ingresso|tolc",
 "tuition":    r"\btasse\b|tuition|contribut|esoner|rimbors|fee|isee|rate.?e.?scadenz",
 "scholarship":r"\bborse?\b|scholarship|assegno|premio|premi\b|grant|bando.?borsa",
 "housing":    r"alloggi|accommodation|residenz|studentato|opera.?universitaria|mensa|canteen",
 "course":     r"regolamento.?didattico|manifesto.?degli.?studi|piano.?di.?studi|scheda.?del.?corso|study.?plan|course.?catalogue|insegnament",
 "graduation": r"laurea|graduation|tesi\b|thesis|esame.?di.?laurea|proclamazione",
 "service":    r"bibliotec|library|centro.?linguistic|language.?centre|test.?center|job.?guidance|tirocin|internship|erasmus|mobilit",
}
PAT = {k: re.compile(v, re.I) for k, v in TOPIC.items()}

fmt = collections.Counter()
grid = collections.defaultdict(collections.Counter)
lost = collections.Counter()
for l in open("evalprep/slim_index.jsonl", encoding="utf-8"):
    r = json.loads(l)
    dt = r.get("doc_type") or "None"
    is_pdf = (r.get("content_type") == "application/pdf")
    blob = unquote(r["url"]) + " " + (r.get("title") or "") + " " + (r.get("snippet") or "")[:400]
    topics = [k for k, p in PAT.items() if p.search(blob)]
    fmt[("pdf" if is_pdf else "html", dt)] += 1
    for t in topics:
        grid[t][dt] += 1
        if is_pdf:
            lost[t] += 1

print("=== doc_type conflates three axes ===")
print("format:  page(22167) pdf(16391)      <- how it was served")
print("genre:   event research news people   <- what kind of thing it is")
print("topic:   admissions course scholarship tuition housing service phd\n")

print("=== keyword-derived topic  x  doc_type as recorded ===")
print("%-12s %8s | %s" % ("topic", "total", "recorded doc_type (top 5)"))
for t, c in grid.items():
    tot = sum(c.values())
    top = ", ".join("%s=%d" % (k, v) for k, v in c.most_common(5))
    print("%-12s %8d | %s" % (t, tot, top))

print("\n=== documents on-topic but typed 'pdf', so invisible to doc_type strata ===")
for t, n in lost.most_common():
    typed = grid[t].get(t, 0)
    print("  %-12s %6d PDFs on topic   vs %5d documents actually typed '%s'" % (t, n, typed, t))
