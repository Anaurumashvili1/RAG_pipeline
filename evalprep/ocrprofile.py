import json, collections, re, sys
sys.path.insert(0, "scripts")
from urllib.parse import urlsplit, unquote
from eval_candidates_v2 import _TOPIC_RE, topics_of, is_answerable, is_student_facing

rows = []
for l in open("dataset.ocr.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r.get("extractor") == "tesseract":
        rows.append(r)

print("OCR-recovered documents: %d" % len(rows))
print("  suspect (noise):        %d" % sum(1 for r in rows if r.get("ocr_suspect")))
print("  >3000 chars, clean:     %d" % sum(1 for r in rows if not r.get("ocr_suspect")
                                           and len(r.get("text") or "") > 3000))
print("  answerable:             %d" % sum(1 for r in rows if is_answerable(r)))
print("  student-facing host:    %d" % sum(1 for r in rows if is_student_facing(r)))

topical = collections.Counter()
none = 0
for r in rows:
    t = topics_of(r)
    if t:
        for k in t:
            topical[k] += 1
    else:
        none += 1
print("\ntopic (title + final path segment):")
for k, v in topical.most_common():
    print("  %-14s %d" % (k, v))
print("  %-14s %d" % "(no topic)", ) if False else print("  %-14s %d" % ("(none)", none))

# what are they actually about? crude subject buckets on the body
BUCKETS = {
 "staff/union":  r"ccnl|comparto|fondo ex art|contrattazione|personale tecnico|sindacal|trattamento accessorio",
 "governance":   r"consiglio di dipartimento|senato accademico|verbale|resoconto|commissione paritetica|cpds",
 "research pub": r"quaderni|working paper|dipartimento di politica sociale|issn",
 "travel/admin": r"missioni|rimborso spese|regolamento missioni|economato",
 "teaching":     r"programma del corso|syllabus|guida dello studente|piano di studi",
}
B = {k: re.compile(v, re.I) for k, v in BUCKETS.items()}
buck = collections.Counter()
for r in rows:
    blob = (r.get("title") or "") + " " + (r.get("text") or "")[:3000]
    hit = [k for k, p in B.items() if p.search(blob)]
    for k in hit:
        buck[k] += 1
    if not hit:
        buck["unclassified"] += 1
print("\ncontent buckets (body text):")
for k, v in buck.most_common():
    print("  %-14s %d" % (k, v))

print("\nhosts:")
for k, v in collections.Counter(urlsplit(r["url"]).netloc for r in rows).most_common(10):
    print("  %-32s %d" % (k, v))
