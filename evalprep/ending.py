import json, re
from urllib.parse import urlsplit

SRC = re.compile(r"Certificate Part 1|Certificato Parte 1|attestato parte 1", re.I)
SOC = re.compile(r"sociolog", re.I)
NAMED = re.compile(r"Name_Surname|Nome_Cognome", re.I)

src, soc_alt = [], []
for l in open("dataset.ocr.jsonl", encoding="utf-8"):
    r = json.loads(l)
    t = r.get("text") or ""
    u = r.get("url") or ""
    if len(t) < 150 or r.get("duplicate_of") or r.get("low_content"):
        continue
    if SRC.search(t) or NAMED.search(t):
        src.append((r.get("lang"), len(t), bool(NAMED.search(t)), bool(SOC.search(t)),
                    (r.get("title") or "")[:50], u))
    # sociology's own internship-closure instructions
    if "sociologia" in urlsplit(u).netloc and re.search(r"tirocin|internship", u, re.I):
        soc_alt.append((r.get("lang"), len(t), (r.get("title") or "")[:50], u))

print("=== pages describing the closure procedure ===")
for lg, n, named, soc, ti, u in sorted(src, key=lambda x: x[5]):
    print("  %-3s %6dc name_rule=%-5s mentions_socio=%-5s %s" % (lg, n, named, soc, ti))
    print("      %s" % u)
print("\n=== sociology internship pages (the carve-out target) ===")
for lg, n, ti, u in sorted(soc_alt, key=lambda x: x[3])[:12]:
    print("  %-3s %6dc %-50s %s" % (lg, n, ti, u[:90]))
