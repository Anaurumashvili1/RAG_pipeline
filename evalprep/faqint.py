import json, re
from urllib.parse import urlsplit, unquote

FAM = re.compile(r"FAQ.{0,6}(tirocin|internship)", re.I)
K15 = re.compile(r"15 days|15 giorni")
ABROAD = re.compile(r"without financial support|senza copertura|senza supporto economic", re.I)

fam, k15, ab = [], [], []
for l in open("dataset.ocr.jsonl", encoding="utf-8"):
    r = json.loads(l)
    t = r.get("text") or ""
    u = r.get("url") or ""
    ti = r.get("title") or ""
    if len(t) < 150 or r.get("duplicate_of") or r.get("low_content"):
        continue
    if FAM.search(unquote(u)) or FAM.search(ti):
        fam.append((r.get("lang"), r.get("effective_year"), len(t), ti[:58], u))
    if K15.search(t) and re.search(r"tirocin|internship", t, re.I) and re.search(r"30 (days|giorni)", t):
        k15.append((r.get("lang"), len(t), ti[:50], u))
    if ABROAD.search(t):
        ab.append((r.get("lang"), len(t), ti[:50], u))

print("=== FAQ tirocini / internship family ===")
for lg, y, n, ti, u in sorted(fam, key=lambda x: x[4]):
    print("  %-3s %-5s %6dc %s" % (lg, y, n, ti))
    print("      %s" % u[:120])
print("\n=== docs with the 15-day / 30-day post-graduate rule ===")
for lg, n, ti, u in k15:
    print("  %-3s %6dc %-50s %s" % (lg, n, ti, u[:100]))
print("\n=== docs with 'without financial support' ===")
for lg, n, ti, u in ab:
    print("  %-3s %6dc %-50s %s" % (lg, n, ti, u[:100]))
