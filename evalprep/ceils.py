import json, re
from urllib.parse import urlsplit, unquote

want = re.compile(r"guidelines.{0,40}tra[sn]?sfer|linee.?guida.{0,30}trasferiment", re.I)
ceils = re.compile(r"\bCEILS\b")
hits, fam = [], []
for l in open("dataset.ocr.jsonl", encoding="utf-8"):
    r = json.loads(l)
    u = r.get("url") or ""
    t = r.get("text") or ""
    ti = r.get("title") or ""
    if want.search(unquote(u)) or want.search(ti):
        fam.append((r.get("lang"), r.get("effective_year"), len(t),
                    bool(r.get("duplicate_of")), ti[:70], u))
    if ceils.search(t) and re.search(r"60 ECTS|30 ECTS", t):
        hits.append((r.get("lang"), len(t), ti[:70], u))

print("=== 'guidelines transfer' family ===")
for h in sorted(fam, key=lambda x: x[5]):
    print("  %-3s %-5s %6dc dup=%-5s %s" % (h[0], h[1], h[2], h[3], h[4]))
    print("      %s" % h[5])
print("\n=== docs mentioning CEILS with a 30/60 ECTS figure ===")
for h in hits:
    print("  %-3s %6dc %-60s" % (h[0], h[1], h[2]))
    print("      %s" % h[3])
