import json, re
from urllib.parse import urlsplit

# the claim itself, not the wording: a 66-110 scale for the final exam
PAT = re.compile(r"66.{0,40}110|110.{0,40}66", re.I | re.S)
CUM = re.compile(r"cum laude|e lode", re.I)

hits = []
for l in open("dataset.ocr.jsonl", encoding="utf-8"):
    r = json.loads(l)
    t = r.get("text") or ""
    if len(t) < 150 or r.get("duplicate_of") or r.get("low_content"):
        continue
    m = PAT.search(t)
    if not m:
        continue
    ctx = " ".join(t[max(0, m.start() - 200):m.end() + 200].split())
    if not re.search(r"thesis|tesi|final exam|esame di laurea|prova finale|discussion|discussione|committee|commissione", ctx, re.I):
        continue
    hits.append((r.get("lang"), r.get("effective_year"), r.get("url"),
                 (r.get("title") or "")[:60], bool(CUM.search(ctx)), ctx[:200]))

print("documents stating a 66-110 final-exam scale: %d\n" % len(hits))
for lg, y, u, ti, cum, ctx in sorted(hits, key=lambda h: h[2]):
    print("%-3s %-5s cum_laude=%-5s %s" % (lg, y, cum, ti))
    print("    %s" % u)
print("\n--- sample context ---")
for h in hits[:3]:
    print("\n%s\n  %s" % (h[2], h[5]))
