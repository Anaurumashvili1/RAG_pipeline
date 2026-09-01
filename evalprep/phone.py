import json, re, collections
from urllib.parse import urlsplit

NUM = "282587"
pages, contacts = [], 0
for l in open("dataset.ocr.jsonl", encoding="utf-8"):
    r = json.loads(l)
    t = r.get("text") or ""
    if len(t) < 150 or r.get("duplicate_of") or r.get("low_content") or r.get("boilerplate"):
        continue
    if NUM in t.replace(" ", ""):
        u = r.get("url") or ""
        pages.append((r.get("lang"), r.get("title") or "", u))
        if "/contacts" in u or "contatti" in u:
            contacts += 1

print("index-eligible documents containing %s: %d  (of which contact pages: %d)"
      % (NUM, len(pages), contacts))
print("\nby host:")
for k, v in collections.Counter(urlsplit(p[2]).netloc for p in pages).most_common():
    print("  %-22s %d" % (k, v))
print("\nfirst 20:")
for lg, ti, u in sorted(pages, key=lambda p: p[2])[:20]:
    print("  %-3s %-42s %s" % (lg, ti[:42], u[:95]))
