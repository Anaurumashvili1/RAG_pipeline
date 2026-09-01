"""duplicate_of pointing at a URL that is not itself in the corpus.

The loader drops any record with duplicate_of set, on the assumption that the
canonical version is present. Where the canonical URL was never kept, dropping
the duplicate deletes the content outright.

Example found by hand:
  https://www.biblioteca.unitn.it/node/345  (English BUD page, 1,366 chars)
    duplicate_of -> https://www.biblioteca.unitn.it/en/345/bud-university-digital-library
  ...which is absent from the corpus. Net effect: the English BUD page is gone.
"""
import json, collections
from urllib.parse import urlsplit

all_urls, dups = set(), []
for l in open("dataset.ocr.jsonl", encoding="utf-8"):
    r = json.loads(l)
    u = (r.get("url") or "").strip()
    if u:
        all_urls.add(u.rstrip("/"))
    d = r.get("duplicate_of")
    if d:
        dups.append((u, d.strip(), len(r.get("text") or ""), (r.get("title") or "")[:50]))

dangling = [d for d in dups if d[1].rstrip("/") not in all_urls]
print("records with duplicate_of set : %d" % len(dups))
print("of which the canonical URL is ABSENT from the corpus: %d (%.1f%%)"
      % (len(dangling), 100.0 * len(dangling) / max(1, len(dups))))

substantial = [d for d in dangling if d[2] >= 800]
print("  ...and the dropped record has >=800 chars of text: %d" % len(substantial))
print("  total characters deleted this way: %d" % sum(d[2] for d in dangling))

print("\nby host (dangling, >=800 chars):")
for k, v in collections.Counter(urlsplit(d[0]).netloc for d in substantial).most_common(12):
    print("  %-32s %d" % (k, v))

print("\nlargest 12 losses:")
for u, d, n, ti in sorted(substantial, key=lambda x: -x[2])[:12]:
    print("  %6dc %-50s" % (n, ti))
    print("        dropped : %s" % u[:100])
    print("        points to: %s" % d[:100])
