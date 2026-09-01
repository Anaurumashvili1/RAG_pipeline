"""What the new year-resolution order actually changes, corpus-wide.

Old order (for comparison, reimplemented here so the diff is honest):
  academic_year START year -> crawl effective_year -> filename only when the
  crawler had fallen back to the current year -> regex fallback.
"""
import json, re, sys, collections
sys.path.insert(0, "src")
from unitn_rag import text as T

CY = 2026
MAXY = T.max_plausible_year(CY)


def old_resolve(raw):
    v = raw.get("academic_year")
    if v:
        m = re.match(r"\s*((?:19|20)\d{2})\s*/\s*((?:19|20)?\d{2})\s*$", str(v))
        if m:
            start, end = int(m.group(1)), int(m.group(2))
            if end < 100:
                end += (start // 100) * 100
            if end - start == 1 and 1990 <= start <= MAXY:
                return start
    c = raw.get("effective_year")
    if isinstance(c, str) and c.isdigit():
        c = int(c)
    if isinstance(c, int) and c == MAXY - 1:
        t = T.year_from_title(raw.get("title"), CY)   # end-year now, close enough
        if t and t != c:
            return t
    if isinstance(c, int) and 1990 <= c <= MAXY:
        return c
    return T.extract_effective_year(url=raw.get("url") or "", text=raw.get("text") or "",
                                    last_modified=raw.get("last_modified"), current_year=CY)


n = changed = 0
delta = collections.Counter()
reason = collections.Counter()
old_dist = collections.Counter()
new_dist = collections.Counter()
samples = collections.defaultdict(list)

for line in open("dataset.v2.jsonl", encoding="utf-8"):
    r = json.loads(line)
    n += 1
    o, w = old_resolve(r), T.resolve_effective_year(r, CY)
    old_dist[o] += 1
    new_dist[w] += 1
    if o == w:
        continue
    changed += 1
    delta[(w - o) if (o and w) else "none->year" if w else "year->none"] += 1
    ay = T.parse_academic_year(r.get("academic_year"), CY)
    doc = T.year_from_document(r.get("text"), CY)
    ttl = T.year_from_title(r.get("title"), CY)
    why = ("academic_year end-year" if ay == w and ay else
           "document states it" if doc == w and doc else
           "filename edition" if ttl == w and ttl else "other")
    reason[why] += 1
    if len(samples[why]) < 6:
        samples[why].append((o, w, (r.get("url") or "")[-88:]))

print("records %d   changed %d  (%.1f%%)" % (n, changed, 100 * changed / n))
print("\nby reason:")
for k, v in reason.most_common():
    print("  %-24s %6d" % (k, v))
print("\nby shift (new - old):")
for k, v in sorted(delta.items(), key=lambda x: -x[1])[:12]:
    print("  %-12s %6d" % (k, v))
print("\ndocuments dated 2026 or later:  old %d   new %d"
      % (sum(v for k, v in old_dist.items() if isinstance(k, int) and k >= 2026),
         sum(v for k, v in new_dist.items() if isinstance(k, int) and k >= 2026)))
print("documents with no year:         old %d   new %d" % (old_dist[None], new_dist[None]))
for why, rows in samples.items():
    print("\n--- %s ---" % why)
    for o, w, u in rows:
        print("   %s -> %-5s  %s" % (o, w, u))
