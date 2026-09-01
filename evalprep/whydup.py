"""Why is duplicate_of set on records whose canonical is absent?

Two candidate mechanisms:
  (a) observed duplicate - the crawler saw two records with identical content
      and kept one. Then the cleaning phase deleted the survivor, orphaning the
      pointer. If so, the dropped record's content_sha256 should match a KEPT
      record's hash.
  (b) declared canonical - the crawler read <link rel="canonical"> or applied a
      URL normalisation and recorded a URL it never actually fetched. If so, the
      hash matches nothing.
"""
import json, collections, re
from urllib.parse import urlsplit

kept_hash, all_urls, recs = collections.defaultdict(list), set(), []
for l in open("dataset.ocr.jsonl", encoding="utf-8"):
    r = json.loads(l)
    u = (r.get("url") or "").strip()
    if u:
        all_urls.add(u.rstrip("/"))
    h = r.get("content_sha256") or r.get("text_sha256")
    if not r.get("duplicate_of") and h:
        kept_hash[h].append(u)
    recs.append(r)

dangling = [r for r in recs
            if r.get("duplicate_of")
            and (r["duplicate_of"] or "").rstrip("/") not in all_urls
            and len(r.get("text") or "") >= 800]

match = collections.Counter()
patterns = collections.Counter()
for r in dangling:
    h = r.get("content_sha256") or r.get("text_sha256")
    match["hash matches a kept record" if h in kept_hash else "hash matches NOTHING kept"] += 1

    a, b = r["url"], r["duplicate_of"]
    pa, pb = urlsplit(a), urlsplit(b)
    if pa.scheme != pb.scheme and pa.netloc == pb.netloc and pa.path == pb.path:
        patterns["scheme only (http -> https)"] += 1
    elif pa.netloc != pb.netloc and pa.path == pb.path:
        patterns["host only (www / no-www)"] += 1
    elif b.startswith(a) or a.startswith(b):
        patterns["suffix appended (.1, index.html, /)"] += 1
    elif re.search(r"/node/\d+", pa.path) or re.search(r"/node/\d+", pb.path):
        patterns["CMS alias (node/N <-> pretty URL)"] += 1
    elif pa.netloc == pb.netloc:
        patterns["same host, different path"] += 1
    else:
        patterns["different host and path"] += 1

print("substantial dangling records: %d\n" % len(dangling))
print("does the dropped content exist anywhere that survived?")
for k, v in match.most_common():
    print("  %-32s %5d  %4.1f%%" % (k, v, 100.0 * v / len(dangling)))
print("\nhow the dropped URL differs from its declared canonical:")
for k, v in patterns.most_common():
    print("  %-38s %5d  %4.1f%%" % (k, v, 100.0 * v / len(dangling)))
