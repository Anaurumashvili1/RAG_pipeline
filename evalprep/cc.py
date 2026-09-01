import json, collections, re
from urllib.parse import urlsplit, unquote

CORPUS = "dataset.v2.jsonl"
HOST = "unitn.coursecatalogue.cineca.it"

urls = collections.Counter()
referrers = collections.Counter()
record = None
for l in open(CORPUS, encoding="utf-8"):
    r = json.loads(l)
    u = r.get("url") or ""
    if HOST in u:
        record = r
    for x in (r.get("out_links") or ()):
        if HOST in x:
            urls[x] += 1
            referrers[urlsplit(u).netloc.lower()] += 1

print("the single record the crawler has for %s:" % HOST)
if record:
    print("   url        : %s" % record.get("url"))
    print("   text       : %r" % (record.get("text") or "")[:200])
    print("   chars      : %d" % len(record.get("text") or ""))
    print("   status     : %s" % record.get("http_status"))
    print("   content_type: %s" % record.get("content_type"))
    print("   title      : %r" % record.get("title"))
    print("   low_content: %s  needs_ocr: %s" % (record.get("low_content"), record.get("needs_ocr")))

print("\ndistinct linked URLs on that host : %d" % len(urls))
print("total inbound link occurrences     : %d" % sum(urls.values()))

def shape(u):
    p = urlsplit(u)
    parts = [seg for seg in unquote(p.path).split("/") if seg]
    out = []
    for seg in parts:
        if re.fullmatch(r"\d+", seg):
            out.append("<id>")
        elif re.fullmatch(r"[0-9a-f-]{16,}", seg):
            out.append("<uuid>")
        else:
            out.append(seg)
    return "/" + "/".join(out[:4]) + ("?" + re.sub(r"=[^&]*", "=…", p.query) if p.query else "")

print("\nURL shapes:")
for k, v in collections.Counter(shape(u) for u in urls).most_common(20):
    print("   %-58s %d" % (k[:58], v))

print("\nwhich sites link to it:")
for k, v in referrers.most_common(12):
    print("   %-40s %d" % (k, v))

print("\nsample URLs:")
for u, n in urls.most_common(12):
    print("   %3dx %s" % (n, u[:112]))
