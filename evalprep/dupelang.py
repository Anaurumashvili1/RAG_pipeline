"""Do the dangling duplicate_of drops leave a surviving sibling, and in which language?

If a dropped English page has a surviving Italian twin, a multilingual embedder
softens the loss. If nothing survives, the content is simply gone - and no model
can retrieve what is not indexed.
"""
import json, collections, re
from urllib.parse import urlsplit


def norm_group(url):
    p = urlsplit(url)
    host = p.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    path = re.sub(r"/(en|it)(/|$)", "/", p.path)
    path = re.sub(r"[._-](en|it)(\.\w+)$", r"\2", path)
    return host + path.rstrip("/")


kept, dropped, all_urls = [], [], set()
for l in open("dataset.ocr.jsonl", encoding="utf-8"):
    r = json.loads(l)
    u = (r.get("url") or "").strip()
    t = r.get("text") or ""
    if u:
        all_urls.add(u.rstrip("/"))
    row = (u, (r.get("lang") or "").lower(), len(t), r.get("duplicate_of"))
    if r.get("duplicate_of"):
        dropped.append(row)
    elif len(t) >= 150 and not r.get("low_content") and not r.get("boilerplate"):
        kept.append(row)

dangling = [d for d in dropped if (d[3] or "").rstrip("/") not in all_urls and d[2] >= 800]
kept_groups = collections.defaultdict(set)
for u, lg, n, _ in kept:
    kept_groups[norm_group(u)].add(lg)

print("corpus baseline language mix of KEPT documents:")
base = collections.Counter(k[1] or "unset" for k in kept)
tot = sum(base.values())
for k, v in base.most_common():
    print("  %-6s %6d  %4.1f%%" % (k, v, 100.0 * v / tot))

print("\nlanguage mix of the %d substantial dangling drops:" % len(dangling))
dl = collections.Counter(d[1] or "unset" for d in dangling)
for k, v in dl.most_common():
    print("  %-6s %6d  %4.1f%%" % (k, v, 100.0 * v / len(dangling)))

surv = collections.Counter()
for u, lg, n, _ in dangling:
    langs = kept_groups.get(norm_group(u), set())
    if not langs:
        surv["nothing survives"] += 1
    elif lg in langs:
        surv["same-language page survives"] += 1
    else:
        surv["only other-language sibling survives"] += 1

print("\nwhat survives for each dangling drop:")
for k, v in surv.most_common():
    print("  %-38s %5d  %4.1f%%" % (k, v, 100.0 * v / len(dangling)))
