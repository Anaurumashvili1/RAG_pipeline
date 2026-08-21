"""Can the IT/EN sibling be recovered from out_links instead of hreflang?

URL-symmetry pairing works on 2% of www.unitn.it because unitn.it translates the
path segments as well as the /it//en/ prefix. But most pages carry a language
switcher, which is an ordinary outbound link. If A links to B and B links back
to A, and they are the same host in opposite languages, that is almost certainly
a translation pair.
"""
import json, collections
from urllib.parse import urlsplit

HOSTS = {"www.unitn.it", "corsi.unitn.it", "borse.unitn.it", "phd.unitn.it",
         "www.cla.unitn.it", "www.sociologia.unitn.it", "iecs.unitn.it"}

lang, links = {}, {}
eligible = set()
for l in open("dataset.jsonl", encoding="utf-8"):
    r = json.loads(l)
    u = (r.get("url") or "").strip()
    if not u or urlsplit(u).netloc.lower() not in HOSTS:
        continue
    if (len(r.get("text") or "") < 150 or r.get("duplicate_of")
            or r.get("low_content") or r.get("boilerplate")):
        continue
    lg = (r.get("lang") or "").lower().split("-")[0]
    if lg not in ("it", "en"):
        continue
    eligible.add(u)
    lang[u] = lg
    links[u] = {x.rstrip("/") for x in (r.get("out_links") or [])}

norm = {u.rstrip("/"): u for u in eligible}

pairs, ambiguous, none = {}, 0, 0
for u, lg in lang.items():
    host = urlsplit(u).netloc.lower()
    cands = []
    for tgt in links.get(u, ()):
        real = norm.get(tgt)
        if not real or real == u:
            continue
        if urlsplit(real).netloc.lower() != host or lang.get(real) == lg:
            continue
        # reciprocal: the sibling links back
        if u.rstrip("/") in links.get(real, ()):
            cands.append(real)
    if len(cands) == 1:
        pairs[u] = cands[0]
    elif cands:
        ambiguous += 1
    else:
        none += 1

per_host = collections.Counter(urlsplit(u).netloc.lower() for u in pairs)
tot = collections.Counter(urlsplit(u).netloc.lower() for u in lang)
print("%-26s %8s %10s %8s" % ("host", "docs", "paired", "cover"))
for h in sorted(HOSTS):
    print("%-26s %8d %10d %7.1f%%" % (h, tot[h], per_host[h],
                                      100.0 * per_host[h] / tot[h] if tot[h] else 0))
print("\nunique reciprocal pairs: %d  |  ambiguous: %d  |  no sibling: %d"
      % (len(pairs) // 2, ambiguous, none))
print("\nexamples:")
for i, (a, b) in enumerate(pairs.items()):
    if lang[a] == "it" and i < 400:
        print("  IT %s\n  EN %s\n" % (a, b))
    if i > 40:
        break
