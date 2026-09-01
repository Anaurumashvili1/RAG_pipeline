"""Give every eval target the full set of URLs that count as a correct retrieval.

`hit@k` compares a retrieved URL against the gold one. `select_pages()` collapses
an IT/EN pair into a single slot and reports whichever sibling ranked higher, so
a set naming only one of the two records a correct retrieval as a miss - a false
negative that lands on exactly the bilingual items this corpus is full of.

The pairing rule is the reciprocal-link one from eval_candidates_v2: A and B are
siblings if A links to B, B links back to A, same host, different language. The
language switcher is an ordinary outbound link and `out_links` was already
captured, so no re-crawl is needed. Reciprocity is what makes it safe - a
one-way link is a mention, a two-way link is the same page in the other
language.

Writes evalprep/siblings.json  (target_url -> [acceptable, ...]).
"""
import json
from urllib.parse import urlsplit

CORPUS = "dataset.v2.jsonl"
EVAL = "evalprep/evaluation_set.fixed.json"
OUT = "evalprep/siblings.json"


def key(u):
    """Scheme-insensitive, trailing-slash-insensitive comparison key."""
    u = (u or "").strip().rstrip("/")
    for p in ("https://", "http://"):
        if u.startswith(p):
            return u[len(p):]
    return u


def host_of(u):
    u = (u or "").strip()
    return urlsplit(u if "//" in u else "//" + u).netloc.lower()


def eligible(r):
    t = r.get("text") or ""
    return (len(t) >= 150 and not r.get("duplicate_of") and not r.get("low_content")
            and not r.get("boilerplate")
            and (r.get("lang") or "").lower().split("-")[0] in ("it", "en", ""))


items = json.load(open(EVAL, encoding="utf-8"))
hosts = {host_of(i["target_url"]) for i in items}

# One pass. Keeping only rows on the eval targets' hosts holds memory down; the
# language switcher never points off-host, so no possible sibling is lost.
lang, links, elig, title, display = {}, {}, {}, {}, {}
seen = 0
for line in open(CORPUS, encoding="utf-8"):
    r = json.loads(line)
    url = r.get("url") or ""
    if host_of(url) not in hosts:
        continue
    k = key(url)
    if not k:
        continue
    seen += 1
    lang[k] = (r.get("lang") or "").lower().split("-")[0]
    links[k] = {key(x) for x in (r.get("out_links") or [])}
    elig[k] = eligible(r)
    title[k] = (r.get("title") or "")[:90]
    display[k] = url

print("rows on eval hosts: %d   (%d hosts)" % (seen, len(hosts)))

result, report = {}, []
for i in items:
    t = key(i["target_url"])
    host, lg = host_of(i["target_url"]), lang.get(t, "?")
    cands = []
    for other in links.get(t, ()):
        if other == t or other not in lang:
            continue
        if host_of(other) != host:
            continue
        if lang[other] not in ("it", "en") or lang[other] == lg:
            continue
        if t in links.get(other, ()):                 # reciprocal
            cands.append(other)
    if cands:
        result[i["target_url"]] = [display[c] for c in cands]
    report.append((i["id"], lg, cands))

json.dump(result, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

paired = [r for r in report if r[2]]
ambiguous = [r for r in report if len(r[2]) > 1]
missing = [r for r in report if r[1] == "?"]
print("\n%d of %d targets have a reciprocal sibling; %d ambiguous (>1 candidate)"
      % (len(paired), len(items), len(ambiguous)))
if missing:
    print("targets not found on their own host (check): ids %s" % [r[0] for r in missing])

print("\n--- per item ---")
for _id, lg, cands in paired:
    print("  id %-3s target lang=%s" % (_id, lg))
    for c in cands:
        print("      %-3s eligible=%-5s %s" % (lang[c], elig[c], display[c][:96]))
        print("          %s" % title[c])

bad = sorted({i for i, _, cs in report for c in cs if not elig[c]})
if bad:
    print("\nsiblings present but NOT index-eligible (harmless to list, they can "
          "never be retrieved): ids %s" % bad)
print("\nwrote %s" % OUT)
