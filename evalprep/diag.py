"""Why is each eval target_url missing from the corpus?

For every target we ask, in order:
  1. exact URL present?
  2. present and index-eligible (survives load_documents)?
  3. a URL VARIANT present (trailing slash / percent-encoding / http-https / www)?
  4. the same page under a different alias (node/N, /en/ prefix stripped)?
  5. named as some record's duplicate_of (so the crawler knew of it)?
  6. present in any record's out_links (so it was discoverable)?
"""
import json, re, collections
from urllib.parse import urlsplit, unquote, quote

EVAL = "evalprep/evaluation_set.json"
CORPUS = "dataset.v2.jsonl"


def variants(u):
    u = u.strip()
    out = {u, u.rstrip("/"), u + "/"}
    for base in list(out):
        out.add(unquote(base))
        out.add(quote(unquote(base), safe=":/?&=#%"))
        p = urlsplit(base)
        other = "www." + p.netloc if not p.netloc.startswith("www.") else p.netloc[4:]
        for scheme in ("http", "https"):
            for host in (p.netloc, other):
                out.add(f"{scheme}://{host}{p.path}" + (("?" + p.query) if p.query else ""))
    return {x.rstrip("/") for x in out if x}


def alias_keys(u):
    """Keys that identify the same CMS page: host + numeric id, and lang-stripped path."""
    p = urlsplit(u)
    host = p.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    keys = set()
    path = unquote(p.path)
    for m in re.finditer(r"/(\d{2,7})(?:/|$)", path):
        keys.add(f"{host}#id{m.group(1)}")
    lang = re.sub(r"^/(en|it)(/|$)", "/", path)
    keys.add(f"{host}{lang.rstrip('/')}")
    return keys


items = json.load(open(EVAL, encoding="utf-8"))
targets = {i["id"]: i["target_url"] for i in items if i.get("target_url")}
allvars = {}
for k, u in targets.items():
    allvars[k] = variants(u)
want_flat = {v: k for k, vs in allvars.items() for v in vs}
want_alias = collections.defaultdict(set)
for k, u in targets.items():
    for a in alias_keys(u):
        want_alias[a].add(k)

exact, eligible, alias_hit = {}, {}, collections.defaultdict(list)
dup_named, linked = collections.defaultdict(list), collections.defaultdict(int)

for l in open(CORPUS, encoding="utf-8"):
    r = json.loads(l)
    u = (r.get("url") or "").strip()
    t = r.get("text") or ""
    ok = (len(t) >= 150 and not r.get("duplicate_of") and not r.get("low_content")
          and not r.get("boilerplate")
          and (r.get("lang") or "").lower().split("-")[0] in ("it", "en", ""))
    key = want_flat.get(u.rstrip("/"))
    if key is not None:
        exact[key] = u
        eligible[key] = (ok, len(t), r.get("duplicate_of"), r.get("low_content"),
                         r.get("lang"), r.get("http_status"))
    for a in alias_keys(u):
        for k in want_alias.get(a, ()):
            if k not in exact:
                alias_hit[k].append((u, len(t), ok))
    d = (r.get("duplicate_of") or "").strip().rstrip("/")
    if d in want_flat:
        dup_named[want_flat[d]].append(u)
    for x in (r.get("out_links") or ()):
        k = want_flat.get(x.strip().rstrip("/"))
        if k is not None:
            linked[k] += 1

DELETE = {23, 29, 37, 38, 39, 40, 42, 45}
rows = []
for k in sorted(targets):
    if k in exact:
        ok, n, dup, low, lang, st = eligible[k]
        verdict = "PRESENT + eligible" if ok else "PRESENT but DROPPED"
        detail = "" if ok else f"dup={bool(dup)} low={low} chars={n} lang={lang}"
    elif alias_hit.get(k):
        best = sorted(alias_hit[k], key=lambda x: -x[1])[0]
        verdict = "ALIAS present"
        detail = f"{best[1]}c eligible={best[2]} -> {best[0][:78]}"
    elif dup_named.get(k):
        verdict = "named as canonical of a dropped record"
        detail = dup_named[k][0][:78]
    elif linked.get(k):
        verdict = "NEVER FETCHED (but linked)"
        detail = f"{linked[k]} inbound links"
    else:
        verdict = "NEVER FETCHED, never linked"
        detail = ""
    rows.append((k, verdict, detail, targets[k]))

print("%-4s %-38s %s" % ("id", "verdict", "detail"))
for k, v, d, u in rows:
    mark = " (user: delete)" if k in DELETE else ""
    print("%-4s %-38s %s%s" % (k, v, d, mark))
    if not v.startswith("PRESENT + eligible"):
        print("      %s" % u[:118])

print("\n=== summary (excluding the ones you marked delete) ===")
c = collections.Counter(v for k, v, d, u in rows if k not in DELETE)
for k, v in c.most_common():
    print("  %-40s %d" % (k, v))
