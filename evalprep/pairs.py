import json, collections, re
from urllib.parse import urlsplit

def norm_group(url):
    p = urlsplit(url)
    host = p.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    path = re.sub(r"/(en|it)(/|$)", "/", p.path)
    path = re.sub(r"[._-](en|it)(\.\w+)$", r"\2", path)
    return host + path.rstrip("/")

by_host = collections.defaultdict(lambda: collections.defaultdict(set))
counts = collections.Counter()
for l in open("evalprep/slim_index.jsonl", encoding="utf-8"):
    r = json.loads(l)
    h = urlsplit(r["url"]).netloc.lower()
    lang = (r["lang"] or "").lower()
    counts[(h, lang)] += 1
    if lang in ("it", "en"):
        by_host[h][norm_group(r["url"])].add(lang)

print("%-28s %7s %7s %9s %9s" % ("host", "it", "en", "pairs", "pair%ofEN"))
for h in ("www.unitn.it", "corsi.unitn.it", "webapps.unitn.it", "borse.unitn.it",
          "phd.unitn.it", "www.cla.unitn.it", "orienta.unitn.it",
          "www.disi.unitn.it", "www.sociologia.unitn.it", "iecs.unitn.it"):
    it, en = counts[(h, "it")], counts[(h, "en")]
    pairs = sum(1 for v in by_host[h].values() if len(v) == 2)
    pct = (100.0 * pairs / en) if en else 0
    print("%-28s %7d %7d %9d %8.1f%%" % (h, it, en, pairs, pct))

tot_pairs = sum(1 for h in by_host for v in by_host[h].values() if len(v) == 2)
print("\ntotal URL-symmetric IT/EN pairs corpus-wide: %d" % tot_pairs)
print("corpus_stats() reports 'translated_pairs' as len(docs)-len(groups), which")
print("counts every group collision, not just real translations.")
