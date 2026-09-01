"""Rewrite evaluation_set.json against dataset.v2.jsonl.

  - drop the items marked for deletion
  - replace /en/N/slug targets with the node/N URL actually present in the corpus
  - verify every remaining target is present AND index-eligible
  - renumber ids contiguously, keeping the original as `old_id`
"""
import json, re
from urllib.parse import urlsplit, unquote

SRC = "evalprep/evaluation_set.json"
OUT = "evalprep/evaluation_set.fixed.json"
CORPUS = "dataset.v2.jsonl"

DELETE = {23, 29, 37, 38, 39, 40, 42, 45}
REMAP = {
     6: "https://www.dicam.unitn.it/node/1829",
    16: "https://www.giurisprudenza.unitn.it/node/1025",
    20: "https://www.dicam.unitn.it/node/1721",
    22: "https://www.giurisprudenza.unitn.it/node/3453",
    24: "https://www.giurisprudenza.unitn.it/node/997",
    26: "https://www.giurisprudenza.unitn.it/node/683",
    27: "https://www.giurisprudenza.unitn.it/node/683",
    32: "https://www.centro3a.unitn.it/node/340",
    34: "https://www.cogsci.unitn.it/node/339",
    35: "https://www.economia.unitn.it/node/2457",
    36: "https://www.cogsci.unitn.it/node/1402",
    41: "https://www.sociologia.unitn.it/node/2302",
    43: "https://www.sociologia.unitn.it/node/264",
    44: "https://www.cogsci.unitn.it/node/1401",
    46: "https://www.physics.unitn.it/node/433",
    47: "https://www.cimec.unitn.it/node/202",
}

items = [i for i in json.load(open(SRC, encoding="utf-8")) if i["id"] not in DELETE]
for i in items:
    if i["id"] in REMAP:
        i["source_url"] = i["target_url"]          # the readable URL, for citation checks
        i["target_url"] = REMAP[i["id"]]

wanted = {i["target_url"].rstrip("/") for i in items}
info = {}
for l in open(CORPUS, encoding="utf-8"):
    r = json.loads(l)
    u = (r.get("url") or "").strip().rstrip("/")
    if u in wanted:
        t = r.get("text") or ""
        eligible = (len(t) >= 150 and not r.get("duplicate_of")
                    and not r.get("low_content") and not r.get("boilerplate")
                    and (r.get("lang") or "").lower().split("-")[0] in ("it", "en", ""))
        info[u] = (eligible, len(t), r.get("title") or "", r.get("lang"),
                   r.get("effective_year"))

out = []
problems = []
for n, i in enumerate(items):
    u = i["target_url"].rstrip("/")
    if u not in info:
        problems.append((i["id"], "ABSENT", i["target_url"]))
        continue
    ok, chars, title, lang, year = info[u]
    if not ok:
        problems.append((i["id"], "present but DROPPED at load", i["target_url"]))
    rec = {
        "id": n,
        "old_id": i["id"],
        "question": i["question"],
        "gold_answer": i["gold_answer"],
        "target_url": i["target_url"],
        "title": i.get("title", ""),
        "corpus_chars": chars,
        "corpus_lang": lang,
        "corpus_year": year,
    }
    if "source_url" in i:
        rec["source_url"] = i["source_url"]
    out.append(rec)

json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("in  : %d items" % (len(items) + len(DELETE)))
print("deleted: %d" % len(DELETE))
print("remapped to node/N: %d" % len(REMAP))
print("out : %d items -> %s" % (len(out), OUT))
if problems:
    print("\nSTILL PROBLEMATIC:")
    for pid, why, u in problems:
        print("  id %-3s %-28s %s" % (pid, why, u[:90]))
else:
    print("\nevery target is present and index-eligible")

dupes = {}
for r in out:
    dupes.setdefault(r["target_url"], []).append(r["id"])
shared = {k: v for k, v in dupes.items() if len(v) > 1}
if shared:
    print("\ntargets shared by more than one question (correlated failures):")
    for k, v in shared.items():
        print("  ids %-14s %s" % (",".join(map(str, v)), k[:88]))
