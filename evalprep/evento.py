import json, collections, re
from urllib.parse import urlsplit

EV = re.compile(r"webmagazine\.unitn\.it/.*?/evento/")
records, linked = set(), set()
cal_pages, cal_ev_links = 0, 0
wm_records = 0

for l in open("dataset.ocr.jsonl", encoding="utf-8"):
    r = json.loads(l)
    u = (r.get("url") or "")
    if "webmagazine.unitn.it" in u:
        wm_records += 1
    if EV.search(u):
        records.add(u.rstrip("/"))
    links = r.get("out_links") or ()
    is_cal = "webmagazine.unitn.it" in u and "/calendario/" in u
    if is_cal:
        cal_pages += 1
    for x in links:
        if EV.search(x):
            linked.add(x.rstrip("/"))
            if is_cal:
                cal_ev_links += 1

print("webmagazine.unitn.it records in corpus      : %d" % wm_records)
print("  of which /evento/ pages                   : %d" % len(records))
print("distinct /evento/ URLs seen in out_links     : %d" % len(linked))
print("  linked but NEVER fetched                   : %d" % len(linked - records))
print("  fetched                                    : %d" % len(linked & records))
print("\ncalendario index pages in corpus            : %d" % cal_pages)
print("  outgoing /evento/ links from those pages   : %d" % cal_ev_links)

print("\nsample of linked-but-never-fetched:")
for x in sorted(linked - records)[:8]:
    print("   %s" % x[:118])
