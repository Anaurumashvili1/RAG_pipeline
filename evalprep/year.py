import json, sys, re
sys.path.insert(0, "src")
from unitn_rag.text import resolve_effective_year

URLS = {
    "FAQ tirocinio 28.10.25.pdf (crawl says 2028)":
        "d65b4ee4-1976-4d37-9b4e-e419760d37e8",
    "FAQ tirocini inglese.pdf (your PDF)":
        "3baf6ad6-f5f3-495a-af6a-d6f5f3695a60",
}
FACTS = {
    "15 days rule": re.compile(r"15 (giorni|days)", re.I),
    "30 days rule": re.compile(r"30 (giorni|days)", re.I),
    "fifth day of internship": re.compile(r"quinto giorno|fifth day", re.I),
    "12 months post-graduate": re.compile(r"12 (mesi|months)|twelve months", re.I),
}

bad = 0
for l in open("dataset.ocr.jsonl", encoding="utf-8"):
    r = json.loads(l)
    u = r.get("url") or ""
    for label, frag in URLS.items():
        if frag in u:
            print("=== %s ===" % label)
            print("  crawl effective_year : %s" % r.get("effective_year"))
            print("  academic_year        : %s" % r.get("academic_year"))
            print("  resolved by loader   : %s" % resolve_effective_year(r, current_year=2026))
            t = r.get("text") or ""
            print("  contains:")
            for k, p in FACTS.items():
                print("     %-26s %s" % (k, bool(p.search(t))))
            print()
    y = r.get("effective_year")
    if isinstance(y, int) and y > 2027:
        bad += 1

print("raw crawl records with effective_year > 2027: %d" % bad)
