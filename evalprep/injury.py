import json, re
from urllib.parse import urlsplit

TYPO = "assicuazioni"
CORRECT = "assicurazioni@unitn.it"
INJ = re.compile(r"infortunio|injury", re.I)

typo, correct_only, both = [], [], []
for l in open("dataset.ocr.jsonl", encoding="utf-8"):
    r = json.loads(l)
    t = r.get("text") or ""
    if len(t) < 150 or r.get("duplicate_of") or r.get("low_content"):
        continue
    has_typo = TYPO in t
    has_ok = CORRECT in t
    if not (has_typo or has_ok) or not INJ.search(t):
        continue
    row = (r.get("lang"), r.get("effective_year"), len(t), (r.get("title") or "")[:55], r.get("url"))
    (both if (has_typo and has_ok) else typo if has_typo else correct_only).append(row)

for label, rows in (("TYPO 'gestione.assicuazioni'", typo),
                    ("correct 'gestione.assicurazioni'", correct_only),
                    ("both spellings", both)):
    print("=== %s: %d ===" % (label, len(rows)))
    for lg, y, n, ti, u in sorted(rows, key=lambda x: x[4] or ""):
        print("  %-3s %-5s %6dc %-55s" % (lg, y, n, ti))
        print("      %s" % u)
    print()
