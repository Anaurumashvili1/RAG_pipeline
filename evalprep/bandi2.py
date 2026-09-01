import json, re, collections, difflib
from urllib.parse import unquote

rows = []
for l in open("dataset.v2.jsonl", encoding="utf-8"):
    r = json.loads(l)
    u = unquote(r.get("url") or "")
    t = r.get("text") or ""
    if len(t) < 400 or r.get("duplicate_of") or r.get("low_content"):
        continue
    if "borse.unitn.it" in u and re.search(r"bando[_ ]di[_ ]concorso", u, re.I):
        rows.append((r.get("effective_year"), r.get("url"), u, t))

rows.sort(key=lambda x: unquote(x[1]).lower())
print("bandi di concorso on borse.unitn.it: %d\n" % len(rows))
for y, raw, u, t in rows:
    print("  %-5s %7dc %s" % (y, len(t), u.split("/")[-1][:76]))

# group by series code, e.g. B_5_24 / B 5_26 -> series 5
def series(u):
    m = re.search(r"/[bB][_ ]?(\d+)[_ ](\d\d)[_ ]", u + " ")
    return m.group(1) if m else None

byser = collections.defaultdict(list)
for y, raw, u, t in rows:
    s = series(u)
    if s:
        byser[s].append((y, raw, u, t))

MONEY = re.compile(r"€\s?[\d.]+,\d{2}|\b\d{1,3}(?:\.\d{3})*,\d{2}\b")
print("\nseries with more than one year:")
for s, items in byser.items():
    if len(items) < 2:
        continue
    items.sort(key=lambda x: -(x[0] or 0))
    print("\n=== series B_%s : %d editions ===" % (s, len(items)))
    for y, raw, u, t in items:
        amounts = MONEY.findall(t)[:4]
        print("   %-5s %7dc %-56s amounts=%s" % (y, len(t), u.split("/")[-1][:56], amounts))
    a, b = items[0], items[1]
    for lab, (y, raw, u, t) in (("NEW", a), ("OLD", b)):
        i = t.find("compenso lordo")
        j = t.lower().find("termine di scadenza")
        print("\n   %s (%s) %s" % (lab, y, raw[:100]))
        if i >= 0:
            print("      compenso : %s" % " ".join(t[max(0, i - 90):i + 130].split()))
        if j >= 0:
            print("      scadenza : %s" % " ".join(t[max(0, j - 190):j + 60].split()))
