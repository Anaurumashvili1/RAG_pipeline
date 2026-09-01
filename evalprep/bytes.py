import json, re, unicodedata

U = "https://corsi.unitn.it/sites/cds/files/2026-07/regolamento-didattico-lm-energy-engineering-2026.pdf"
for l in open("dataset.v2.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r.get("url") != U:
        continue
    t = r.get("text") or ""
    i = t.find("La prova finale per il conseguimento")
    seg = t[i:i + 200]
    print("raw segment:")
    print(repr(seg))
    print("\ncodepoints that are not plain ASCII:")
    seen = {}
    for ch in seg:
        if ord(ch) > 127 or ch in "\t\n\r":
            seen.setdefault(ch, unicodedata.name(ch, "?"))
    for ch, name in seen.items():
        print("   U+%04X  %-40s %r" % (ord(ch), name, ch))
    norm = re.sub(r"\s+", " ", seg.replace("­", "").replace("​", ""))
    print("\nnormalised: %r" % norm[:180])
    print("\nliteral 'redatto in lingua inglese' in raw : %s" % ("redatto in lingua inglese" in seg))
    print("literal 'redatto in lingua inglese' in norm: %s" % ("redatto in lingua inglese" in norm))
    break
