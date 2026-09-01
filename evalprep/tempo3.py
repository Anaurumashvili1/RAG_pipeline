import json, re

CORPUS = "dataset.v2.jsonl"
FRAGS = ["regolamento-didattico-lm-energy-engineering",
         "regolamento-didattico-lm-ingegneria-energetica"]

PROBES = {
    "1 CFU = 10 ore":     "ogni credito formativo corrisponde mediamente a 10 ore",
    "tesi in inglese":    "redatto in lingua inglese",
    "piano studi 2 anno": "iscrizione al 2",
    "lezioni Bolzano+TN": "sia presso la sede dell",
    "classe LM-30":       "LM-30",
    "commissione laurea": "La Commissione di laurea",
}

rows = []
wide = {k: 0 for k in PROBES}
for l in open(CORPUS, encoding="utf-8"):
    r = json.loads(l)
    t = r.get("text") or ""
    u = r.get("url") or ""
    if len(t) < 150 or r.get("duplicate_of") or r.get("low_content"):
        continue
    for k, p in PROBES.items():
        if p in t:
            wide[k] += 1
    if any(f in u for f in FRAGS):
        rows.append((r.get("effective_year"), u, t))

rows.sort(key=lambda x: -(x[0] or 0))
print("%-22s %s" % ("probe", "editions containing it"))
for k, p in PROBES.items():
    marks = []
    for y, u, t in rows:
        marks.append("Y" if p in t else ".")
    print("  %-20s %s      corpus-wide: %d docs" % (k, " ".join(marks), wide[k]))
print("\n  columns, newest first:")
for i, (y, u, t) in enumerate(rows):
    print("    %d) %-5s %s" % (i + 1, y, u.split("/")[-1]))

# exact wording of the two best candidates in the newest edition
newest = rows[0][2]
for label, needle in (("1 CFU = 10 ore", "ogni credito formativo corrisponde mediamente a 10 ore"),
                      ("tesi in inglese", "La prova finale per il conseguimento")):
    i = newest.find(needle)
    print("\n#### %s" % label)
    print("   %s" % (" ".join(newest[max(0, i - 120):i + 420].split()) if i >= 0 else "NOT FOUND"))
