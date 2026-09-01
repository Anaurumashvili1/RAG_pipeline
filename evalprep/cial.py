import json, re

CORPUS = "dataset.v2.jsonl"
PICK = {
    "2025-26 (current)": "GUIDA%20GIURISPRUDENZA%202025-26_PORTALE.pdf",
    "2007-08 (stamped 2026)": "02_Guida%20Magistrale%202007-08.pdf",
    "2009-10 (stamped 2026)": "05_Guida%20Magistrale%202009-10.pdf",
}
docs = {}
years = {}
for l in open(CORPUS, encoding="utf-8"):
    r = json.loads(l)
    u = r.get("url") or ""
    for label, frag in PICK.items():
        if frag in u:
            docs[label] = " ".join((r.get("text") or "").split())
            years[label] = (r.get("effective_year"), r.get("academic_year"), len(r.get("text") or ""))

for label in PICK:
    y = years.get(label)
    print("%-24s effective_year=%s academic_year=%s chars=%s" % (label, y[0], y[1], y[2]) if y else "%s MISSING" % label)

print("\n--- CIAL language credits ---")
for label, t in docs.items():
    i = t.find("CIAL sono attivati corsi di lingua")
    print("\n%s:" % label)
    print("   %s" % (t[max(0, i - 60):i + 420] if i >= 0 else "not found"))

print("\n--- 'appelli' in the exam session ---")
for label, t in docs.items():
    hits = re.findall(r"[^.]{0,90}\(\s*\d+\s*appell[oi]\s*\)[^.]{0,60}", t)
    print("\n%s: %s" % (label, hits[:2] if hits else "not found"))

print("\n--- how many guides are stamped effective_year 2026 ---")
n = 0
old = []
for l in open(CORPUS, encoding="utf-8"):
    r = json.loads(l)
    u = r.get("url") or ""
    if "giurisprudenza" in u.lower() and "guida" in u.lower() and r.get("effective_year") == 2026:
        n += 1
        old.append((len(r.get("text") or ""), u.split("/")[-1][:70]))
print("   %d" % n)
for c, name in sorted(old, reverse=True):
    print("   %7dc %s" % (c, name))
