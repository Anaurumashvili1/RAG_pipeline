"""Look for a Type B temporal case: same recurring document, value changes per year.

Candidates: annual scholarship calls (bandi di concorso) on borse.unitn.it, and
the administrative determinazioni families.
"""
import json, re, collections, difflib

CORPUS = "dataset.v2.jsonl"
FAMS = {
    "bandi di concorso (borse)": lambda u, t: "borse.unitn.it" in u and re.search(r"bando[_ ]di[_ ]concorso", u, re.I),
    "determinazioni DPI": lambda u, t: "Determinazioni_DPI" in u,
    "determinazioni DCRE": lambda u, t: "Determinazioni_DCRE" in u,
}

def norm(t):
    t = re.sub(r"---\s*PDF PAGE BREAK\s*---", " ", t)
    t = re.sub(r"\[page \d+\]", " ", t)
    return t

def sentences(t):
    out = []
    for p in re.split(r"(?<=[.;:])\s+|\n", norm(t)):
        p = " ".join(p.split())
        if 45 <= len(p) <= 260:
            out.append(p)
    return out

MONEY = re.compile(r"\b\d{1,3}(?:[.\s]\d{3})*(?:,\d{2})?\s*(?:€|euro|EUR)\b|€\s*\d[\d.,]*", re.I)
DATE = re.compile(r"\b\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+20\d{2}\b|\b\d{1,2}/\d{1,2}/20\d{2}\b", re.I)

groups = collections.defaultdict(list)
for l in open(CORPUS, encoding="utf-8"):
    r = json.loads(l)
    u = r.get("url") or ""
    t = r.get("text") or ""
    if len(t) < 400 or r.get("duplicate_of") or r.get("low_content"):
        continue
    for name, test in FAMS.items():
        if test(u, t):
            groups[name].append((r.get("effective_year"), u, t))

for name, items in groups.items():
    items.sort(key=lambda x: -(x[0] or 0))
    print("=" * 78)
    print("%s : %d documents" % (name, len(items)))
    for y, u, t in items[:8]:
        print("   %-5s %7dc %s" % (y, len(t), u.split("/")[-1][:70]))
    if len(items) < 2:
        continue
    (yn, un, tn), (yo, uo, to) = items[0], items[1]
    sn, so = sentences(tn), sentences(to)
    print("\n   newest: %s\n   prev  : %s" % (un.split("/")[-1], uo.split("/")[-1]))
    shown = 0
    for s in sn:
        if not (MONEY.search(s) or DATE.search(s)):
            continue
        near = difflib.get_close_matches(s, so, n=1, cutoff=0.80)
        if not near:
            continue
        a = near[0]
        if a == s:
            continue
        va = MONEY.findall(a) + DATE.findall(a)
        vb = MONEY.findall(s) + DATE.findall(s)
        if va == vb:
            continue
        print("\n   OLD (%s): %s" % (yo, a[:230]))
        print("   NEW (%s): %s" % (yn, s[:230]))
        shown += 1
        if shown >= 6:
            break
    if not shown:
        print("   no value changes detected")
    print()
