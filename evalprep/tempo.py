"""Build a temporal test case: the same document across editions.

Two shapes, and only one of them is easy to grade.

  Type A - the fact is IDENTICAL in every edition. Any edition answers it
           correctly, so only the CITED URL reveals which one was used. You must
           grade the citation, not the answer.

  Type B - the fact CHANGED between editions. The ANSWER ITSELF reveals which
           edition was used, and a stale answer is a real-world error rather
           than a bookkeeping one. Much stronger, and far easier to grade.

This finds Type B candidates by diffing consecutive editions of one document
family and reporting numeric statements that differ.
"""
import json, re, difflib, collections

CORPUS = "dataset.v2.jsonl"
FAMILIES = {
    "energy engineering regolamento": [
        "regolamento-didattico-lm-energy-engineering",
        "regolamento-didattico-lm-ingegneria-energetica",
    ],
}

docs = collections.defaultdict(list)
for l in open(CORPUS, encoding="utf-8"):
    r = json.loads(l)
    u = r.get("url") or ""
    t = r.get("text") or ""
    if len(t) < 150 or r.get("duplicate_of") or r.get("low_content"):
        continue
    for fam, frags in FAMILIES.items():
        if any(f in u for f in frags):
            docs[fam].append((r.get("effective_year"), u, t))


def sentences(t):
    t = re.sub(r"---\s*PDF PAGE BREAK\s*---", " ", t)
    t = re.sub(r"\[page \d+\]", " ", t)
    parts = re.split(r"(?<=[.;:])\s+", t)
    out = []
    for p in parts:
        p = " ".join(p.split())
        if 40 <= len(p) <= 400:
            out.append(p)
    return out


NUM = re.compile(r"\b\d+([.,]\d+)?\b")

for fam, items in docs.items():
    items.sort(key=lambda x: -(x[0] or 0))
    print("=" * 78)
    print("%s - %d editions" % (fam, len(items)))
    for y, u, t in items:
        print("  %-5s %7dc  %s" % (y, len(t), u.split("/")[-1]))

    if len(items) < 2:
        continue
    (ynew, unew, tnew), (yold, uold, told) = items[0], items[1]
    snew, sold = sentences(tnew), sentences(told)
    setold = set(sold)
    print("\nnewest = %s   previous = %s" % (unew.split("/")[-1], uold.split("/")[-1]))

    # sentences present in the new edition and absent from the old, carrying numbers
    changed = [s for s in snew if s not in setold and NUM.search(s)]
    # pair each with its closest match in the old edition, to show what it replaced
    print("\n--- numeric statements that CHANGED between the two editions ---")
    shown = 0
    for s in changed:
        near = difflib.get_close_matches(s, sold, n=1, cutoff=0.75)
        if not near:
            continue
        a, b = near[0], s
        if NUM.findall(a) == NUM.findall(b):
            continue          # wording changed, numbers did not
        print("\n  OLD (%s): %s" % (yold, a[:260]))
        print("  NEW (%s): %s" % (ynew, b[:260]))
        shown += 1
        if shown >= 12:
            break
    if not shown:
        print("  none found - this family is Type A (facts stable across editions)")

    # Type A material: sentences identical in every edition
    common = set(sentences(items[0][2]))
    for _, _, t in items[1:]:
        common &= set(sentences(t))
    common = [s for s in common if NUM.search(s) and len(s) > 90]
    print("\n--- identical in ALL %d editions (Type A candidates) ---" % len(items))
    for s in sorted(common, key=len, reverse=True)[:6]:
        print("  * %s" % s[:250])
