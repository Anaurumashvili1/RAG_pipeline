import json, re, difflib, collections

CORPUS = "dataset.v2.jsonl"

def norm(t):
    t = re.sub(r"---\s*PDF PAGE BREAK\s*---", " ", t)
    t = re.sub(r"\[page \d+\]", " ", t)
    t = re.sub(r"INDICE \d+ UNIVERSIT[AÀ] DEGLI STUDI DI TRENTO GUIDA DELLA FACOLT[AÀ] DI GIURISPRUDENZA [\d/\-]+", " ", t)
    return " ".join(t.split())

def sentences(t):
    out = []
    for p in re.split(r"(?<=[.;:])\s+|•", norm(t)):
        p = " ".join(p.split())
        if 40 <= len(p) <= 240:
            out.append(p)
    return out

guides = []
for l in open(CORPUS, encoding="utf-8"):
    r = json.loads(l)
    u = r.get("url") or ""
    t = r.get("text") or ""
    ti = (r.get("title") or "")
    if len(t) < 40000 or r.get("duplicate_of") or r.get("low_content"):
        continue
    if "giurisprudenza" in u.lower() and ("guida" in u.lower() or "GUIDA" in ti):
        guides.append((r.get("effective_year"), u, ti, t))

guides.sort(key=lambda x: -(x[0] or 0))
print("Giurisprudenza guide editions >=40k chars: %d" % len(guides))
for y, u, ti, t in guides[:14]:
    print("   %-5s %7dc %s" % (y, len(t), (ti or u.split('/')[-1])[:70]))

NUM = re.compile(r"\b\d{1,4}\b")
if len(guides) >= 2:
    new = guides[0]
    for old in guides[1:5]:
        sn, so = sentences(new[3]), sentences(old[3])
        print("\n" + "=" * 74)
        print("NEW %s (%s)  vs  OLD %s (%s)"
              % (new[2][:36] or new[1].split('/')[-1][:36], new[0],
                 old[2][:36] or old[1].split('/')[-1][:36], old[0]))
        shown = 0
        for s in sn:
            if not NUM.search(s):
                continue
            near = difflib.get_close_matches(s, so, n=1, cutoff=0.85)
            if not near or near[0] == s:
                continue
            a = near[0]
            if NUM.findall(a) == NUM.findall(s):
                continue
            print("\n   OLD: %s" % a[:210])
            print("   NEW: %s" % s[:210])
            shown += 1
            if shown >= 5:
                break
        if not shown:
            print("   no numeric differences found")
