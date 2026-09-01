"""Find document families where the SAME question has a DIFFERENT answer per edition.

That is the temporal case worth testing. If every edition gives the same answer,
only the cited URL reveals which was used and you must grade citations. If the
value moves year to year, the ANSWER ITSELF exposes a stale retrieval - much
easier to grade, and a stale answer is a real error rather than a bookkeeping one.

Groups documents by a version-stripped title, keeps families spanning >=2
effective years, and reports sentences that are near-identical across editions
except for a date or amount.
"""
import json, re, collections, difflib
from urllib.parse import urlsplit, unquote

CORPUS = "dataset.v2.jsonl"

VERSION_NOISE = re.compile(
    r"(20\d{2}[-_/.]?\d{0,4})|(\bv\d+\b)|(\bed\b)|(\brev\b)|(\d{1,2}[._-]\d{1,2}[._-]\d{2,4})",
    re.IGNORECASE)


def title_key(title, url):
    t = unquote(title or "")
    t = re.sub(r"\.(pdf|docx?|xlsx?)$", "", t, flags=re.IGNORECASE)
    t = VERSION_NOISE.sub(" ", t)
    t = re.sub(r"[^0-9a-zà-ÿ]+", " ", t.lower()).strip()
    return re.sub(r"\s+", " ", t)


def sentences(t):
    t = re.sub(r"---\s*PDF PAGE BREAK\s*---", " ", t)
    t = re.sub(r"\[page \d+\]", " ", t)
    # drop running headers: "DR n. 637 del 13/07/2026 Pagina 3 di 18"
    t = re.sub(r"Emanato con DR n\.?\s*\d+\s*del\s*\d{2}/\d{2}/\d{4}", " ", t)
    t = re.sub(r"Pagina\s*\d+\s*di\s*\d+", " ", t)
    out = []
    for p in re.split(r"(?<=[.;:])\s+|\n", t):
        p = " ".join(p.split())
        if 45 <= len(p) <= 300:
            out.append(p)
    return out


VALUE = re.compile(r"\b\d{1,2}\s+(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre|January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}\b"
                   r"|\b\d{1,2}/\d{1,2}/20\d{2}\b|\b\d[\d.,]*\s*(€|euro|EUR)\b|\b\d{1,3}(?:[.,]\d{2})?\s*%", re.IGNORECASE)

fam = collections.defaultdict(list)
for l in open(CORPUS, encoding="utf-8"):
    r = json.loads(l)
    t = r.get("text") or ""
    u = r.get("url") or ""
    if len(t) < 400 or r.get("duplicate_of") or r.get("low_content"):
        continue
    host = urlsplit(u).netloc.lower()
    k = title_key(r.get("title"), u)
    if len(k) < 12:
        continue
    fam[(host, k, (r.get("lang") or ""))].append((r.get("effective_year"), u, t))

results = []
for key, items in fam.items():
    years = {y for y, _, _ in items if isinstance(y, int)}
    if len(items) < 2 or len(years) < 2:
        continue
    items.sort(key=lambda x: -(x[0] or 0))
    new, old = items[0], items[1]
    snew, sold = sentences(new[2]), sentences(old[2])
    if not snew or not sold:
        continue
    pairs = []
    for s in snew:
        if not VALUE.search(s):
            continue
        near = difflib.get_close_matches(s, sold, n=1, cutoff=0.72)
        if not near:
            continue
        a = near[0]
        va, vb = VALUE.findall(a), VALUE.findall(s)
        if va != vb and a != s:
            pairs.append((a, s))
    if pairs:
        results.append((len(pairs), key, new, old, items, pairs))

results.sort(key=lambda x: -x[0])
print("families where a dated/priced value changed between editions: %d\n" % len(results))
for n, key, new, old, items, pairs in results[:8]:
    host, k, lang = key
    print("=" * 78)
    print("%s | %s | lang=%s | %d editions | %d changed values"
          % (host, k[:52], lang, len(items), n))
    for y, u, t in items[:5]:
        print("   %-5s %7dc %s" % (y, len(t), u[:96]))
    for a, b in pairs[:3]:
        print("\n   OLD (%s): %s" % (old[0], a[:230]))
        print("   NEW (%s): %s" % (new[0], b[:230]))
    print()
