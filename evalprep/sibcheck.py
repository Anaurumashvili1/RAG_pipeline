"""Does the sibling actually state the gold fact? Adding it blind is unsafe.

A reciprocal link proves two pages are the same page in two languages. It does
NOT prove they carry the same content. Item 3 is the counter-example already
written into the set: the Italian sibling defers to an annex and never states
the grading scale, so accepting it as a hit would turn a real retrieval failure
into a pass - trading the false negative for a false positive.

So each candidate is checked against the gold answer before it is accepted:
pull the numbers and dates out of the gold answer and look for them in the
sibling's text. Numeric facts decide themselves; anything else is left for a
human, marked REVIEW.
"""
import json, re

CORPUS = "dataset.v2.jsonl"
EVAL = "evalprep/evaluation_set.fixed.json"
SIB = "evalprep/siblings.json"

items = json.load(open(EVAL, encoding="utf-8"))
sibs = json.load(open(SIB, encoding="utf-8"))
want = {u for v in sibs.values() for u in v} | set(sibs)

text = {}
for line in open(CORPUS, encoding="utf-8"):
    r = json.loads(line)
    if (r.get("url") or "") in want:
        text[r["url"]] = r.get("text") or ""

EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
MONTHS = ("gennaio febbraio marzo aprile maggio giugno luglio agosto settembre "
          "ottobre novembre dicembre january february march april may june july "
          "august september october november december").split()
NUM = re.compile(r"\b\d+(?:[.,/]\d+)*\b")


def facts(ans):
    """The checkable atoms of a gold answer.

    Numbers, month names and email addresses survive translation unchanged, so
    their presence or absence in the sibling is decidable without reading it.
    Prose does not, which is why the rest falls to REVIEW.
    """
    nums = [n for n in NUM.findall(ans) if n not in ("1",)]
    months = [m for m in MONTHS if m in ans.lower()]
    mails = EMAIL.findall(ans)
    return nums, months, mails


def found(hay, token):
    return re.search(r"(?<!\d)" + re.escape(token) + r"(?!\d)", hay) is not None


verdicts = {}
for it in items:
    target = it["target_url"]
    cands = sibs.get(target)
    if not cands:
        continue
    ans = it["gold_answer"]
    nums, months, mails = facts(ans)
    print("=" * 78)
    print("id %-3s  %s" % (it["id"], it["question"][:88]))
    print("  gold: %s" % ans[:220])
    for c in cands:
        t = text.get(c, "")
        low = t.lower()
        miss_n = [n for n in nums if not found(t, n)]
        miss_m = [m for m in months if m not in low]
        miss_e = [e for e in mails if e.lower() not in low]
        miss_n += miss_e
        nums = nums + mails
        if not t:
            v = "ABSENT"
        elif not nums and not months:
            v = "REVIEW"                       # nothing mechanically checkable
        elif not miss_n and not miss_m:
            v = "SUPPORTS"
        elif len(miss_n) + len(miss_m) == len(nums) + len(months):
            v = "CONTRADICTS-or-SILENT"
        else:
            v = "PARTIAL"
        verdicts.setdefault(str(it["id"]), []).append({"url": c, "verdict": v})
        print("  -> %-22s %s  (%d chars)" % (v, c[:80], len(t)))
        if nums or months:
            print("     checked %s%s   missing %s%s"
                  % (nums, months, miss_n, miss_m))
        # show the sibling's own sentences around the first found number
        for tok in (nums + months):
            if found(t, tok) or tok in low:
                m = re.search(r"[^.\n]{0,110}" + re.escape(tok) + r"[^.\n]{0,110}", t)
                if m:
                    print("     ...%s..." % " ".join(m.group(0).split())[:200])
                break
        if v == "REVIEW":
            print("     --- sibling text, for the human call ---")
            for ln in " ".join(t.split()).split(". "):
                if ln.strip():
                    print("       %s." % ln.strip()[:170])

json.dump({k: v for k, v in verdicts.items()},
          open("evalprep/sibcheck.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\nwrote evalprep/sibcheck.json")
