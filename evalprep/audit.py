import json, csv, collections
keep = {}
for l in open("dataset.jsonl", encoding="utf-8"):
    r = json.loads(l)
    keep[r.get("url", "")] = (
        bool(r.get("duplicate_of")), bool(r.get("low_content")),
        bool(r.get("boilerplate")), len(r.get("text") or ""),
        (r.get("lang") or "").lower(),
    )
eligible = {json.loads(l)["url"] for l in open("evalprep/slim_index.jsonl", encoding="utf-8")}
rows = list(csv.DictReader(open("evalprep/cand_targets.csv", encoding="utf-8")))
bad = collections.Counter()
lines = []
for t in rows:
    u = t["expected_url"]
    dup, low, bp, n, lang = keep.get(u, (False, False, False, 0, ""))
    ok = u in eligible
    if not ok:
        why = ("duplicate_of" if dup else "low_content" if low else "boilerplate" if bp
               else "short_text" if n < 150 else
               ("lang_" + lang) if lang not in ("it", "en", "") else "unknown")
        bad[t["category"] + " / " + why] += 1
    lines.append((t, ok, n))
with open("evalprep/audit.txt", "w", encoding="utf-8") as f:
    f.write("=== targets NOT index-eligible ===\n")
    for k, v in bad.most_common():
        f.write("  %3d  %s\n" % (v, k))
    f.write("\n%d of %d targets un-retrievable by construction\n" % (sum(bad.values()), len(rows)))
    for cat in ("admissions", "course", "scholarship", "bilingual", "temporal", "trap"):
        f.write("\n--- %s ---\n" % cat)
        for t, ok, n in lines:
            if t["category"] != cat:
                continue
            f.write("  %s %-2s %-5s %-11s %7dc  %-55s %s\n" % (
                "OK  " if ok else "DROP", t["lang"], t["year"],
                (t["doc_type"] or "-"), n, t["page_title"][:55], t["expected_url"][:78]))
print("written")
