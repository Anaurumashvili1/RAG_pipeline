import json, sys
sys.path.insert(0, "scripts")
from eval_candidates_v2 import is_answerable, is_student_facing, topics_of

rows = []
for l in open("dataset.ocr.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r.get("extractor") != "tesseract" or r.get("ocr_suspect"):
        continue
    if not (is_answerable(r) and is_student_facing(r)):
        continue
    rows.append((topics_of(r), len(r.get("text") or ""), r.get("title") or "", r.get("url")))

rows.sort(key=lambda x: (not x[0], -x[1]))
print("OCR docs that are answerable AND student-facing: %d\n" % len(rows))
for tp, n, ti, u in rows[:25]:
    print("  %-24s %7dc  %s" % (",".join(tp) or "-", n, ti[:60]))
    print("        %s" % u[:110])
