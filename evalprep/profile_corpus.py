#!/usr/bin/env python3
"""Profile dataset.jsonl and emit a slim index of index-eligible documents.

Applies the same filters load_documents() applies, so the profile describes the
population the eval set can legitimately draw gold documents from.
"""
import json, collections, sys, re
from urllib.parse import urlsplit

SRC = sys.argv[1] if len(sys.argv) > 1 else "dataset.jsonl"
OUT_SLIM = "evalprep/slim_index.jsonl"
OUT_STATS = "evalprep/corpus_profile.json"

stats = {
    "total_lines": 0, "kept": 0,
    "drop_reason": collections.Counter(),
    "doc_type": collections.Counter(),
    "doc_type_kept": collections.Counter(),
    "lang_kept": collections.Counter(),
    "department_kept": collections.Counter(),
    "year_kept": collections.Counter(),
    "extractor_kept": collections.Counter(),
    "content_type_kept": collections.Counter(),
    "host_kept": collections.Counter(),
    "len_bucket_kept": collections.Counter(),
    "keys_seen": collections.Counter(),
}

def bucket(n):
    for b in (150, 400, 800, 1500, 3000, 8000, 20000, 60000):
        if n < b: return f"<{b}"
    return ">=60000"

with open(SRC, encoding="utf-8") as f, open(OUT_SLIM, "w", encoding="utf-8") as out:
    for line in f:
        line = line.strip()
        if not line: continue
        stats["total_lines"] += 1
        r = json.loads(line)
        if stats["total_lines"] <= 5000:
            for k in r: stats["keys_seen"][k] += 1
        dt = r.get("doc_type") or "None"
        stats["doc_type"][dt] += 1

        url = (r.get("url") or "").strip()
        text = r.get("text") or ""
        if not url: stats["drop_reason"]["no_url"] += 1; continue
        if len(text) < 150: stats["drop_reason"]["short_text"] += 1; continue
        if r.get("duplicate_of"): stats["drop_reason"]["duplicate_of"] += 1; continue
        if r.get("low_content"): stats["drop_reason"]["low_content"] += 1; continue
        if r.get("boilerplate"): stats["drop_reason"]["boilerplate"] += 1; continue
        lang = (r.get("lang") or "").strip().lower().split("-")[0]
        if lang and lang not in ("it", "en"):
            stats["drop_reason"][f"lang_{lang}"] += 1; continue

        stats["kept"] += 1
        stats["doc_type_kept"][dt] += 1
        stats["lang_kept"][lang or "unset"] += 1
        stats["department_kept"][r.get("department") or "None"] += 1
        y = r.get("effective_year")
        stats["year_kept"][str(y) if y else "unknown"] += 1
        stats["extractor_kept"][r.get("extractor") or "None"] += 1
        stats["content_type_kept"][(r.get("content_type") or "None").split(";")[0]] += 1
        stats["host_kept"][urlsplit(url).netloc.lower()] += 1
        stats["len_bucket_kept"][bucket(len(text))] += 1

        out.write(json.dumps({
            "url": url,
            "title": (r.get("title") or "")[:200],
            "lang": lang,
            "doc_type": r.get("doc_type"),
            "department": r.get("department"),
            "effective_year": y,
            "academic_year": r.get("academic_year"),
            "content_type": (r.get("content_type") or "").split(";")[0],
            "extractor": r.get("extractor"),
            "last_modified": r.get("last_modified"),
            "text_len": len(text),
            "nav_ratio": r.get("nav_ratio"),
            "snippet": text[:700],
        }, ensure_ascii=False) + "\n")

ser = {k: (dict(v.most_common(120)) if isinstance(v, collections.Counter) else v)
       for k, v in stats.items()}
with open(OUT_STATS, "w", encoding="utf-8") as f:
    json.dump(ser, f, ensure_ascii=False, indent=2)
print("done", stats["total_lines"], stats["kept"])
