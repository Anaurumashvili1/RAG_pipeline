"""How many PDFs did pypdf extract with a space between every character?

The FAQ tirocini PDF comes out as "W e l c o m e  t o  t h e  I n t e r n s h i p".
17,489 characters, so it passes every length and low_content gate - but every
token is one letter, so the embedding is meaningless and the document cannot be
retrieved by any phrase in it.

The detector is the one already validated for OCR noise in PROGRESS 3.4:
long_token_ratio, the share of whitespace-separated tokens with 4+ characters.
It was never applied to pypdf output.
"""
import json, collections
from urllib.parse import urlsplit


def long_token_ratio(text: str, sample: int = 4000) -> float:
    toks = text[:sample].split()
    if not toks:
        return 1.0
    return sum(1 for t in toks if len(t) >= 4) / len(toks)


buckets = collections.Counter()
bad, by_ext = [], collections.Counter()
total = 0
for l in open("dataset.ocr.jsonl", encoding="utf-8"):
    r = json.loads(l)
    t = r.get("text") or ""
    if len(t) < 150 or r.get("duplicate_of") or r.get("low_content"):
        continue
    total += 1
    ratio = long_token_ratio(t)
    buckets[round(min(ratio, 0.95), 1)] += 1
    if ratio < 0.20:
        bad.append((round(ratio, 3), r.get("extractor"), len(t),
                    (r.get("title") or "")[:52], r.get("url") or ""))
        by_ext[r.get("extractor")] += 1

print("index-eligible documents scanned: %d" % total)
print("\nlong_token_ratio distribution:")
for k in sorted(buckets):
    print("  %.1f  %6d  %s" % (k, buckets[k], "#" * min(60, buckets[k] // 300)))

print("\ndocuments below 0.20 (letter-spaced / shredded text): %d" % len(bad))
print("by extractor: %s" % dict(by_ext))
print("\nby host:")
for k, v in collections.Counter(urlsplit(b[4]).netloc for b in bad).most_common(10):
    print("  %-32s %d" % (k, v))
print("\nworst 15:")
for ratio, ext, n, ti, u in sorted(bad)[:15]:
    print("  %.3f %-10s %7dc %-52s" % (ratio, ext, n, ti))
    print("        %s" % u[:110])
