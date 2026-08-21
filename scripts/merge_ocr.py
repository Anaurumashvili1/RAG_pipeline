#!/usr/bin/env python3
"""Fold OCR results back into the corpus.

The crawler left 873 PDFs as zero-text records. ocr_pending.py recovered the
text for most of them; this writes that text into the corpus so the documents
stop being empty.

    python3 scripts/merge_ocr.py                    # report only, writes nothing
    python3 scripts/merge_ocr.py --write            # produce the merged corpus

Never edits the input in place. The output is a new file, so a bad merge costs
a rerun rather than the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_CORPUS = Path("dataset.jsonl")
DEFAULT_OCR = Path.home() / "scraper" / "ocr_output.jsonl"
DEFAULT_OUT = Path("dataset.ocr.jsonl")


def load_ocr(path: Path, include_suspect: bool) -> tuple[dict[str, dict], dict[str, int]]:
    """URL -> OCR record, keeping only results worth writing into the corpus."""
    kept: dict[str, dict] = {}
    counts = {"total": 0, "empty": 0, "noise": 0, "kept": 0}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            counts["total"] += 1

            if not r.get("ok", True):
                continue
            if r.get("ocr_empty"):
                counts["empty"] += 1
                if not include_suspect:
                    continue
            if r.get("ocr_noise"):
                counts["noise"] += 1
                if not include_suspect:
                    continue
            if not (r.get("ocr_text") or "").strip():
                continue

            kept[r["url"]] = r
            counts["kept"] += 1

    return kept, counts


def merge_record(doc: dict, ocr: dict) -> dict:
    """Write OCR text into one corpus record."""
    text = ocr["ocr_text"]
    doc["text"] = text
    doc["text_len"] = len(text)
    doc["extractor"] = "tesseract"

    # The crawler left text_sha256 as None because there was no text layer.
    # Downstream deduplication compares that hash, so leaving it None means two
    # URLs serving the identical scanned PDF both survive - which is exactly
    # what Alfresco does, exposing each file under a UUID and a filename.
    doc["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # The crawler flagged these as low_content because they had no text layer.
    # That is now false, and it matters: the loader drops low_content records,
    # so leaving the flag set would discard every document we just recovered.
    doc["low_content"] = False
    doc["needs_ocr"] = False
    doc["ocr_reason"] = None
    doc["note"] = "ocr_tesseract"

    # Carry the quality signals through, so retrieval can down-weight marginal
    # scans later without re-deriving anything.
    for k in ("ocr_pages", "chars_per_page", "long_token_ratio",
              "one_char_ratio", "mean_token_len", "ocr_suspect",
              "ocr_empty", "ocr_noise", "ocr_lang", "ocr_dpi"):
        if k in ocr:
            doc[k] = ocr[k]

    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--ocr", type=Path, default=DEFAULT_OCR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--write", action="store_true",
                    help="actually write the output; without this it only reports")
    ap.add_argument("--include-suspect", action="store_true",
                    help="also merge documents flagged empty or noise")
    args = ap.parse_args()

    ocr, counts = load_ocr(args.ocr, args.include_suspect)
    print(f"[merge] ocr_output.jsonl: {counts['total']} records, "
          f"{counts['empty']} empty, {counts['noise']} noise, {counts['kept']} to merge")

    merged = 0
    chars_before = chars_after = 0
    still_empty = 0
    unmatched = set(ocr)

    out_f = open(args.out, "w", encoding="utf-8") if args.write else None
    try:
        with open(args.corpus, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                url = doc.get("url", "")

                if url in ocr:
                    chars_before += len(doc.get("text") or "")
                    doc = merge_record(doc, ocr[url])
                    chars_after += len(doc["text"])
                    merged += 1
                    unmatched.discard(url)
                elif doc.get("needs_ocr") and not (doc.get("text") or "").strip():
                    still_empty += 1

                if out_f:
                    out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    finally:
        if out_f:
            out_f.close()

    print(f"[merge] merged {merged} documents")
    print(f"[merge] text: {chars_before:,} chars -> {chars_after:,} chars "
          f"(+{chars_after - chars_before:,})")
    print(f"[merge] {still_empty} scanned PDFs remain without text (teseo and fetch failures)")
    if unmatched:
        print(f"[merge] WARNING {len(unmatched)} OCR results had no matching corpus URL")
        for u in list(unmatched)[:5]:
            print(f"          {u}")

    if args.write:
        print(f"[merge] wrote {args.out}")
    else:
        print("[merge] dry run - nothing written. Re-run with --write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
