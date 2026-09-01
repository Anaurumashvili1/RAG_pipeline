"""Drop old_id and add `objective`: what each item is actually testing.

The objective is the reason the item exists. Without it, a failure is just a red
cell - you cannot tell whether retrieval missed, generation flattened a
conditional, or the metric is unfair to the item.
"""
import json

SRC = "evalprep/evaluation_set.fixed.json"
OUT = "evalprep/evaluation_set.fixed.json"

OBJ = {
    0:  "single fact under distractor pressure: the same course has a non-EU track page with a different deadline, and 33 courses share this leaf path. English query against an Italian-only page.",
    1:  "conditional branch: the 22/30 minimum is not the whole answer. A bare '22/30' is wrong; the 22-24/30 band requires the Collegio review plus colloquio.",
    2:  "counterintuitive fact buried in a list; the plausible default answer is 'no'. Tests whether one embedded sentence is found rather than the page summarised.",
    3:  "cross-language: Italian question, English-only source. The Italian sibling defers to an annex and does not state the scale. Near-identical sibling pages give a conflicting 0-110 scale.",
    4:  "two-part conditional keyed to language of instruction (30 vs 60 ECTS). Answering '60' alone is the expected failure.",
    5:  "negative fact. RAG systems invent a procedure rather than state that one does not exist; a described third-year transfer route is a hallucination.",
    6:  "numeric lookup from a table on a department grant page.",
    7:  "role attribution unique to this page; distinguishes the coordinator from the two other named delegates.",
    8:  "enumeration: the cc addresses are dropped. The English source also carries a broken address (gestione.assicuazioni); the gold uses the correct Italian spelling deliberately.",
    9:  "exemption stated once at the end, contradicting the page's own headline instruction that everyone writes a final report.",
    10: "precise procedural deadline inside a long FAQ.",
    11: "two numbers keyed to degree level (15 vs 30 days) plus a 12-month window. Also stated on two HTML pages, so retrieval may legitimately land elsewhere.",
    12: "ANSWER-ONLY. The address appears in 53 documents (boilerplate contact block) and the question names no course, so no retriever can pick this target. Tests discrimination between the three emails on the page. Exclude from hit@k.",
    13: "OCR-only: this document had 0 characters before the OCR pass. Retrieval-completeness test - proves recovered text became reachable.",
    14: "the English version of this page mistranslates the hours (says Monday-Friday). Gold follows the correct Italian source. Tests whether the answer is right, not merely faithful to the retrieved page.",
    15: "multi-item enumeration from a research page; partial lists are the expected failure.",
    16: "date list scoped to one academic year - the page also lists 2024/2025, so the wrong year is the adjacent trap.",
    17: "deadline that is defined by reference rather than stated absolutely; tests whether the model reports the dependency instead of inventing a date.",
    18: "relative deadline (seven days before graduation) on a page shared with item 40.",
    19: "deadline defined by reference to Esse3; a fabricated concrete date is the failure.",
    20: "numeric eligibility threshold for Erasmus+ selection.",
    21: "relative deadline expressed in months before the session.",
    22: "time window for modifying the Learning Agreement.",
    23: "calendar date for a specific degree and semester; other degrees' calendars are the distractors.",
    24: "three-tier rule (3 / 2 / 1 CFU). Answering '3 CFU' alone is incomplete. Ten guide editions from 2020-2025 state it, so the live test is whether the current edition wins on recency.",
    25: "person-to-number lookup on a staff list; the page holds many names and numbers.",
    26: "same fact as item 25 reached by role rather than by name - tests whether the role is resolved to the person.",
    27: "cohort reasoning: the current-cohort answer is 6, the correct answer for a 2019 enrolment is 7. Five editions state the rule.",
    28: "basic programme attribute; a control item that should never fail.",
    29: "exact title extraction from a call for applications.",
    30: "project objective, one sentence, from a research-centre page.",
    31: "institutional figure stated approximately ('over 800'); tests whether the hedge is preserved rather than sharpened into a false precision.",
    32: "short definition of a lab's function.",
    33: "conditional: credit transfer depends on the internship being compulsory in the study plan.",
    34: "small enumerated range (one or two semesters).",
    35: "definition of a lab's remit, phrased as a long single sentence.",
    36: "procedural navigation answer (Esse3 menu path) rather than a fact.",
    37: "multi-sentence procedural explanation; tests summarisation without dropping the selection-procedure step.",
    38: "three-item document checklist; partial lists are the expected failure.",
    39: "three-item document checklist with a conditional third item (only if earned at UniTrento).",
    40: "format requirements (cover template, single unprotected PDF); shares its source page with item 18.",
}

items = json.load(open(SRC, encoding="utf-8"))
missing = [i["id"] for i in items if i["id"] not in OBJ]
if missing:
    raise SystemExit("no objective written for ids: %s" % missing)

out = []
for i in items:
    rec = {
        "id": i["id"],
        "question": i["question"],
        "gold_answer": i["gold_answer"],
        "objective": OBJ[i["id"]],
        "target_url": i["target_url"],
        "title": i.get("title", ""),
    }
    if "source_url" in i:
        rec["source_url"] = i["source_url"]
    for k in ("corpus_chars", "corpus_lang", "corpus_year"):
        rec[k] = i.get(k)
    out.append(rec)

json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("%d items written to %s" % (len(out), OUT))
print("old_id removed, objective added to every item")
answer_only = [r["id"] for r in out if r["objective"].startswith("ANSWER-ONLY")]
print("answer-only items (exclude from hit@k): %s" % answer_only)
