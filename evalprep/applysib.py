"""Write the scoreable evaluation set: acceptable_urls + the items hit@k can't judge.

Three changes over evaluation_set.fixed.json, each one made per item rather than
in bulk:

  acceptable_urls  the reciprocal-link sibling, but ONLY where sibcheck showed
                   the sibling actually states the gold fact. Two of the eleven
                   candidates do not, and accepting them would have swapped a
                   false negative for a false positive - a worse trade, because
                   a false positive hides a real retrieval failure.

  answer_only      item 12's fact is a boilerplate contact block repeated across
                   53 documents and the question names no course, so no
                   retriever can be expected to pick the target. It leaves the
                   hit@k denominator and stays in the generation grade.

  sibling_note     why a rejected sibling was rejected, so it is not re-added by
                   the next person who runs the pairing.
"""
import json
from pathlib import Path

SRC = "evalprep/evaluation_set.fixed.json"
SIB = "evalprep/siblings.json"
OUT = "data/evaluation_set.v3.json"

# Verdicts from evalprep/sibcheck.txt. SUPPORTS was decided mechanically
# (numbers, months and email addresses survive translation); the rest were read.
ACCEPT = {
    8:  "same three addresses, cc list included",
    12: "same address; item is answer-only anyway",
    18: "'Entro sette giorni prima della data di laurea' - the English gold spells the number out, which is why the numeric check missed it",
    19: "'entro la scadenza in Esse3' - the same deferral, not a date",
    21: "'4 mesi prima dell'appello di laurea'",
    28: "'DURATA: 2 anni'",
    29: "states the same fellowship title in Italian; the fact is present, the language differs",
    31: "'oltre 800 professioniste e professionisti' - the hedge survives translation",
    40: "'frontespizio come nel modello allegato ... file unico, non protetto in formato PDF'",
}
REJECT = {
    3: "the Italian sibling defers to an annex and never states the 66-110 scale; "
       "accepting it would score a retrieval that cannot answer the question as a hit",
    9: "the Italian sibling states the opposite - everyone writes a Relazione finale - "
       "and never carries the exemption the gold answer turns on",
}
ANSWER_ONLY = {12}

items = json.load(open(SRC, encoding="utf-8"))
sibs = json.load(open(SIB, encoding="utf-8"))

out = []
for i in items:
    rec = dict(i)
    sib = sibs.get(i["target_url"], [])
    if i["id"] in ACCEPT and sib:
        rec["acceptable_urls"] = [i["target_url"]] + sib
        rec["sibling_note"] = ACCEPT[i["id"]]
    elif i["id"] in REJECT:
        rec["acceptable_urls"] = [i["target_url"]]
        rec["sibling_rejected"] = sib
        rec["sibling_note"] = REJECT[i["id"]]
    else:
        rec["acceptable_urls"] = [i["target_url"]]
    if i["id"] in ANSWER_ONLY:
        rec["answer_only"] = True
    out.append(rec)

unused = {int(k) for k in list(ACCEPT) + list(REJECT)} - {i["id"] for i in items}
assert not unused, "verdict written for ids not in the set: %s" % unused
paired = [i["id"] for i in out if len(i["acceptable_urls"]) > 1]

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("wrote %s - %d items" % (OUT, len(out)))
print("  %d carry a second acceptable URL: %s" % (len(paired), paired))
print("  %d sibling(s) found but rejected: %s" % (len(REJECT), sorted(REJECT)))
print("  %d answer-only (out of the hit@k denominator): %s"
      % (len(ANSWER_ONLY), sorted(ANSWER_ONLY)))
print("  retrieval-scored items: %d of %d" % (len(out) - len(ANSWER_ONLY), len(out)))
