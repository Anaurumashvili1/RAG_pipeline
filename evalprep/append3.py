import json

SRC = OUT = "evalprep/evaluation_set.fixed.json"
CORPUS = "dataset.v2.jsonl"

ENERGY = "https://corsi.unitn.it/sites/cds/files/2026-07/regolamento-didattico-lm-energy-engineering-2026.pdf"
GUIDA = ("https://www.giurisprudenza.unitn.it/alfresco/download/workspace/SpacesStore/"
         "39e382d7-76a2-42fc-a382-d776a2e2fcfd/GUIDA%20GIURISPRUDENZA%202025-26_PORTALE.pdf")
STALE_FRAGS = ["02_Guida%20Magistrale%202007-08", "04_Guida%20Magistrale%202008-09",
               "05_Guida%20Magistrale%202009-10", "03_Guida%20Specialistica%202008-09",
               "06_Guida%20Specialistica%202009-10",
               "01_Guida%20Triennale%20e%20Specialistica"]

meta, stale = {}, []
for l in open(CORPUS, encoding="utf-8"):
    r = json.loads(l)
    u = r.get("url") or ""
    if u in (ENERGY, GUIDA):
        meta[u] = (len(r.get("text") or ""), r.get("lang"), r.get("effective_year"))
    if any(f in u for f in STALE_FRAGS):
        stale.append((r.get("effective_year"), len(r.get("text") or ""), u))

stale.sort(key=lambda x: -x[1])
for u in (ENERGY, GUIDA):
    assert u in meta, "target not found: %s" % u
assert len(stale) == 6, "expected 6 vintage guides, found %d" % len(stale)

items = json.load(open(SRC, encoding="utf-8"))
nxt = max(i["id"] for i in items) + 1


def rec(u, question, gold, objective, title, extra=None):
    chars, lang, year = meta[u]
    r = {"id": None, "question": question, "gold_answer": gold, "objective": objective,
         "target_url": u, "title": title,
         "corpus_chars": chars, "corpus_lang": lang, "corpus_year": year}
    if extra:
        r.update(extra)
    return r


new = [
    rec(ENERGY,
        "What language do I have to write my thesis in for the master's in Energy Engineering?",
        "In English. The final exam consists of the discussion of an original thesis written in English.",
        "counterintuitive for a degree taught across Trento and Bolzano in Italian and German; the plausible guess is Italian. Source is a two-column IT/DE PDF whose columns pypdf interleaves, so the surrounding sentence is truncated - the fact must be found inside one clause.",
        "REGOLAMENTO DIDATTICO LM INGEGNERIA ENERGETICA 2026"),
    rec(ENERGY,
        "When do I have to submit my study plan for the master's in Energy Engineering?",
        "When enrolling in the second year of the course. You may submit an individual study plan with adequate justification, subject to approval by the Consiglio di Corso di Studi Interateneo.",
        "procedural timing plus the exception clause; answering only 'second year' drops the individual-plan route. The rule is word-identical in all five editions of this regolamento, so retrieval cannot be scored on which edition it found.",
        "REGOLAMENTO DIDATTICO LM INGEGNERIA ENERGETICA 2026"),
    rec(GUIDA,
        "How many credits is the final thesis worth in the five-year Law degree at Trento?",
        "22 credits.",
        "TEMPORAL, and the answer itself exposes staleness. The value changed from 20 to 22 credits. Six Giurisprudenza guides from 2007-2010 still state 'Prova finale 20' and carry effective_year 2026 (taken from their upload path), while the current 2025-26 guide carries 2025 (from its academic_year). recency_penalty therefore scores the obsolete guides 1.0 and the correct one 0.5, and the vintage guides are ~600k chars against 97k, so they contribute far more chunks. An answer of '20 credits' means the decay picked the wrong document.",
        "GUIDA DELLA FACOLTA DI GIURISPRUDENZA 2025-26",
        {"stale_urls": [u for _, _, u in stale],
         "stale_answer": "20 credits",
         "scoring": "temporal"}),
]

for i, r in enumerate(new):
    r["id"] = nxt + i
items.extend(new)
json.dump(items, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

print("%d items (added %d)" % (len(items), len(new)))
print("\nvintage Law guides listed as stale_urls:")
for y, c, u in stale:
    print("   effective_year=%-5s %7dc  %s" % (y, c, u.split("/")[-1][:64]))
print("\nnew items:")
for r in new:
    print("   id %-3s %s" % (r["id"], r["question"][:82]))
