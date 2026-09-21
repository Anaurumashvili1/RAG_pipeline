#!/usr/bin/env python3
"""Isolate whether recency_weight=0.3 is what pushed the #4/#5/#13/#16 targets
out of the top pages, by re-running retrieval for each with apply_recency
True (current config) vs False (recency multiplier skipped, everything else
identical: same translation union, same rerank, same family_cap).

Usage (from ~/RAG_pipeline, venv active):
    python scripts/diag_recency_regression.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unitn_rag.config import load_config          # noqa: E402
from unitn_rag.pipeline import RagPipeline        # noqa: E402

CASES = [
    (3, "Come vengono valutate la tesi e la discussione nella magistrale in Human-Computer Interaction?",
     "https://corsi.unitn.it/en/human-computer-interaction/graduation/final-exam"),
    (4, "I'm transferring into the CEILS bachelor at Trento. What's the maximum number of ECTS credits they can recognise from my previous university?",
     "https://corsi.unitn.it/sites/cds/files/2025-02/guidelines_trasfer_b-comparative-european_internation_legal_studies.pdf"),
    (5, "Can I transfer into the third year of the CEILS bachelor at Trento?",
     "https://corsi.unitn.it/sites/cds/files/2025-02/guidelines_trasfer_b-comparative-european_internation_legal_studies.pdf"),
    (13, "I'm flying to a conference on university business. Am I allowed to book anything above economy class?",
     "https://www.centro3a.unitn.it/alfresco/download/workspace/SpacesStore/9c17f239-218a-44d9-9d49-6f9309669185/regolamento%20missioni_Engl.pdf"),
    (16, "When are the final assessment and thesis defense sessions for the Five-year Degree in Law for academic year 2023/2024?",
     "https://www.giurisprudenza.unitn.it/node/1025"),
    # 2026-09-10 (continued): six more regressions from the SAME root cause -
    # an irrelevant document that merely states the current/next academic
    # year in its filename got a full 1.0 recency multiplier (age~0), beating
    # a correct-but-undated document stuck at recency_unknown's old 0.5.
    (10, "The company and my university supervisor have both signed my training project. Is there a deadline for sending it back to the Job Guidance Office?",
     "https://www.jobguidance.unitn.it/alfresco/download/workspace/SpacesStore/3baf6ad6-f5f3-495a-af6a-d6f5f3695a60/FAQ%20tirocini%20inglese.pdf"),
    (19, "What is the deadline for submitting the graduation application in Esse3 for computer science?",
     "https://corsi.unitn.it/en/computer-science-master/graduation/graduation-calendar"),
    (23, "When does the classes of second semester start for the faculity of law?",
     "https://www.giurisprudenza.unitn.it/node/997"),
    (31, "How many employees does University of Trento have?",
     "https://lavoraconnoi.unitn.it/en/pta-cel"),
    (39, "What documents are required to apply online for the internship at CIMeC?",
     "https://www.cimec.unitn.it/node/202"),
]


def find_rank(pages, target_url):
    for p in pages:
        if p.url == target_url:
            return p
    return None


def show(label, pages, target_url):
    hit = find_rank(pages, target_url)
    print(f"  [{label}] target rank: {hit.rank if hit else 'NOT IN TOP ' + str(len(pages))}"
          f"{'  year=' + str(hit.effective_year) + ' score=' + format(hit.score, '.4f') if hit else ''}")
    print(f"  [{label}] top 8:")
    for p in pages[:8]:
        marker = " <== TARGET" if p.url == target_url else ""
        print(f"      #{p.rank:>2}  score={p.score:.4f}  year={p.effective_year}  {p.url}{marker}")


def main() -> None:
    cfg = load_config("config.yaml")
    print(f"[diag] recency_weight in config: {cfg.retrieval.recency_weight}")
    print(f"[diag] family_cap in config    : {getattr(cfg.retrieval, 'family_cap', 0)}")
    pipeline = RagPipeline(cfg, guardrail=False)
    retriever = pipeline.retriever

    for qid, question, target_url in CASES:
        print(f"\n=== #{qid}: {question}")
        print(f"    target: {target_url}")
        with_recency = retriever.retrieve(question, max_pages=20, apply_recency=True)
        no_recency = retriever.retrieve(question, max_pages=20, apply_recency=False)
        show("apply_recency=True  (weight=%.2f, current)" % cfg.retrieval.recency_weight, with_recency, target_url)
        show("apply_recency=False (recency off)          ", no_recency, target_url)


if __name__ == "__main__":
    main()
