"""Rerank retrieved chunks by scoring them jointly with the question.

Dense retrieval compresses the question to one vector before it ever meets a
passage, so a token that ought to be decisive gets averaged away. Measured on
this corpus: for "...external institution for masters students in physics",
the computer science page ranks 1 and physics.unitn.it/node/433 ranks 146,
because 'physics' is one of nine words in a single pooled vector while the CS
page matches 'external', 'research' and 'period' in its own slug.

A reranker sees query and passage together, so a rare decisive token - physics,
CEILS, giurisprudenza - gets its own say.

Two backends, whichever is installed:

  cross_encoder  sentence-transformers + BAAI/bge-reranker-v2-m3. One joint
                 forward pass per pair. Stronger, needs a ~2.3GB download.
  colbert        FlagEmbedding + BGE-M3's multi-vector head, scoring by MaxSim.
                 No download (the weights are already local for embedding) but
                 needs the FlagEmbedding package.

Both are normalised to (0, 1] before returning. This is not cosmetic:
``select_pages`` *multiplies* score by ``recency_penalty``, so a negative score
times 0.5 becomes larger and silently inverts the freshness ranking.
"""
from __future__ import annotations

import math
import re

_BACKEND = None
_MODEL = None
_NAME = None

_HEADER_RE = re.compile(
    r"^(?:TITLE|SOURCE|LANGUAGE|ACADEMIC YEAR):.*(?:\n|$)", re.MULTILINE
)


def passage_text(node, strip_header: bool = False) -> str:
    """Chunk text as the reranker should see it.

    ``inject_header: true`` bakes TITLE/SOURCE/LANGUAGE/ACADEMIC YEAR into the
    chunk itself, so the reranker sees them whether we like it or not. TITLE
    carries the course name and helps; the raw SOURCE URL is mostly noise
    tokens. ``strip_header`` exists to measure which way that trade goes.
    """
    text = node.get_content(metadata_mode="none")
    return _HEADER_RE.sub("", text).strip() if strip_header else text


def _load(model_name: str, backend: str):
    """Import lazily: nothing should be loaded when rerank is off."""
    global _BACKEND, _MODEL, _NAME
    if _MODEL is not None and _NAME == (model_name, backend):
        return _BACKEND, _MODEL

    errors = []

    if backend in ("auto", "cross_encoder"):
        try:
            from sentence_transformers import CrossEncoder

            print(f"[rerank] loading cross-encoder {model_name}")
            _MODEL = CrossEncoder(model_name, max_length=512)
            _BACKEND = "cross_encoder"
            _NAME = (model_name, backend)
            return _BACKEND, _MODEL
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cross_encoder: {type(exc).__name__}: {exc}")

    if backend in ("auto", "colbert"):
        try:
            from FlagEmbedding import BGEM3FlagModel

            print("[rerank] loading BGE-M3 multi-vector head")
            _MODEL = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
            _BACKEND = "colbert"
            _NAME = (model_name, backend)
            return _BACKEND, _MODEL
        except Exception as exc:  # noqa: BLE001
            errors.append(f"colbert: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "no reranker backend available. Install one of:\n"
        "  pip install sentence-transformers   (then BAAI/bge-reranker-v2-m3 downloads on first use)\n"
        "  pip install FlagEmbedding           (reuses the BGE-M3 weights already cached)\n"
        "tried: " + " | ".join(errors)
    )


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def rerank(
    question: str,
    nodes,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    backend: str = "auto",
    strip_header: bool = False,
    batch_size: int = 32,
):
    """Rescore ``nodes`` in place and return them ordered best-first.

    Scores land in (0, 1] so the multiplicative re-scoring in ``select_pages``
    keeps working. Order of the returned list is what ``select_pages`` sees,
    but it re-sorts anyway - the scores are what matter.
    """
    if not nodes:
        return nodes

    kind, model = _load(model_name, backend)
    passages = [passage_text(n, strip_header) for n in nodes]

    if kind == "cross_encoder":
        # Raw outputs are logits; squash them rather than trusting the
        # library's activation default, which varies across versions.
        raw = model.predict(
            [(question, p) for p in passages],
            batch_size=batch_size,
            show_progress_bar=False,
        )
        scores = [_sigmoid(float(s)) for s in raw]
    else:
        q = model.encode([question], return_dense=False, return_sparse=False,
                         return_colbert_vecs=True)["colbert_vecs"][0]
        d = model.encode(passages, batch_size=batch_size, return_dense=False,
                         return_sparse=False, return_colbert_vecs=True)["colbert_vecs"]
        # MaxSim is roughly [-1, 1]; map to (0, 1] on the same footing as above.
        scores = [max(1e-6, (float(model.colbert_score(q, v)) + 1.0) / 2.0) for v in d]

    for n, s in zip(nodes, scores):
        n.score = float(s)
    return sorted(nodes, key=lambda n: n.score or 0.0, reverse=True)
