"""Chunking with metadata header injection.

The v1 pipeline embedded raw body text only. That caused the failure:
 a query matching a page *title* could retrieve the right page but the wrong chunk,
because the title was never part of any vector.

Here every chunk carries its own header in the embedded text:

    TITLE: Final exam - Master in Computer Science
    SOURCE: https://www.unitn.it/en/...
    LANGUAGE: en
    ACADEMIC YEAR: 2026

    <chunk body>

The header is in the *text*, not just in metadata, so it reaches the embedding
model. Metadata keys are excluded from embedding to avoid duplicating it.
"""

from __future__ import annotations

import hashlib
import time
from typing import Iterable

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LIDocument, TextNode

from .data import Doc

_META_KEYS = ["doc_id", "url", "title", "lang", "doc_group_id", "effective_year",
              "chunk_index", "chunk_strategy"]


def build_header(doc: Doc) -> str:
    lines = [f"TITLE: {doc.title or '(untitled)'}", f"SOURCE: {doc.url}"]
    if doc.lang:
        lines.append(f"LANGUAGE: {doc.lang}")
    if doc.effective_year:
        lines.append(f"ACADEMIC YEAR: {doc.effective_year}")
    return "\n".join(lines)


def make_splitter(chunk_size: int = 512, chunk_overlap: int = 100) -> SentenceSplitter:
    """Sentence-aware splitter.

    Note on sizes: v1 used 256/50. The error analysis showed facts being severed
    from their section headers, so the default here is 512/100 (~20% overlap).
    Paragraph breaks are preferred boundaries - which only works because
    ``clean_text(keep_breaks=True)`` now preserves newlines. It previously
    collapsed them, so ``paragraph_separator`` could never match.
    """
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
    )


def make_semantic_splitter(
    embed_model,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
):
    """Embedding-based splitter: cut where consecutive sentences diverge.

    Applied only to long documents - see ``semantic_min_chars`` in
    ``chunk_documents``. On a short page the percentile threshold is computed
    over that page's own similarity distribution, so *something* always looks
    like a boundary and a single-topic page gets fragmented. Long documents have
    real topic shifts to find.
    """
    from llama_index.core.node_parser import SemanticSplitterNodeParser

    return SemanticSplitterNodeParser(
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        embed_model=embed_model,
    )


def _split_semantic(parser, text: str) -> list[str]:
    """Run the semantic parser and return plain strings.

    Returns the pieces only; node construction stays in one place so both
    strategies produce identically-shaped nodes. If the two paths built nodes
    differently, an ablation between them would be measuring two changes.
    """
    nodes = parser.get_nodes_from_documents([LIDocument(text=text)])
    return [n.get_content() for n in nodes]


def chunk_document(
    doc: Doc,
    splitter: SentenceSplitter,
    inject_header: bool = True,
    semantic_parser=None,
    semantic_min_chars: int = 0,
) -> list[TextNode]:
    """Split one document into embed-ready nodes.

    Routes on document length: long documents go to the semantic parser when one
    is supplied, everything else uses the sentence splitter.
    """
    use_semantic = (
        semantic_parser is not None
        and semantic_min_chars > 0
        and len(doc.text) >= semantic_min_chars
    )

    if use_semantic:
        try:
            pieces = _split_semantic(semantic_parser, doc.text)
        except Exception as e:  # noqa: BLE001 - never lose a document to a parser error
            print(f"[chunk] semantic split failed ({type(e).__name__}), "
                  f"falling back to sentence split: {doc.url}")
            pieces = splitter.split_text(doc.text)
    else:
        pieces = splitter.split_text(doc.text)

    header = build_header(doc) if inject_header else ""

    nodes: list[TextNode] = []
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue

        text = f"{header}\n\n{piece}" if header else piece
        node = TextNode(
            text=text,
            metadata={
                "doc_id": doc.doc_id,
                "url": doc.url,
                "title": doc.title,
                "lang": doc.lang,
                "doc_group_id": doc.doc_group_id,
                "effective_year": doc.effective_year,
                "chunk_index": i,
                # Which strategy produced this chunk. Needed for the ablation:
                # without it you cannot tell whether a retrieval win came from
                # semantically-split documents or the fixed-size ones.
                "chunk_strategy": "semantic" if use_semantic else "fixed",
            },
        )
        # Header already carries this information; don't send it twice.
        node.excluded_embed_metadata_keys = list(_META_KEYS)
        node.excluded_llm_metadata_keys = list(_META_KEYS)
        nodes.append(node)

    return nodes


def chunk_documents(
    docs: Iterable[Doc],
    chunk_size: int = 512,
    chunk_overlap: int = 100,
    inject_header: bool = True,
    deduplicate: bool = True,
    semantic_min_chars: int = 0,
    embed_model=None,
    semantic_buffer_size: int = 1,
    semantic_percentile: int = 95,
) -> list[TextNode]:
    """Chunk the whole corpus.

    Identical bodies are stored once, with every URL they appeared on recorded
    in ``duplicate_urls``.

    ``semantic_min_chars`` enables hybrid chunking: documents at or above that
    length are split semantically, shorter ones by sentence count. Set to 0 to
    disable semantic chunking entirely (the fixed-size baseline).

    Why a threshold rather than corpus-wide: the median document here is ~1,540
    characters - a single coherent page. The semantic parser has no minimum size
    and its breakpoint threshold is a percentile over each document's own
    similarity distribution, so it will always find "boundaries" and fragment a
    single-topic page. Long documents (>=8k chars, 81% of corpus text, 96% of
    them PDFs with no headings) have real topic shifts and no other structural
    signal.
    """
    splitter = make_splitter(chunk_size, chunk_overlap)

    semantic_parser = None
    if semantic_min_chars > 0:
        if embed_model is None:
            raise ValueError(
                "semantic_min_chars is set but no embed_model was passed - "
                "semantic chunking needs one to measure sentence similarity"
            )
        semantic_parser = make_semantic_splitter(
            embed_model,
            buffer_size=semantic_buffer_size,
            breakpoint_percentile_threshold=semantic_percentile,
        )

    nodes: list[TextNode] = []
    by_hash: dict[str, TextNode] = {}
    n_semantic = n_fixed = 0

    # Semantic chunking is slow and gives no output of its own: it embeds every
    # sentence, one document at a time. Without this the run looks hung.
    docs = list(docs)
    total = len(docs)
    report_every = 100 if semantic_parser is not None else 5000
    t0 = time.time()

    for n, doc in enumerate(docs, 1):
        if semantic_parser is not None and len(doc.text) >= semantic_min_chars:
            n_semantic += 1
        else:
            n_fixed += 1

        if n % report_every == 0:
            rate = n / max(1e-9, time.time() - t0)
            left = (total - n) / rate / 60
            print(f"[chunk] {n:,}/{total:,} docs · {len(nodes):,} chunks · "
                  f"semantic={n_semantic:,} · ~{left:.0f} min left", flush=True)

        for node in chunk_document(
            doc, splitter, inject_header,
            semantic_parser=semantic_parser,
            semantic_min_chars=semantic_min_chars,
        ):
            if not deduplicate:
                nodes.append(node)
                continue

            body = node.text.split("\n\n", 1)[-1] if inject_header else node.text
            h = hashlib.sha256(body.encode("utf-8")).hexdigest()

            existing = by_hash.get(h)
            if existing is None:
                node.metadata["text_sha256"] = h
                node.metadata["duplicate_urls"] = []
                by_hash[h] = node
                nodes.append(node)
            else:
                dupes = existing.metadata.setdefault("duplicate_urls", [])
                if node.metadata["url"] not in dupes:
                    dupes.append(node.metadata["url"])

    if semantic_parser is not None:
        print(f"[chunk] semantic: {n_semantic:,} docs (>={semantic_min_chars:,} chars) · "
              f"fixed: {n_fixed:,} docs · {len(nodes):,} chunks")

    return nodes
