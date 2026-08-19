"""Chunking with metadata header injection.

The v1 pipeline embedded raw body text only. That caused the failure mode
documented in the paper: a query matching a page *title* could retrieve the
right page but the wrong chunk, because the title was never part of any vector.

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
from typing import Iterable

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from .data import Doc

_META_KEYS = ["doc_id", "url", "title", "lang", "doc_group_id", "effective_year", "chunk_index"]


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
    Paragraph breaks are preferred boundaries, which approximates the structural
    splitting in your notes without requiring HTML to still be present.
    """
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
    )


def chunk_document(
    doc: Doc,
    splitter: SentenceSplitter,
    inject_header: bool = True,
) -> list[TextNode]:
    """Split one document into embed-ready nodes."""
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
) -> list[TextNode]:
    """Chunk the whole corpus.

    ``deduplicate`` implements the pass from your Chunking note: boilerplate
    blocks (cookie banners, footers, contact strips) repeat across thousands of
    pages. Identical bodies are stored once, with every URL they appeared on
    recorded in ``duplicate_urls``. On a 71k-page corpus this typically removes
    a large share of nodes and keeps retrieval results clean.
    """
    splitter = make_splitter(chunk_size, chunk_overlap)

    nodes: list[TextNode] = []
    by_hash: dict[str, TextNode] = {}

    for doc in docs:
        for node in chunk_document(doc, splitter, inject_header):
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

    return nodes
