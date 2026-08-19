"""FAISS index construction and loading (Colab cell 5)."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore

from .config import Config
from .embeddings import configure_settings, embedding_dim

MANIFEST = "index_manifest.json"


def build_index(nodes: list[TextNode], cfg: Config) -> VectorStoreIndex:
    """Embed nodes and persist a FAISS index to disk."""
    embed_model = configure_settings(cfg.embedding)
    dim = embedding_dim(embed_model)
    print(f"[index] embedding dimension detected: {dim}")

    faiss_index = faiss.IndexFlatIP(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"[index] embedding {len(nodes)} nodes (this is the slow part)")
    index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)

    out = Path(cfg.paths.index_dir)
    out.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(out))

    # Record what produced this index so a mismatched embedding model is caught
    # at load time rather than silently returning nonsense.
    (out / MANIFEST).write_text(
        json.dumps(
            {
                "embedding_model": cfg.embedding.model_name,
                "dimension": dim,
                "num_nodes": len(nodes),
                "chunk_size": cfg.chunking.chunk_size,
                "chunk_overlap": cfg.chunking.chunk_overlap,
                "header_injected": cfg.chunking.inject_header,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[index] persisted to {out}")
    return index


def load_index(cfg: Config) -> VectorStoreIndex:
    """Load a previously persisted index, verifying the embedding model matches."""
    persist_dir = Path(cfg.paths.index_dir)
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"No index at {persist_dir}. Run: python scripts/build_index.py"
        )

    manifest_path = persist_dir / MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("embedding_model") != cfg.embedding.model_name:
            raise ValueError(
                f"Index was built with '{manifest.get('embedding_model')}' but config "
                f"says '{cfg.embedding.model_name}'. Rebuild the index or fix config.yaml."
            )

    configure_settings(cfg.embedding)
    vector_store = FaissVectorStore.from_persist_dir(str(persist_dir))
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=str(persist_dir)
    )
    return load_index_from_storage(storage_context)
