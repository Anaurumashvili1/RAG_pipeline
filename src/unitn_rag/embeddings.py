"""Embedding model factory.

Swapping models was a pain in the notebook because the FAISS dimension was
hardcoded to 768. Here the dimension is measured from the model itself, so
changing ``embedding.model_name`` in config.yaml is the only edit required.
"""

from __future__ import annotations

import inspect

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .config import EmbeddingCfg, resolve_device

# bge-*-en-v1.5 was trained with an asymmetric query prefix; bge-m3 was not.
_BGE_EN_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _needs_query_instruction(model_name: str) -> bool:
    n = model_name.lower()
    return "bge" in n and "-en" in n and "m3" not in n


def resolve_dtype(requested: str, device: str):
    """Torch dtype for the embedding model, or None to leave the default.

    Half precision is the largest single speedup available on GPU (measured
    4.6x on a GB10) and is a poor idea on CPU, where fp16 kernels are emulated.
    """
    if requested and requested != "auto":
        import torch

        return {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(requested.lower())

    if device == "cuda":
        import torch

        return torch.float16
    return None


def build_embed_model(cfg: EmbeddingCfg) -> HuggingFaceEmbedding:
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)

    kwargs = {
        "model_name": cfg.model_name,
        "device": device,
        "embed_batch_size": cfg.batch_size,
        # Cosine similarity via inner product requires unit-norm vectors.
        "normalize": True,
    }
    if dtype is not None:
        # Passed through to SentenceTransformer(model_kwargs=...). Queries and
        # chunks must use the same precision - an index built in fp16 and
        # queried in fp32 compares subtly different vectors.
        kwargs["model_kwargs"] = {"torch_dtype": dtype}
    if _needs_query_instruction(cfg.model_name):
        kwargs["query_instruction"] = _BGE_EN_QUERY_INSTRUCTION

    # Tolerate signature differences between llama-index versions.
    accepted = set(inspect.signature(HuggingFaceEmbedding.__init__).parameters)
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}

    dt = getattr(dtype, "__name__", None) or str(dtype).replace("torch.", "")
    print(f"[embeddings] loading {cfg.model_name} on {device}"
          f"{f' ({dt})' if dtype is not None else ''}")
    return HuggingFaceEmbedding(**kwargs)


def embedding_dim(embed_model) -> int:
    """Measure output dimensionality instead of hardcoding it."""
    return len(embed_model.get_query_embedding("dimension probe"))


def configure_settings(cfg: EmbeddingCfg):
    """Install the embedding model globally and disable LlamaIndex's default LLM.

    We call the LLM ourselves through an OpenAI-compatible client, so LlamaIndex
    must not try to instantiate an OpenAI LLM (which would demand OPENAI_API_KEY).
    """
    embed_model = build_embed_model(cfg)
    Settings.embed_model = embed_model
    Settings.llm = None
    return embed_model
