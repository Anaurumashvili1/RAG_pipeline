"""Configuration loading.

Single source of truth for paths, model names and hyperparameters.
Secrets come from .env, everything else from config.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(p: str | None) -> Path | None:
    if p is None:
        return None
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class Paths:
    corpus: Path
    index_dir: Path
    eval_set: Path
    eval_results: Path


@dataclass
class DataCfg:
    min_chars: int = 150
    max_docs: int | None = None
    drop_duplicates: bool = True
    drop_low_content: bool = True
    drop_boilerplate: bool = True
    keep_languages: list[str] | None = field(default_factory=lambda: ["it", "en"])


@dataclass
class ChunkingCfg:
    chunk_size: int = 512
    chunk_overlap: int = 100
    inject_header: bool = True
    # Hybrid chunking. 0 disables semantic splitting entirely (fixed-size
    # baseline). Above 0, documents at least this long are split semantically.
    semantic_min_chars: int = 0
    semantic_buffer_size: int = 1
    semantic_percentile: int = 95


@dataclass
class EmbeddingCfg:
    model_name: str = "BAAI/bge-m3"
    device: str = "auto"
    batch_size: int = 32
    # auto | float16 | bfloat16 | float32
    # Measured on a GB10: fp32 50 chunks/sec, fp16 231 - a 4.6x difference and
    # the single largest speedup available. 'auto' means fp16 on CUDA, fp32
    # elsewhere, since half precision is slow or unsupported on CPU.
    dtype: str = "auto"


@dataclass
class RetrievalCfg:
    similarity_top_k: int = 20
    max_pages: int = 5
    dedup_by: str = "doc_group"
    prefer_query_language: bool = True
    # Chunks are 512 *tokens* - roughly 1500-2500 characters of Italian, plus the
    # injected TITLE/SOURCE/LANGUAGE/YEAR header. At 1200 the tail of every
    # retrieved chunk was discarded after retrieval had already succeeded, which
    # is why facts kept going missing from pages that ranked first.
    chunk_char_limit: int = 2500
    # Recompute a document's year from its filename at query time. The crawl's
    # effective_year falls back to the upload/crawl year for Alfresco PDFs, so
    # undated 2002 handbooks scored age 0 while the current guide scored age 1.
    resolve_year_from_title: bool = True
    # Mild boost when the question's words appear in the document's URL host or
    # course slug, to separate near-identical sibling pages. 0.0 disables.
    url_affinity_weight: float = 0.25
    # Cap how many of the final selected pages may share a "document family"
    # (same filename with only the edition year differing, e.g. three years of
    # 'regolamento-didattico-lm-hci-20XX.pdf'). 0 disables. Added 2026-09-10:
    # with reranking on, near-duplicate stale editions of one regulation were
    # filling several of max_pages' slots and crowding out the current page.
    # Distinct from dedup_by=doc_group, which only collapses IT/EN translations.
    family_cap: int = 0
    # Rerank retrieved chunks by scoring them jointly with the question.
    # Off by default: turning it on changes what reaches the LLM.
    rerank: bool = False
    rerank_backend: str = "auto"          # auto | cross_encoder | colbert
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # Pool pulled from FAISS when reranking. 200, not 20: measured target
    # depths are #38 at 146, #3 at 132 after translation, #27 at 34. A pool
    # of 100 would miss the first two.
    rerank_top_k: int = 200
    rerank_batch_size: int = 32
    rerank_strip_header: bool = False
    # Freshness. weight 1.0 = the original 1/(1+age); 0.0 disables it.
    # unknown = what an undated document scores; 0.5 punishes missing metadata
    # as if it were staleness.
    recency_weight: float = 1.0
    # "No resolvable edition year" must mean "no evidence of staleness", not
    # "presumed slightly stale". At 0.5 (the old default), any document that
    # merely states the current/next academic year somewhere in its filename -
    # an admission ranking list, an unrelated department's calendar, a call
    # for applications - scores a full, unpenalised 1.0 and beats a correct,
    # relevant, genuinely undated document stuck at 0.85. Six questions in the
    # 2026-09-10 eval regressed on exactly this: #10, #13, #19, #23, #31, #39
    # all lost to a freshly-dated but irrelevant document once recency_weight
    # went from 0 to 0.3. 1.0 makes "unknown" tie with "confidently current"
    # instead of losing to it, so freshness can no longer manufacture an
    # advantage a document's actual relevance didn't earn.
    recency_unknown: float = 1.0
    # Retrieve in both languages. Off by default; needs an LLM client, so the
    # pipeline wires it in and the bare Retriever works without one.
    translate_query: bool = False
    translation_cache: str = "data/translation_cache.json"


@dataclass
class LLMCfg:
    model: str
    temperature: float = 0.0
    max_tokens: int = 700
    timeout: int = 60
    base_url: str = ""
    api_key: str = ""


@dataclass
class Config:
    paths: Paths
    data: DataCfg = field(default_factory=DataCfg)
    chunking: ChunkingCfg = field(default_factory=ChunkingCfg)
    embedding: EmbeddingCfg = field(default_factory=EmbeddingCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    # No default model. A silent fallback (this used to be "gpt-4o-mini", left
    # over from the notebook's OpenAI backend) turns a missing config key into a
    # confusing "model not loaded" from whatever endpoint is configured, instead
    # of saying the model was never set.
    llm: LLMCfg = field(default_factory=lambda: LLMCfg(model=""))


def load_config(path: str | Path = "config.yaml") -> Config:
    """Read config.yaml + .env into a typed Config object."""
    load_dotenv(PROJECT_ROOT / ".env")

    cfg_path = _resolve(str(path))
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    p = raw.get("paths", {})
    paths = Paths(
        corpus=_resolve(p.get("corpus", "dataset.jsonl")),
        index_dir=_resolve(p.get("index_dir", "storage/index")),
        eval_set=_resolve(p.get("eval_set", "evaluation_set.json")),
        eval_results=_resolve(p.get("eval_results", "results/eval_results.json")),
    )

    llm_raw = raw.get("llm", {})
    if not llm_raw.get("model"):
        raise ValueError(
            f"llm.model is not set in {cfg_path}. Run "
            "'python scripts/check_llm.py --list' to see what the endpoint serves."
        )
    llm = LLMCfg(
        model=llm_raw["model"],
        temperature=llm_raw.get("temperature", 0.0),
        max_tokens=llm_raw.get("max_tokens", 700),
        timeout=llm_raw.get("timeout", 60),
        base_url=os.getenv("LLM_BASE_URL", ""),
        api_key=os.getenv("LLM_API_KEY", ""),
    )

    return Config(
        paths=paths,
        data=DataCfg(**raw.get("data", {})),
        chunking=ChunkingCfg(**raw.get("chunking", {})),
        embedding=EmbeddingCfg(**raw.get("embedding", {})),
        retrieval=RetrievalCfg(**raw.get("retrieval", {})),
        llm=llm,
    )


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
