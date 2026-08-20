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


@dataclass
class RetrievalCfg:
    similarity_top_k: int = 20
    max_pages: int = 5
    dedup_by: str = "doc_group"
    prefer_query_language: bool = True
    chunk_char_limit: int = 1200


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
