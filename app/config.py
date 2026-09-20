"""Validated application configuration.

All values are read once at import time so invalid deployment configuration
fails before a document is processed or a cache is written.
"""

from pathlib import Path
import os

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(ENV_FILE)

if not CONFIG_FILE.is_file():
    raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")

with CONFIG_FILE.open("r", encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

if not isinstance(CONFIG, dict):
    raise ValueError("config.yaml must contain a YAML mapping.")


def _string_setting(name, default, *, choices=None):
    value = str(CONFIG.get(name, default)).strip()

    if not value:
        raise ValueError(f"{name} cannot be empty.")

    if choices is not None and value.lower() not in choices:
        supported = ", ".join(sorted(choices))
        raise ValueError(
            f"Unsupported {name} '{value}'. Supported values: {supported}"
        )

    return value


def _integer_setting(name, default, *, minimum=1):
    try:
        value = int(CONFIG.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc

    if value < minimum:
        comparison = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {comparison}.")

    return value


# Parsing and chunking
PARSER = _string_setting(
    "parser",
    "docling",
    choices={"docling", "pymupdf"},
).lower()

CHUNK_SIZE = _integer_setting("chunk_size", 700)
CHUNK_OVERLAP = _integer_setting(
    "chunk_overlap",
    200,
    minimum=0,
)

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("chunk_overlap must be smaller than chunk_size.")


# Dense and sparse embedding models
EMBEDDING_MODEL = _string_setting(
    "embedding_model",
    "sentence-transformers/all-MiniLM-L6-v2",
)

EMBEDDING_DIMENSION = _integer_setting(
    "embedding_dimension",
    384,
)

EMBEDDING_BATCH_SIZE = _integer_setting(
    "embedding_batch_size",
    32,
)

SPARSE_MODEL = _string_setting(
    "sparse_model",
    "ibm-granite/granite-embedding-30m-sparse",
)

SPARSE_DOCUMENT_MAX_ACTIVE_DIMS = _integer_setting(
    "sparse_document_max_active_dims",
    192,
)

SPARSE_QUERY_MAX_ACTIVE_DIMS = _integer_setting(
    "sparse_query_max_active_dims",
    50,
)

SPARSE_BATCH_SIZE = _integer_setting(
    "sparse_batch_size",
    16,
)


# Retrieval and context construction
RETRIEVAL_TOP_K = _integer_setting(
    "retrieval_top_k",
    20,
)

RERANK_TOP_K = _integer_setting(
    "rerank_top_k",
    8,
)

if RERANK_TOP_K > RETRIEVAL_TOP_K * 3:
    raise ValueError(
        "rerank_top_k cannot exceed three times retrieval_top_k, "
        "the maximum hybrid candidate pool size."
    )

RRF_K = _integer_setting(
    "rrf_k",
    60,
)

CONTEXT_MAX_CHARS = _integer_setting(
    "context_max_chars",
    16000,
)


# Models and request limits
RERANKER_MODEL = _string_setting(
    "reranker_model",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

LLM_PROVIDER = _string_setting(
    "llm_provider",
    "huggingface",
    choices={"huggingface"},
).lower()

LLM_MODEL = _string_setting(
    "llm_model",
    "Qwen/Qwen3-30B-A3B-Instruct-2507:featherless-ai",
)

MAX_UPLOAD_MB = _integer_setting(
    "max_upload_mb",
    50,
)

MAX_QUESTION_CHARS = _integer_setting(
    "max_question_chars",
    4000,
)


# Keep secrets out of config.yaml.
HF_TOKEN = os.getenv("HF_TOKEN")

if LLM_PROVIDER == "huggingface" and not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found. Set it in .env or the environment."
    )