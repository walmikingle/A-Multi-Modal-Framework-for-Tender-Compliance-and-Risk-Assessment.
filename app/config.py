from pathlib import Path
import os

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CONFIG_FILE = (
    PROJECT_ROOT
    / "config.yaml"
)

ENV_FILE = (
    PROJECT_ROOT
    / ".env"
)

load_dotenv(
    ENV_FILE
)


# ============================================================
# LOAD CONFIGURATION
# ============================================================

if not CONFIG_FILE.exists():

    raise FileNotFoundError(
        f"Configuration file not found: "
        f"{CONFIG_FILE}"
    )


with open(
    CONFIG_FILE,
    "r",
    encoding="utf-8"
) as file:

    CONFIG = yaml.safe_load(
        file
    )


if not isinstance(
    CONFIG,
    dict
):

    raise ValueError(
        "config.yaml must contain "
        "a YAML mapping."
    )


# ============================================================
# PARSER
# ============================================================

PARSER = str(
    CONFIG.get(
        "parser",
        "docling"
    )
).strip().lower()


SUPPORTED_PARSERS = {
    "docling",
    "pymupdf"
}


if PARSER not in SUPPORTED_PARSERS:

    raise ValueError(
        f"Unsupported parser '{PARSER}'. "
        f"Supported parsers: "
        f"{', '.join(sorted(SUPPORTED_PARSERS))}"
    )


# ============================================================
# CHUNKING
# ============================================================

CHUNK_SIZE = int(
    CONFIG.get(
        "chunk_size",
        700
    )
)

CHUNK_OVERLAP = int(
    CONFIG.get(
        "chunk_overlap",
        200
    )
)


if CHUNK_SIZE <= 0:

    raise ValueError(
        "chunk_size must be greater than 0."
    )


if CHUNK_OVERLAP < 0:

    raise ValueError(
        "chunk_overlap cannot be negative."
    )


if CHUNK_OVERLAP >= CHUNK_SIZE:

    raise ValueError(
        "chunk_overlap must be smaller "
        "than chunk_size."
    )


# ============================================================
# DENSE EMBEDDINGS
# ============================================================

EMBEDDING_MODEL = str(
    CONFIG.get(
        "embedding_model",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
).strip()


EMBEDDING_DIMENSION = int(
    CONFIG.get(
        "embedding_dimension",
        384
    )
)


if EMBEDDING_DIMENSION <= 0:

    raise ValueError(
        "embedding_dimension must be "
        "greater than 0."
    )


# ============================================================
# SPARSE RETRIEVAL
# ============================================================

SPARSE_MODEL = str(
    CONFIG.get(
        "sparse_model",
        "ibm-granite/granite-embedding-30m-sparse"
    )
).strip()


SPARSE_DOCUMENT_MAX_ACTIVE_DIMS = int(
    CONFIG.get(
        "sparse_document_max_active_dims",
        192
    )
)


SPARSE_QUERY_MAX_ACTIVE_DIMS = int(
    CONFIG.get(
        "sparse_query_max_active_dims",
        50
    )
)


# ============================================================
# RETRIEVAL
# ============================================================

RETRIEVAL_TOP_K = int(
    CONFIG.get(
        "retrieval_top_k",
        10
    )
)


RERANK_TOP_K = int(
    CONFIG.get(
        "rerank_top_k",
        5
    )
)


if RETRIEVAL_TOP_K <= 0:

    raise ValueError(
        "retrieval_top_k must be greater than 0."
    )


if RERANK_TOP_K <= 0:

    raise ValueError(
        "rerank_top_k must be greater than 0."
    )


if RERANK_TOP_K > RETRIEVAL_TOP_K:

    raise ValueError(
        "rerank_top_k cannot be greater than "
        "retrieval_top_k."
    )


# ============================================================
# RERANKER
# ============================================================

RERANKER_MODEL = str(
    CONFIG.get(
        "reranker_model",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
).strip()


# ============================================================
# LLM
# ============================================================

LLM_PROVIDER = str(
    CONFIG.get(
        "llm_provider",
        "groq"
    )
).strip().lower()


SUPPORTED_LLM_PROVIDERS = {
    "groq"
}


if LLM_PROVIDER not in SUPPORTED_LLM_PROVIDERS:

    raise ValueError(
        f"Unsupported LLM provider "
        f"'{LLM_PROVIDER}'. "
        f"Supported providers: "
        f"{', '.join(sorted(SUPPORTED_LLM_PROVIDERS))}"
    )


LLM_MODEL = str(
    CONFIG.get(
        "llm_model",
        "openai/gpt-oss-20b"
    )
).strip()


# Keep the secret outside config.yaml.
GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)


# ============================================================
# DATA DIRECTORY
# ============================================================

DATA_DIR = (
    PROJECT_ROOT
    / "data"
)