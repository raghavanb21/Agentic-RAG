from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "agentic-rag")
LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING", "true")
FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY", "")
PAGEINDEX_API_KEY: str = os.getenv("PAGEINDEX_API_KEY", "")

os.environ.setdefault("LANGSMITH_TRACING", LANGSMITH_TRACING)
os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGSMITH_API_KEY", LANGSMITH_API_KEY)
if GROQ_API_KEY:
    os.environ.setdefault("GROQ_API_KEY", GROQ_API_KEY)
if PAGEINDEX_API_KEY:
    os.environ.setdefault("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY)

GROQ_GUARD_MODEL: str = "meta-llama/llama-prompt-guard-2-86m"
GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"
GROQ_MAIN_MODEL: str = "llama-3.3-70b-versatile"
EMBED_MODEL: str = "BAAI/bge-base-en-v1.5"

PDF_COLLECTION: str = "pdf_text"
WEB_COLLECTION: str = "web_cache"
# ArXiv uses a vectorless PageIndex (hierarchical ToC JSON), not ChromaDB.

PDF_SIM_THRESHOLD: float = 0.8
BLOG_CRAG_THRESHOLD: float = 0.9
WEB_CRAG_THRESHOLD: float = 0.8
ARXIV_TTL_SECONDS: int = 86400

GUARD_MAX_CHARS: int = 400
MAX_REWRITE_COUNT: int = 1
TOP_K_CHUNKS: int = 5

PAGEINDEX_POLL_INTERVAL_S: float = 5.0
PAGEINDEX_INDEX_MAX_WAIT_S: int = 600
PAGEINDEX_QUERY_POLL_INTERVAL_S: float = 1.0
PAGEINDEX_QUERY_MAX_WAIT_S: int = 90
PAGEINDEX_PARALLEL_WORKERS: int = 3

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150

CHROMA_PERSIST_DIR: str = str(PROJECT_ROOT / "data" / "chroma")
AGENTS_MD_PATH: str = str(PROJECT_ROOT / "agents.md")

PDF_DATA_DIR: str = str(PROJECT_ROOT / "data" / "pdfs")
ARXIV_DATA_DIR: str = str(PROJECT_ROOT / "data" / "arxiv")
ARXIV_INDEX_DIR: str = str(PROJECT_ROOT / "data" / "arxiv_index")
PAGEINDEX_MAPPING_PATH: str = str(PROJECT_ROOT / "data" / "arxiv_index" / "doc_ids.json")
