# Agentic RAG — AI Concepts Research Assistant

An agentic Retrieval-Augmented Generation (RAG) system for AI concepts. It uses a LangGraph state machine to route user queries through one of four retrieval paths — local PDF corpus, ArXiv paper cache, an explicit blog URL, or direct LLM — with a jailbreak guard, prompt-injection filter per retrieved document, and Corrective-RAG (CRAG) grading with a single rewrite-and-retry. The final answer is streamed token-by-token to a Streamlit chat UI.

## Architecture

The orchestrator (`llama-3.1-8b-instant` via Groq) selects one of four routes:

- **pdf_chromadb** — local PDF corpus (ChromaDB collection `pdf_text`, `bge-base-en-v1.5` embeddings). If max similarity < 0.8, rewrite once; if still below, fall through to a Firecrawl web search.
- **arxiv** — **vectorless PageIndex** retrieval, via the [`pageindex`](https://pypi.org/project/pageindex/) cloud SDK. At ingest time each PDF in `data/arxiv/` is uploaded via `PageIndexClient.submit_document(pdf)`; the hierarchical ToC tree and the page text are built and persisted on VectifyAI's servers (not on disk). The only thing we store locally is `data/arxiv_index/doc_ids.json` — a `arxiv_id → doc_id` map. Submissions and polls run in a thread pool, so ingesting N papers takes about the wall time of the slowest one, not the sum. At query time the `arxiv_node` calls `submit_query(doc_id, question)` and polls `get_retrieval(retrieval_id)`; PageIndex's server-side agent walks the tree, picks relevant page ranges, and returns them. If the user mentions an arxiv id that isn't in the mapping, `arxiv_tool.fetch_and_index(arxiv_id)` downloads the paper via the `arxiv` Python library and submits it on the fly. No chunking, no embeddings, no vector database for arxiv.
- **blog_url** — Firecrawl scrapes the supplied URL. Each fetched doc passes through a prompt-injection check; flagged docs are skipped. If all are flagged the route falls back to `direct_llm`. Clean docs are CRAG-graded against a 0.9 threshold.
- **direct_llm** — greeting / small-talk / non-AI query. Goes straight to the streaming answer LLM with no retrieval context.

The pipeline:

```
chat input → normalize → jailbreak guard → orchestrator
                                            ├─ pdf_chromadb → CRAG → maybe web fallback
                                            ├─ arxiv (cache or fetch)
                                            ├─ blog_url (Firecrawl + injection + CRAG)
                                            └─ direct_llm
                                                  ↓
                                            citation builder
                                                  ↓
                                            streaming answer (Groq SSE)
                                                  ↓
                                            latency • sources • skipped
```

## Folder Structure

```
agentic-rag/
├── .env                          # API keys
├── .env.example                  # template
├── requirements.txt              # dependencies
├── commands.txt                  # step-by-step macOS setup
├── README.md                     # this file
├── CLAUDE.md                     # guidance for Claude Code
├── agents.md                     # single source of truth for all system prompts
├── app.py                        # Streamlit chat UI
│
├── data/
│   ├── pdfs/                     # 3 AI-concepts PDFs
│   ├── arxiv/                    # 3 ArXiv PDFs named arxiv_id.pdf
│   ├── arxiv_index/
│   │   └── doc_ids.json          # local map: arxiv_id → PageIndex doc_id (trees live in the cloud)
│   └── chroma/                   # ChromaDB persistent store (auto-created)
│
├── config/
│   └── settings.py               # constants, model names, thresholds
│
├── ingestion/
│   ├── pdf_ingestor.py           # data/pdfs/ → pdf_text (vector)
│   ├── arxiv_ingestor.py         # data/arxiv/ → PageIndex JSON trees (vectorless)
│   └── ingest.py                 # single entry point
│
├── tools/
│   ├── chromadb_tool.py          # singleton ChromaDB + bge-base embeddings (pdf_text, web_cache only)
│   ├── arxiv_tool.py             # wraps cloud PageIndexClient: submit/poll, parallel batch, retrieve_for_query, arxiv_id→doc_id mapping
│   └── firecrawl_tool.py         # Firecrawl scrape + search
│
├── agents/
│   ├── __init__.py               # parses agents.md into PROMPTS
│   ├── _groq_client.py           # shared Groq client singleton
│   ├── normalizer.py             # pure string ops
│   ├── jailbreak_guard.py        # llama-prompt-guard-2-86m
│   ├── orchestrator.py           # llama-3.1-8b-instant routing
│   ├── injection_checker.py      # per-doc prompt-injection check
│   ├── crag_grader.py            # CRAG grade + rewrite
│   └── answer_generator.py       # llama-3.3-70b-versatile streaming
│
├── graph/
│   ├── state.py                  # AgentState + Pydantic models
│   ├── nodes.py                  # all node functions with @traceable + latency
│   ├── edges.py                  # conditional edge logic
│   └── graph.py                  # build_graph()
│
└── citations/
    └── builder.py                # Citation Pydantic builder
```

## Prerequisites

- macOS with [Homebrew](https://brew.sh)
- [uv](https://github.com/astral-sh/uv) (`brew install uv`)
- A [Groq](https://console.groq.com) API key (free tier)
- A [LangSmith](https://smith.langchain.com) API key (free tier — for tracing)
- A [Firecrawl](https://www.firecrawl.dev) API key (free tier — required for `blog_url` and web-fallback)
- A [PageIndex](https://pageindex.ai) API key (`PAGEINDEX_API_KEY` in `.env` — required for the `arxiv` route)

## Setup

```bash
brew install uv
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env       # then fill in API keys
```

## Running the App

```bash
streamlit run app.py
```

A fresh chat session begins on every browser refresh. Within a session, the LangGraph `MemorySaver` checkpointer preserves conversation context (so follow-up questions inherit prior turns).

## Example Queries

| Route | Query |
| --- | --- |
| direct_llm | `hi` |
| direct_llm | `what is the capital of France` |
| pdf_chromadb | `explain transformers` |
| pdf_chromadb | `what is RLHF` |
| arxiv | `explain attention is all you need 1706.03762` |
| blog_url | `explain this blog https://example.com/rag` |

## Latency Panel

After each response the UI shows:

- a one-line total latency (sum of per-node wall-clock)
- a collapsible expander listing each node's latency in execution order

Latency is measured inside each node with `time.perf_counter()` and accumulated into `state["node_latencies"]`.

## LangSmith Tracing

Every node is decorated with `@traceable` from `langsmith`. With `LANGSMITH_TRACING=true` and a valid `LANGSMITH_API_KEY`, runs appear under the `agentic-rag` project at https://smith.langchain.com.

## Known Limitations

- Streaming happens for `answer_node` and `direct_llm_node` only — these are invoked from `app.py` after the LangGraph state machine completes its non-streaming stages, so token streaming visibly happens after the orchestrator/retrieval steps finish.
- The Groq guard model (`llama-prompt-guard-2-86m`) is a classifier model. Its chat-completion output format is parsed defensively (categorical label OR JSON), and on parse failure the system defaults to "SAFE" to avoid silently dropping legitimate queries.
- The blog rewrite-and-retry re-grades the same scraped document because rescraping a single URL is not useful; only the query is rewritten.
- Firecrawl is required for the `blog_url` and web-fallback paths. Without an API key those paths return empty and the system falls back to `direct_llm`.
- ChromaDB's `where` filter syntax may vary by version; metadata filtering in `has_source` uses the simple key-equals form.
- The arxiv path requires `PAGEINDEX_API_KEY`. PageIndex runs its own LLM server-side for both tree building (at submission time) and the retrieval agent (at query time); this project does NOT run a local agent loop for arxiv. Polling timeouts default to 10 min for indexing and 90 s for query retrieval — see `config/settings.py`.
- `data/arxiv_index/doc_ids.json` is the *only* local persistent state for arxiv. Deleting it forces every paper to be re-submitted on the next ingest run.
