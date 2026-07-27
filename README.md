# HealthPlus AI

RAG chatbot for hospital operations. Ingests PDFs (SOPs, pricing, doctors, policies…) into a vector knowledge base and answers questions with streamed, cited, guardrailed replies.

> ⚠️ All hospital data is fictional. Not for clinical use.

---

## 1. Overview

HealthPlus AI ingests hospital-operations PDF documents — Standard Operating Procedures (SOPs), doctor rosters, diagnostic test catalogs, pricing sheets, FAQs, health packages, reports, and policies — into a semantic vector knowledge base and then answers staff and patient questions through a streaming chat UI.

Every answer is:

- **Grounded** — built only from the retrieved document chunks, never from the LLM's general world knowledge about HealthPlus.
- **Cited** — each claim is tagged with `[n]` that maps to a source panel showing file name, page number, category, and similarity score.
- **Guarded** — the system prompt forbids outside knowledge and hard-redirects any diagnosis or treatment question to a qualified doctor.
- **Streamed** — tokens render as they arrive so the UI feels instant even for long answers.

The application supports two LLM backends that can be switched at runtime in the sidebar: **Claude** (Anthropic) and **ChatGPT** (OpenAI).

### Architecture at a glance

```
Users (Doctor · Receptionist · Patient · Admin)
                    │
                    ▼
Presentation ─── Streamlit portal (app.py)
  streaming chat · inline citations · PDF upload
  KB stats · category filter · LLM provider toggle
                    │  direct Python calls (no HTTP tier)
                    ▼
Application ─────── ChatService
  query normalisation → retrieval (≥0.30 cosine floor)
  → 8,000-char numbered context → guardrailed system prompt
  → conversation memory (last 10 turns)
  + KnowledgeBaseService (ingestion & search)
          │                          │
          ▼                          ▼
Vector Database              LLM clients (llm/)
ChromaDB (cosine space)      ClaudeClient  OpenAIClient
BGE Small embeddings         Anthropic SDK  OpenAI SDK
          ▲
          │
Document Pipeline ─── PDFLoader → TextCleaner → DocumentChunker
          ▲
          │
Knowledge Base ── 8 fictional PDFs · 8 categories
  sops · doctors · test_catalog · pricing · faqs
  health_packages · reports · policies
```

The authoritative diagram lives in [`docs/architecture/healthplus-enterprise-architecture.png`](docs/architecture/healthplus-enterprise-architecture.png).

---

## 2. Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| **Python** | 3.11 | `python3 --version` to check |
| **pip** | bundled with Python | used only to install the package |
| **Git** | any | to clone the repo |
| **Anthropic API key** | — | required to use the Claude chat backend; get one at [console.anthropic.com](https://console.anthropic.com) |
| **OpenAI API key** | — | required only if you want the ChatGPT backend |
| **Internet access** | — | needed once to download the BGE Small embedding model (~130 MB) from Hugging Face; fully offline afterwards |

> **Corporate / VPN users:** if your network intercepts TLS (common on corporate proxies), the Hugging Face model download may fail with an SSL error. See [§10 Troubleshooting](#10-troubleshooting) for the fix.

---

## 3. Project Setup

Run these commands from the project root, one at a time.

```bash
# 1. Clone the repository
git clone <repo-url>
cd HealthPlus-AI

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install the package and all dependencies (including dev tools)
pip install -e ".[dev]"

# 4. Copy the environment template and fill in your API keys
cp .env.example .env
#    Open .env in any editor and set ANTHROPIC_API_KEY (and OPENAI_API_KEY if needed)

# 5. Verify configuration and logging work
python scripts/verify_setup.py

# 6. Ingest the sample knowledge-base PDFs into ChromaDB
#    (This step downloads the BGE Small embedding model the first time — ~130 MB)
python scripts/ingest.py data/knowledge_base/pdfs/*.pdf

# 7. Launch the Streamlit chat portal
streamlit run src/healthplus/presentation/app.py
```

After step 7 the browser opens automatically at `http://localhost:8501`. If it does not, open that URL manually.

> **Without the virtual environment activated** you can also run:
> `.venv/bin/python -m streamlit run src/healthplus/presentation/app.py`

---

## 4. Environment Configuration

All configuration is loaded from environment variables. The easiest way to supply them is via a `.env` file in the project root (`.env` is gitignored — your secrets never reach the repository).

Copy `.env.example` to `.env` and edit the values:

```dotenv
# ── LLM ──────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-...    # Required for Claude (default backend)
OPENAI_API_KEY=sk-...           # Required only for the ChatGPT backend

# ── Application ──────────────────────────────────────────────────────────────
HEALTHPLUS_ENVIRONMENT=development   # development | staging | production
HEALTHPLUS_DEBUG=false
HEALTHPLUS_LOG_LEVEL=INFO            # DEBUG | INFO | WARNING | ERROR | CRITICAL

# ── Claude model ─────────────────────────────────────────────────────────────
HEALTHPLUS_CLAUDE_MODEL=claude-sonnet-5
HEALTHPLUS_CLAUDE_MAX_TOKENS=1024    # Max tokens per LLM reply (applies to both backends)

# ── Retrieval / RAG ───────────────────────────────────────────────────────────
HEALTHPLUS_RETRIEVAL_TOP_K=5         # Candidate chunks fetched from ChromaDB
HEALTHPLUS_RETRIEVAL_MIN_SCORE=0.30  # Cosine-similarity floor; hits below this are dropped
HEALTHPLUS_CONTEXT_MAX_CHARS=8000    # Max characters assembled into the system prompt

# ── Conversation ──────────────────────────────────────────────────────────────
HEALTHPLUS_MEMORY_MAX_TURNS=10       # Conversation turns kept in memory per session

# ── Storage paths (relative to project root) ─────────────────────────────────
HEALTHPLUS_DATA_DIR=data
HEALTHPLUS_LOG_DIR=logs
```

Every variable has a sensible default (shown above), so the only values you **must** set are the API keys. All others are optional.

### How configuration works

`config/settings.py` defines a single `Settings` class using **Pydantic Settings**. The class reads values from environment variables (with the `HEALTHPLUS_` prefix) and validates them at startup. The API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) are exceptions — they have no prefix because the Anthropic and OpenAI SDKs also read them directly.

The settings object is built once and cached via `@lru_cache`; every module obtains it with `get_settings()`. No module anywhere in the codebase reads `os.environ` directly.

---

## 5. Commands Reference

### Install

```bash
pip install -e ".[dev]"        # editable install + pytest
pip install -e .               # editable install without dev tools
```

### Run the app

```bash
streamlit run src/healthplus/presentation/app.py
```

### Ingest PDFs into the knowledge base

```bash
# Ingest all corpus PDFs
python scripts/ingest.py data/knowledge_base/pdfs/*.pdf

# Ingest a single file
python scripts/ingest.py data/knowledge_base/pdfs/Pricing.pdf

# Ingest an uploaded/custom PDF (category is derived from filename by convention)
python scripts/ingest.py /path/to/Doctors.pdf
```

Ingestion is **idempotent**: re-ingesting the same file upserts (not duplicates) chunks, because doc and chunk IDs are SHA-256 content hashes.

### Search the knowledge base from the CLI

```bash
# Unfiltered semantic search
python scripts/search.py "how much does an MRI cost"

# Filter to a single category
python scripts/search.py "refund policy" --category policies

# Control the number of results
python scripts/search.py "cardiologist timings" --top-k 5
```

### Evaluate retrieval quality

```bash
python scripts/eval_retrieval.py            # default: top-3 metric
python scripts/eval_retrieval.py --top-k 5 # top-5 metric
```

Exits `0` if the golden-set top-k hit-rate ≥ 70 %, exits `1` otherwise — suitable for a CI gate.

### Preview the document pipeline (dry run — no DB writes)

```bash
python scripts/preview_pipeline.py data/knowledge_base/pdfs/Pricing.pdf
python scripts/preview_pipeline.py data/knowledge_base/pdfs/Pricing.pdf --page 2
python scripts/preview_pipeline.py data/knowledge_base/pdfs/Pricing.pdf --no-embedding
python scripts/preview_pipeline.py data/knowledge_base/pdfs/Pricing.pdf \
    --output data/previews/Pricing.json
```

### Verify configuration and logging

```bash
python scripts/verify_setup.py
```

### Run tests

```bash
python -m pytest                          # all fast tests
python -m pytest -v                       # verbose output
python -m pytest tests/test_chunker.py   # a single file
python -m pytest -m quality              # live retrieval-quality evals (needs data/chroma)
```

### Build a distribution (optional)

```bash
pip install build
python -m build        # creates dist/healthplus_ai-0.1.0-*.whl
```

---

## 6. Folder Structure

```
HealthPlus-AI/
│
├── src/
│   └── healthplus/                    # Installable Python package
│       ├── __init__.py                # Package version
│       │
│       ├── presentation/              # Layer 2 — Streamlit web portal
│       │   └── app.py                 # Entry point; chat UI, sidebar, upload
│       │
│       ├── application/               # Layer 3 — use-case orchestration
│       │   ├── chat_service.py        # RAG chat pipeline conductor
│       │   ├── components.py          # Protocol contracts (port interfaces)
│       │   ├── context_manager.py     # Budget-capped context assembly + citations
│       │   ├── conversation_memory.py # Windowed in-memory per-session chat history
│       │   ├── knowledge_base_service.py  # Ingestion + search facade
│       │   ├── prompt_builder.py      # System-prompt and message-list construction
│       │   ├── query_processor.py     # Query normalisation and validation
│       │   └── retriever.py           # Similarity-floor filtering over KB search
│       │
│       ├── llm/                       # Layer — LLM adapter boundary
│       │   ├── claude_client.py       # Only file that imports the Anthropic SDK
│       │   └── openai_client.py       # Only file that imports the OpenAI SDK
│       │
│       ├── vector_database/           # Layer 4 — embedding + vector storage
│       │   ├── embeddings.py          # Lazy-loaded SentenceTransformer wrapper
│       │   └── vector_store.py        # Only file that imports chromadb
│       │
│       ├── document_pipeline/         # Layer 5 — PDF processing
│       │   ├── chunker.py             # Page-preserving overlapping text splitter
│       │   ├── pdf_loader.py          # PyMuPDF text extraction + SHA-256 doc ID
│       │   └── text_cleaner.py        # Unicode normalisation + whitespace cleanup
│       │
│       ├── knowledge_base/            # Layer 6 — domain models
│       │   └── models.py              # Pydantic models: Document, Chunk, SearchResult, etc.
│       │
│       ├── config/                    # Cross-cutting — typed configuration
│       │   └── settings.py            # Single Settings class; loaded once via get_settings()
│       │
│       └── core/                      # Cross-cutting — shared utilities
│           ├── exceptions.py          # Custom exception hierarchy
│           └── logging.py             # Console + rotating-file logging setup
│
├── scripts/                           # Operational CLI scripts (not part of the package)
│   ├── ingest.py                      # Batch PDF ingestion with Rich report table
│   ├── search.py                      # CLI semantic search
│   ├── eval_retrieval.py              # Golden-set retrieval-quality evaluation
│   ├── preview_pipeline.py            # Dry-run pipeline visualiser
│   ├── verify_setup.py                # Config + logging smoke test
│   └── create_architecture_corpus.py  # One-time script that generated the sample PDFs
│
├── tests/                             # Pytest test suite
│   ├── test_models.py                 # DocumentCategory taxonomy
│   ├── test_text_cleaner.py           # TextCleaner logic
│   ├── test_chunker.py                # DocumentChunker determinism + metadata
│   ├── test_batch_ingestion.py        # Resilient batch: one failure doesn't stop others
│   ├── test_query_processor.py        # Query normalisation + length validation
│   ├── test_retriever.py              # Similarity-floor filtering
│   ├── test_context_manager.py        # Budget capping + deduplication + numbering
│   ├── test_prompt_builder.py         # System-prompt guardrails + message assembly
│   ├── test_conversation_memory.py    # Windowing + isolation + copy semantics
│   ├── test_claude_client.py          # Fail-fast on missing key; injected client
│   ├── test_chat_service.py           # Orchestration with fakes for all collaborators
│   ├── test_rag_integration.py        # End-to-end RAG over a real (isolated) ChromaDB
│   └── test_retrieval_quality.py      # Golden-set hit-rate eval against live index
│
├── data/
│   ├── knowledge_base/pdfs/           # 8 fictional hospital PDFs (tracked by git)
│   ├── chroma/                        # ChromaDB vector store (gitignored, derived)
│   └── previews/                      # JSON pipeline previews (gitignored)
│
├── docs/
│   ├── architecture/                  # Design records + enterprise diagram
│   └── guides/                        # setup.md · deployment.md
│
├── logs/                              # Rotating log files (gitignored)
│
├── .env.example                       # Environment variable template
├── .gitignore
├── pyproject.toml                     # Package metadata + dependency list
└── README.md                          # This file
```

---

## 7. Technology Stack

### Core language

| Technology | Version | Why / Where used |
|---|---|---|
| **Python** | ≥ 3.11 | Main application language. `StrEnum` (used for `DocumentCategory`) requires 3.11+. |

### LLM backends

| Technology | Version | Why / Where used |
|---|---|---|
| **Anthropic SDK** (`anthropic`) | ≥ 0.40 | Powers the Claude chat backend. Used exclusively in `llm/claude_client.py` to call the Messages API in streaming mode. `claude-sonnet-5` is the default model. |
| **OpenAI SDK** (`openai`) | ≥ 1.50 | Powers the optional ChatGPT backend. Used exclusively in `llm/openai_client.py`. Streaming is used here too so the UI behaviour is identical. |

Both clients are wrapped behind the same `stream_reply(system, messages) → Iterator[str]` interface, defined as a `Protocol` in `application/components.py`. Swapping backends or adding a new one never touches `ChatService` or the UI.

### Vector database and embeddings

| Technology | Version | Why / Where used |
|---|---|---|
| **ChromaDB** (`chromadb`) | ≥ 1.0 | Persistent local vector store. Configured with cosine-similarity space so scores are in [−1, 1] and a meaningful floor (0.30) can be applied. Used exclusively in `vector_database/vector_store.py`. |
| **Sentence Transformers** (`sentence-transformers`) | ≥ 3.0 | Loads the `BAAI/bge-small-en-v1.5` embedding model (~130 MB, 384 dimensions). Embeddings are unit-normalised so cosine similarity is computed correctly. Used in `vector_database/embeddings.py`. Model is loaded lazily on first use. |

### Document processing

| Technology | Version | Why / Where used |
|---|---|---|
| **PyMuPDF** (`pymupdf`) / `fitz` | ≥ 1.24 | Fast, reliable PDF text extraction. Used in `document_pipeline/pdf_loader.py`. Doc IDs are SHA-256 hashes of file bytes, making ingestion idempotent. |
| **LangChain Text Splitters** (`langchain-text-splitters`) | ≥ 0.3 | `RecursiveCharacterTextSplitter` breaks cleaned page text into overlapping chunks at paragraph → line → sentence → word → character boundaries. Used in `document_pipeline/chunker.py`. |

### Configuration and validation

| Technology | Version | Why / Where used |
|---|---|---|
| **Pydantic** (`pydantic`) | ≥ 2.7 | Data models (`Document`, `Chunk`, `SearchResult`, `Settings`, etc.). Validates every domain boundary so malformed data fails loudly at the stage boundary instead of corrupting the store silently. |
| **Pydantic Settings** (`pydantic-settings`) | ≥ 2.2 | Reads `HEALTHPLUS_*` environment variables (and a `.env` file) into a typed, validated `Settings` object. Used in `config/settings.py`. |
| **python-dotenv** (`python-dotenv`) | ≥ 1.0 | Reads the `.env` file into the environment before Pydantic Settings picks it up. |

### Web interface

| Technology | Version | Why / Where used |
|---|---|---|
| **Streamlit** (`streamlit`) | ≥ 1.36 | Single-file web framework for the chat portal. `st.write_stream` renders token-by-token streaming. `st.cache_resource` ensures the heavy singletons (embedding model, ChromaDB client, ChatService) are built once per process. Used in `presentation/app.py`. |

### Developer tooling

| Technology | Version | Why / Where used |
|---|---|---|
| **Rich** (`rich`) | ≥ 13.7 | Beautiful terminal output for CLI scripts (`ingest.py`, `search.py`, etc.) and the `RichHandler` used for colourised console logging in development. |
| **pytest** (`pytest`) | ≥ 8.0 | Test runner. `[dev]` optional dependency. Custom marker `quality` gates retrieval evals against the live index. |
| **httpx** | transitive | Used by both LLM clients to inject a custom CA bundle for TLS verification on macOS with corporate certificates. |
| **certifi** | transitive | Provides the baseline CA bundle that is merged with macOS system keychain certificates in `claude_client._get_ca_bundle()`. |

---

## 8. Environments

The `HEALTHPLUS_ENVIRONMENT` variable selects one of three named environments: `development`, `staging`, or `production`. The application reads this at startup but does not change its behaviour based on it today — the field exists so environment-specific branching (e.g. stricter logging in production, richer debug output in development) can be added without a schema change.

### Local development

```dotenv
HEALTHPLUS_ENVIRONMENT=development
HEALTHPLUS_DEBUG=false
HEALTHPLUS_LOG_LEVEL=DEBUG     # Set to DEBUG for verbose pipeline logging
```

- Run with `streamlit run src/healthplus/presentation/app.py`.
- ChromaDB persists in `data/chroma/` on disk; `logs/` holds rotating log files.
- The embedding model is cached by Sentence Transformers in `~/.cache/huggingface/` after the first download.

### Testing

```dotenv
HEALTHPLUS_ENVIRONMENT=development   # tests don't read this
```

Unit and integration tests are **fully hermetic** — they never touch `data/chroma/`, the real embedding model, or any API. The integration test suite (`test_rag_integration.py`) spins up an isolated ChromaDB instance in pytest's `tmp_path` and uses a deterministic bag-of-words fake embedder.

The retrieval-quality tests (`test_retrieval_quality.py`) are the one exception: they need the real `data/chroma/` index populated by a previous `ingest.py` run. They are marked `@pytest.mark.quality` and skipped automatically when the index is absent.

```bash
python -m pytest                  # skip quality tests (safe for any machine)
python -m pytest -m quality       # run quality tests (requires data/chroma)
```

### Staging / Production

The application is stateless between requests except for the in-process ChromaDB client and the in-memory conversation store. For a production deployment:

- Run behind a reverse proxy (Nginx / Caddy) with TLS termination.
- Serve with `streamlit run ... --server.port 8501 --server.headless true`.
- The `logs/` directory should be mounted to persistent storage or shipped to a log aggregator.
- See [`docs/guides/deployment.md`](docs/guides/deployment.md) for a full Docker + reverse-proxy walkthrough.

> **Not yet implemented:** authentication/RBAC, persistent conversation storage (Redis/Postgres), PHI/PII scrubbing, rate limiting, and observability hooks. These are listed in the [hardening roadmap](docs/architecture/day-05-rag-chat-architecture.md#5-production-hardening-roadmap).

---

## 9. Testing and Debugging

### Running the test suite

```bash
# Full fast suite (~seconds, no network, no real model)
python -m pytest

# With verbose output and short tracebacks
python -m pytest -v --tb=short

# Run one file
python -m pytest tests/test_chat_service.py -v

# Run quality (live-index) tests — needs data/chroma populated
python -m pytest -m quality -v
```

### What each test file covers

| File | What it tests |
|---|---|
| `test_models.py` | `DocumentCategory` taxonomy and filename resolution |
| `test_text_cleaner.py` | Unicode normalisation, soft-hyphen removal, line-break repair |
| `test_chunker.py` | Page provenance, deterministic IDs, max-size constraint |
| `test_batch_ingestion.py` | One failed PDF does not stop others |
| `test_query_processor.py` | Whitespace collapse, empty/over-length rejection |
| `test_retriever.py` | Similarity-floor filtering, top-k passthrough, category filter |
| `test_context_manager.py` | Block numbering, deduplication, budget truncation |
| `test_prompt_builder.py` | System-prompt guardrails, empty-context marker, message assembly |
| `test_conversation_memory.py` | Turn windowing, conversation isolation, copy semantics |
| `test_claude_client.py` | Fail-fast on missing API key; injected-client bypass |
| `test_chat_service.py` | Full orchestration with fakes for all six collaborators |
| `test_rag_integration.py` | End-to-end over real ChromaDB (isolated `tmp_path`), fake embedder |
| `test_retrieval_quality.py` | Golden-set hit-rate ≥ 70 % against the live index |

### Debugging tips

**Increase log verbosity**

Set `HEALTHPLUS_LOG_LEVEL=DEBUG` in `.env` and restart. Every pipeline stage logs its inputs and outputs, so you can trace exactly what the retriever, context manager, and prompt builder receive.

**Inspect the vector store from the CLI**

```bash
# See what the retriever actually retrieves for any query
python scripts/search.py "your question here" --top-k 5

# Check retrieval quality against all 10 golden queries
python scripts/eval_retrieval.py
```

**Visualise the document pipeline**

```bash
# See raw text, cleaned text, chunks, and embeddings for any PDF page — no DB writes
python scripts/preview_pipeline.py data/knowledge_base/pdfs/Pricing.pdf --page 1

# Save a full JSON record for offline inspection
python scripts/preview_pipeline.py data/knowledge_base/pdfs/Pricing.pdf \
    --output data/previews/Pricing.json
```

**Verify configuration**

```bash
python scripts/verify_setup.py
```

This prints every active setting (with the API key masked) so you can confirm `.env` was loaded correctly.

**Log file location**

Logs are written to `logs/healthplus.log` (rotating, max 5 MB, 3 backups). Check this file for any error that may have been suppressed in the browser UI.

---

## 10. Troubleshooting

### "ANTHROPIC_API_KEY is not set" in the chat UI

The API key is missing or `.env` was not loaded. Steps to fix:

1. Make sure `.env` exists in the project root: `ls -la .env`
2. Confirm it contains `ANTHROPIC_API_KEY=sk-ant-...` with a real key.
3. Restart the Streamlit process (the key is read at startup via `@st.cache_resource`).

### SSL / certificate errors during model download

On corporate networks with TLS inspection the first Hugging Face model download may raise an `SSLError`. The `ClaudeClient` already merges macOS system keychain certificates with the certifi bundle to handle this. For the embedding model download, set the environment variable:

```bash
export REQUESTS_CA_BUNDLE=/path/to/your/corporate-ca.crt
# then retry:
python scripts/ingest.py data/knowledge_base/pdfs/*.pdf
```

See [`docs/guides/setup.md`](docs/guides/setup.md) for the full corporate-network guide.

### "No documents ingested yet" — knowledge base is empty

The chat UI works but every query returns no sources. Fix by running the ingest script:

```bash
python scripts/ingest.py data/knowledge_base/pdfs/*.pdf
```

If the ingest fails, check the error output — the most common causes are a missing PDF file or a filename whose stem does not match one of the eight valid categories (`sops`, `doctors`, `test_catalog`, `pricing`, `faqs`, `health_packages`, `reports`, `policies`).

### Answers have no citations / low-quality answers

The retrieval similarity floor is `0.30` by default. If you are seeing the "no relevant documents" message for queries that should match, try:

```bash
python scripts/search.py "your query" --top-k 5
```

If matching chunks appear with scores close to 0.30, lower the floor in `.env`:

```dotenv
HEALTHPLUS_RETRIEVAL_MIN_SCORE=0.20
```

### Re-ingesting does not update the knowledge base

Ingestion is idempotent (same file → same IDs → upsert, not insert). If you want a clean rebuild:

```bash
rm -rf data/chroma                                        # delete the index
python scripts/ingest.py data/knowledge_base/pdfs/*.pdf   # rebuild from scratch
```

### Streamlit reruns and the page flickers

This is normal Streamlit behaviour — the script re-executes on every user interaction. The heavy singletons (ChromaDB client, embedding model, ChatService) are cached with `@st.cache_resource` so they are not rebuilt on rerun.

### `ModuleNotFoundError: No module named 'healthplus'`

The package is not installed in the active Python environment. Run:

```bash
pip install -e ".[dev]"
```

Make sure the virtual environment is activated (`source .venv/bin/activate`).

### Tests fail with `ImportError` or fixture errors

Same root cause as above — the package is not installed in the test environment:

```bash
pip install -e ".[dev]"
python -m pytest
```

---

## 11. Assumptions, Limitations, and Dependencies

### Assumptions

- **Single process** — ChromaDB uses a local persistent client. Running multiple Streamlit workers simultaneously would cause file-locking conflicts. This is a v1 architecture decision, not a fundamental constraint.
- **PDF-only corpus** — the document pipeline only handles PDFs. Other formats (Word, HTML, plain text) would require a new loader.
- **Filename-based category resolution** — the file stem must exactly match one of the eight category values (`pricing`, `doctors`, etc.) unless the category is specified explicitly via the upload UI.
- **English-only** — the `BAAI/bge-small-en-v1.5` embedding model is English-only. Non-English documents will embed but with degraded quality.

### Limitations

- **In-memory conversation history** — each browser tab gets its own conversation session, but history is stored in-process. All conversations are lost on app restart.
- **No authentication** — any user who can reach the Streamlit URL has full access, including PDF upload. Role-based access control is in the hardening roadmap.
- **Context window budget** — each prompt is capped at 8,000 characters of retrieved context. Very large documents may surface only a fraction of their content in a single turn.
- **No streaming abort** — if the user navigates away mid-stream, the memory is not written for that turn (intentional), but the Claude API call continues until it finishes or times out.
- **No PHI/PII scrubbing** — the corpus is fictional, so this is not a concern for the demo. A real deployment would need a PII scanner before documents enter the knowledge base.

### Key dependencies and their roles

| Dependency | Role |
|---|---|
| `chromadb` | The only vector store supported. Removing it would require replacing `vector_database/vector_store.py`. |
| `sentence-transformers` | The only embedding backend. Swapping to OpenAI embeddings or another provider only requires changing `vector_database/embeddings.py`. |
| `anthropic` | Claude LLM backend. Isolated to `llm/claude_client.py`. |
| `openai` | ChatGPT LLM backend. Isolated to `llm/openai_client.py`. |
| `pydantic` v2 | Domain model validation. A downgrade to v1 would require rewriting all models. |
| `streamlit` ≥ 1.36 | `st.write_stream` (used for token-by-token rendering) requires 1.36+. |

---

## Disclaimer

All documents in `data/knowledge_base/` describe a **fictional hospital** and exist purely for RAG development and demonstration. Nothing this system produces is medical advice. The assistant is designed to refuse diagnosis and treatment questions and redirect them to qualified doctors at HealthPlus.
