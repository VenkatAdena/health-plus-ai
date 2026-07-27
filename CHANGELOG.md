# Changelog

## [0.5.0] — 2026-07-17 (Day 5)

### Added
- RAG chat application layer: `ChatService` orchestrating one grounded chat turn — `DefaultQueryProcessor` (input validation/normalization), `KnowledgeBaseRetriever` (0.30 cosine-similarity floor), `ContextWindowManager` (deduped, numbered context blocks capped at 8,000 chars), `HealthcarePromptBuilder`, `InMemoryConversationMemory` (10-turn window, written only after a stream completes), and the `build_chat_service()` wiring factory
- `healthplus.llm` adapter package: `ClaudeClient` streaming `claude-sonnet-5` via the Anthropic SDK — fails fast with `ConfigurationError` when `ANTHROPIC_API_KEY` is missing; SDK/network failures wrapped in `LLMError`
- Healthcare prompt guardrails in one place (`prompt_builder.py`): answer only from numbered context blocks, cite sources inline as `[n]`, never give diagnosis/treatment advice, say "I don't know" instead of guessing
- Streamlit chat portal (`presentation/app.py`): streaming answers via `st.write_stream`, per-answer source-citations expander, PDF upload with explicit category selection, sidebar knowledge-base stats and category filter, conversation reset; degrades gracefully without an API key (chat disabled with guidance, upload/search/stats stay usable)
- Retrieval / RAG settings: `HEALTHPLUS_CLAUDE_MODEL`, `HEALTHPLUS_CLAUDE_MAX_TOKENS`, `HEALTHPLUS_RETRIEVAL_TOP_K`, `HEALTHPLUS_RETRIEVAL_MIN_SCORE`, `HEALTHPLUS_CONTEXT_MAX_CHARS`, `HEALTHPLUS_MEMORY_MAX_TURNS`
- Retrieval-quality evaluation: `scripts/eval_retrieval.py` golden query set with a 70% top-3 hit-rate floor (CI-gateable exit code) and `tests/test_retrieval_quality.py`
- Component and integration tests: query processor, retriever, context manager, prompt builder, conversation memory, Claude client, chat service, and an end-to-end RAG integration test with fakes (no network, no ChromaDB)
- Documentation: `docs/architecture/day-05-rag-chat-architecture.md` (components, decision record, hardening roadmap), `docs/guides/setup.md`, `docs/guides/deployment.md`, README overhaul

## Architecture correction — 2026-07-16

- Adopted the corrected six-layer enterprise diagram as the project source of truth.
- Split application orchestration, document processing, vector database, and knowledge-base contracts into explicit packages.
- Replaced the earlier corpus taxonomy with SOPs, Doctors, Test Catalog, Pricing, FAQs, Health Packages, Reports, and Policies.
- Switched the configured embedding model to `BAAI/bge-small-en-v1.5` as specified by the architecture.
- Added an in-repository copy of the authoritative diagram and a layer mapping record.

## Day 4 — Document processing pipeline

- Added deterministic PDF text normalization before chunking.
- Added resilient multi-document ingestion with structured failure reporting.
- Added focused cleaner and batch orchestration tests.
- Documented pipeline stages, contracts, and operational guarantees.

All notable changes to HealthPlus AI are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); versioning follows SemVer.

## [0.3.0] — 2026-07-15 (Day 3)

### Added
- `DocumentCategory` taxonomy (8 hospital-operations categories) with filename-convention resolution
- `category` field across the whole pipeline: models, loader, chunker, Chroma metadata, search results
- Category-filtered semantic search (`--category` flag on `scripts/search.py`)
- `category_counts()` knowledge-base statistics (first admin-dashboard metric)
- Real 77-page hospital corpus ingested: 91 chunks across 8 categories
- Composable Chroma `where`-clause builder (`$and` handling)
- Model/taxonomy unit tests (10 tests total)

### Changed
- `.gitignore`: source corpus (`data/knowledge_base/`) is now tracked; only derived data (`data/chroma/`) ignored

### Removed
- Clinical-guidelines sample data and `scripts/create_sample_data.py` (wrong product domain)

## [0.2.0] — 2026-07-14 (Day 2)

### Added
- Knowledge base vertical slice: PDF -> chunks -> embeddings -> ChromaDB -> semantic search
- Domain models (`healthplus.knowledge_base`): Document, PageContent, Chunk, SearchResult
- `PDFLoader` (PyMuPDF) with content-hash doc ids and empty-document detection
- `DocumentChunker` (recursive splitting, page-preserving, deterministic chunk ids)
- `EmbeddingService` (Sentence Transformers, cache-first model loading)
- `VectorStore` ChromaDB adapter (cosine similarity, metadata filtering, upsert semantics)
- `KnowledgeBaseService` facade with dependency injection
- Exception hierarchy (`healthplus.core.exceptions`)
- CLI scripts: `create_sample_data.py`, `ingest.py`, `search.py`
- Unit tests for the chunker (5 tests)

## [0.1.0] — 2026-07-14 (Day 1)

### Added
- Enterprise project scaffold with `src/` layout and installable `healthplus` package
- Configuration management via Pydantic Settings (`healthplus.config`): typed, validated, `.env`-driven, secrets as `SecretStr`
- Centralized logging (`healthplus.core.logging`): Rich console handler + rotating file handler
- `scripts/verify_setup.py` smoke test for the foundation
- `pyproject.toml`, `.env.example`, `.gitignore`, README, changelog
