# Project Structure Overview

This document explains the role of each main directory, important subdirectories, and key scripts/files in this repository.

## High-level assessment

The project already had a solid modular base. It separates:

1. **UI** (`src/ui`)
2. **API** (`src/api`)
3. **RAG orchestration** (`src/graph`)
4. **Retrieval** (`src/retrieval`)
5. **Indexing / ingestion** (`src/indexer`)
6. **Prompts and tools** (`src/prompts`, `src/tools`)
7. **Tests and documentation** (`tests`, `docs`)

After cleanup, the structure is now more advanced and consistent because:

- obsolete feedback backend code was removed
- the typo-based prompt package was normalized into `src/prompts/ambiguity`
- tests now align better with the real runtime structure

---

## Repository structure

```text
project/
├── config/
├── data/
├── docs/
├── scripts/
├── src/
│   ├── api/
│   ├── graph/
│   ├── indexer/
│   ├── prompts/
│   │   └── ambiguity/
│   ├── retrieval/
│   ├── tools/
│   └── ui/
├── tests/
│   ├── api/
│   ├── eval/
│   ├── graph/
│   ├── indexer/
│   ├── prompts/
│   ├── retrieval/
│   └── tools/
├── pyproject.toml
└── uv.lock
```

---

## Top-level directories and files

### `config/`
Static project configuration and source metadata.

- `download_guidelines.py` — helper for acquiring source guideline documents
- `sources.json` — source catalog/configuration

### `data/`
Generated data and runtime artifacts.

- `data/eval/` — evaluation dataset outputs

### `docs/`
Documentation, schemas, and raw knowledge assets.

- `evaluation-dataset.schema.json` — schema for evaluation dataset validation
- `guideline-chunk-metadata.schema.json` — schema for chunk metadata validation
- `parsing-and-chunking-strategy.md` — ingestion/chunking design notes
- `project_concept.md` — high-level project concept
- `testing-and-evaluation-strategy.md` — testing strategy
- `project-structure-overview.md` — this overview file

#### `docs/knowledge_base/`
Source PDF guideline documents used for indexing.

#### `docs/superpowers/`
Internal plans and specifications.

### `scripts/`
Operational entry points for local development and maintenance.

- `run_app.py` — launches FastAPI backend + Streamlit frontend
- `run_indexer.py` — runs ingestion/indexing into Milvus + BM25 rebuild
- `generate_eval_dataset.py` — generates evaluation datasets

### `pyproject.toml`
Dependency/tool/project configuration.

### `uv.lock`
Pinned dependency lockfile.

---

## `src/` — application source code

### `src/api/`
FastAPI backend, persistence, analytics, observability, and exports.

- `main.py` — FastAPI app creation and router wiring
- `auth.py` — API key authentication
- `conversation_store.py` — SQLite-backed conversation persistence
- `analytics_service.py` — analytics aggregation/service logic
- `export_utils.py` — JSON/CSV/PDF export helpers
- `observability.py` — logging, trace IDs, error details
- `rate_limit.py` — in-memory rate limiting
- `rate_limit.config.json` — rate-limit route configuration

#### `src/api/routes/`
Route entry modules.

- `chat.py` — primary chat endpoint
- `conversations.py` — conversation listing, creation, deletion, export
- `analytics.py` — analytics endpoint(s)

### `src/graph/`
Workflow orchestration for the RAG pipeline.

- `state.py` — typed shared graph state
- `graph.py` — graph construction and routing
- `messages.py` — message normalization helpers
- `permissions.py` — source/tool permission logic
- `checkpointing.py` — optional checkpoint integration

#### `src/graph/nodes/`
Single-purpose graph steps.

- `guardrail_input.py` — input safety and off-topic filtering
- `rewriter.py` — query rewrite and clarification logic
- `turn_router.py` — follow-up routing decisions
- `agent.py` — tool/retrieval execution
- `confidence.py` — confidence estimation and escalation routing
- `answer.py` — grounded answer creation and memory rewrite
- `guardrail_output.py` — output safety checks
- `external_search.py` — web-search enrichment

### `src/indexer/`
Offline indexing/ingestion pipeline.

- `pipeline.py` — top-level indexing workflow
- `parser.py` — document parsing
- `chunker.py` — text chunking
- `metadata.py` — metadata extraction/normalization
- `enricher.py` — enrichment logic
- `embedder.py` — embeddings generation
- `store.py` — Milvus integration
- `detector.py`, `reference.py` — supporting indexing utilities

### `src/prompts/`
Prompt code and prompt assets.

- `agent.py`, `answer.py`, `turn_router.py` — prompt definitions for runtime steps
- `rewriter.py` — public import surface for rewrite prompts

#### `src/prompts/ambiguity/`
Dedicated prompt bundle for ambiguity detection and query rewriting.

- `loader.py` — builds prompt messages from file assets
- `system.txt` — system prompt
- `instructions.txt` — rewrite/clarification instructions
- `output_schema.json` — required JSON output schema
- `few_shot_examples.json` — example inputs/outputs

### `src/retrieval/`
Search and ranking layer.

- `search.py` — retrieval entrypoint(s)
- `bm25.py` — sparse retrieval support
- `expander.py` — query expansion
- `reranker.py` — reranking logic
- `postprocess.py` — deduplication and result shaping

### `src/tools/`
Callable tools used by the assistant.

- `search_guidelines.py` — guideline search tool
- `web_search.py` — web search tool
- `pubmed_search.py` — literature search
- `compare_guidelines.py` — cross-guideline comparison helper
- `drug_class_lookup.py` — drug class helper
- `lookup_empfehlung.py` — recommendation lookup helper
- `calculate_bmi.py` — lightweight utility tool

### `src/ui/`
Streamlit frontend.

- `app.py` — UI entrypoint

#### `src/ui/components/`
Composable frontend components.

- `chat_page.py` — main chat workspace and layout
- `analytics_dashboard.py` — analytics side panel
- `insights_panels.py` — RAG/usage/tool/web details panels
- `source_cards.py` — Quellen/resources rendering
- `inline_citations.py` — inline citation helpers
- `filters.py` — sidebar filtering UI

### Shared utilities

- `src/citations.py` — citation handling helpers
- `src/telemetry.py` — token usage and trace utilities

---

## `tests/`

The test structure mirrors the application domains, which is a strong sign of a professional layout.

- `tests/api/` — API/auth/rate-limit/export/analytics tests
- `tests/eval/` — evaluation dataset generation tests
- `tests/graph/` — graph routing/node tests
- `tests/indexer/` — ingestion/indexing tests
- `tests/prompts/` — prompt loader tests
- `tests/retrieval/` — retrieval/reranking/expansion tests
- `tests/tools/` — tool tests

---

## Structural verdict

### What is strong now

- clear separation of concerns across major layers
- tests mirror runtime domains
- prompt assets are isolated and file-based
- scripts are separate from reusable source modules
- obsolete feature code has been removed

### Remaining notes

The structure is now **advanced and professional** for a project of this size.

It is not “perfect” in an absolute sense, but it is well-organized, understandable, and maintainable.

### Optional next upgrades

If you want to push it further later:

1. split `src/ui/components/chat_page.py` into smaller view modules
2. move request/response schemas into dedicated API schema modules
3. introduce a `services/` layer for shared business logic
4. add `docs/architecture/` with ADRs
5. separate runtime-generated local artifacts more explicitly from source-controlled data
