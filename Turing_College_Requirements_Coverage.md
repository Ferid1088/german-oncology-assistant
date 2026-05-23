# Sprint 2 — Requirements Coverage Report

**Project:** Oncology RAG Assistant (German S3 Leitlinien)  
**Date:** 2026-05-23

---

## Project Overview

This project is a LangGraph-based retrieval-augmented generation (RAG) system for querying German S3 oncology clinical guidelines (breast, colorectal, lung, prostate cancer). Users can ask clinical questions in German; the system retrieves relevant guideline passages, reasons over them with tool-calling agents, and generates grounded professional and plain-language answers with citations.

---
## Full Checklist

### Core Requirements

- ✅ RAG — knowledge base with domain-relevant documents
- ✅ RAG — document retrieval with embeddings
- ✅ RAG — chunking strategies
- ✅ RAG — similarity search
- ✅ Tool calling — at least 3 different tools (`search_guidelines`, `lookup_empfehlung`, `compare_guidelines`, `drug_class_lookup`, `calculate_bmi`, `pubmed_search`)
- ✅ Domain specialisation — focused knowledge base (4 oncology guidelines)
- ✅ Domain specialisation — domain-specific prompts and responses
- ✅ Domain specialisation — relevant security measures (input/output guardrails, PII redaction)
- ✅ Technical — LangChain/LangGraph with OpenRouter (OpenAI-compatible SDK)
- ✅ Technical — proper error handling
- ✅ Technical — logging and monitoring
- ✅ Technical — user input validation
- ✅ Technical — rate limiting and API key management
- ✅ UI — Streamlit interface
- ✅ UI — relevant context and sources shown
- ✅ UI — tool call results displayed
- ✅ UI — progress indicators for long operations

### Optional Tasks

**Easy**
- ✅ Conversation history and export functionality
- ✅ Visualisation of RAG process
- ✅ Source citations in responses
- ✅ Interactive help feature

**Medium**
- ✅ Calculate and display token usage and costs
- ✅ Visualisation of tool call results
- ✅ Conversation export in various formats (JSON, CSV, PDF)
- [~] Multi-model support (configurable via env vars, no UI toggle)
- [~] Advanced caching strategies (BM25 + auth cache; no query-result cache)
- [~] User authentication and personalisation (API key auth; no login/JWT)
- [ ] Real-time data updates to knowledge base
- [ ] Connect to MCP server

**Hard**
- ✅ Implement advanced analytics dashboard
- ✅ Implement RAG evaluation using RAGAs
- [~] A/B testing for different RAG strategies (offline evaluation only)
- [ ] Deploy to cloud with scaling
- [ ] Advanced indexing (RAPTOR, ColBERT)
- [ ] Fine-tune the model
- [ ] Multi-language support
- [ ] Implement tools as MCP servers

> Legend: ✅ `[x]` = fully implemented · ⚠️ `[~]` = partially implemented · ❌ `[ ]` = not implemented
---
## Core Requirements

### Summary

| Requirement | Status | Key Files |
|---|---|---|
| RAG Implementation | ✅ Fully covered | `src/retrieval/search.py`, `src/indexer/` |
| Tool Calling | ✅ Fully covered — 6 tools | `src/tools/`, `src/graph/nodes/agent.py` |
| Domain Specialisation | ✅ Fully covered | `src/prompts/`, `src/graph/nodes/guardrail_*.py` |
| Technical Implementation | ✅ Fully covered | `src/api/`, `src/graph/graph.py` |
| User Interface | ✅ Fully covered | `src/ui/` |

---

### 1. RAG Implementation

**Status: ✅ Fully Covered**

#### Knowledge Base
Four German S3 oncology PDFs are indexed into a Milvus Lite vector database:
- `mammakarzinom_v4.4.pdf` (breast cancer)
- `kolorektales_v3.0.pdf` (colorectal cancer)
- `lungenkarzinom_v4.0.pdf` (lung cancer)
- `prostatakarzinom_v8.0.pdf` (prostate cancer)

Indexing pipeline: `src/indexer/pipeline.py` → `index_pdf()`  
Storage: `src/indexer/store.py` → `MilvusStore` class (HNSW index, COSINE metric, 3072-dim vectors)

#### Embeddings
- Model: `text-embedding-3-large` via OpenRouter
- Implementation: `src/indexer/embedder.py` → `embed_texts(texts)`, batch size 64
- At index time, each chunk's embedding input is enriched with a generated contextual header and hypothetical questions (`src/indexer/enricher.py`)

#### Chunking Strategy
- Implementation: `src/indexer/chunker.py`
- Target chunk size: 550 tokens, overlap: 70 tokens (~12% sliding window)
- Hierarchical parent/leaf structure: every leaf chunk has a `parent_chunk_id`
- Chunk types classified: `recommendation`, `section`, `evidence`, `rationale`, `table`
- Clinical metadata attached: recommendation grade (A/B/0), evidence level, recommendation ID (e.g. "4.2.1"), section hierarchy

#### Similarity Search
- Implementation: `src/retrieval/search.py` → `hybrid_search()`
- **Dual dense search:** two Milvus ANN searches — one over all chunk types, one restricted to `chunk_type=recommendation` (prevents long prose from burying terse recommendations)
- **Sparse BM25 search:** pre-built index at startup, persisted as `bm25_index.pkl` (`src/retrieval/bm25.py`)
- **RRF fusion:** Reciprocal Rank Fusion (`k=60`, 2 passes) merges all three ranked lists (`src/retrieval/search.py`)
- **CrossEncoder reranker:** `BAAI/bge-reranker-v2-m3`, candidate pool ≤ 12, returns top 5 (`src/retrieval/reranker.py`)
- **Parent context expansion:** leaf chunks fetch parent text from Milvus and merge page ranges (`src/retrieval/expander.py`)

---

### 2. Tool Calling

**Status: ✅ Fully Covered — 6 domain tools implemented**

The agent node (`src/graph/nodes/agent.py` → `run_agent()`) runs a 2-iteration GPT-4o loop. Iteration 1 always forces `search_guidelines`. Iteration 2 is free-choice (`tool_choice="auto"`). All tools are defined as OpenAI-compatible function specs in `TOOLS_SPEC`.

| Tool | File | What It Does |
|---|---|---|
| `search_guidelines` | `src/tools/search_guidelines.py` | Hybrid RAG retrieval over all 4 guidelines. Main retrieval tool. |
| `lookup_empfehlung` | `src/tools/lookup_empfehlung.py` | Fetches a specific recommendation by ID (e.g. "4.2.1") from a named guideline. |
| `compare_guidelines` | `src/tools/compare_guidelines.py` | Side-by-side comparison of two guidelines on a topic — calls `search_guidelines` twice and returns paired results. |
| `drug_class_lookup` | `src/tools/drug_class_lookup.py` | Cross-guideline search for a drug/substance — searches all 4 guidelines, groups results by guideline and grade. |
| `calculate_bmi` | `src/tools/calculate_bmi.py` | Computes BMI from weight (kg) and height (cm), returns value + WHO category in German. |
| `pubmed_search` | `src/tools/pubmed_search.py` | External literature search via NCBI E-utilities API. Returns PMID, title, authors, date, URL. |

An additional tool `web_search_snippets` (`src/tools/web_search.py`) is used by the external search node (Google Custom Search with DuckDuckGo fallback), not by the agent directly.

Tool dispatch with RBAC: `_dispatch_tool_with_state(state, name, args)` checks `is_tool_allowed()` and `is_source_allowed()` before executing any tool (`src/graph/permissions.py`).

---

### 3. Domain Specialisation

**Status: ✅ Fully Covered**

#### Domain Focus
German oncology — specifically the four S3-Leitlinien from Deutsche Krebsgesellschaft. All prompts, outputs, error messages, and guardrail responses are in German.

#### Domain-Specific Knowledge Base
- Recommendation grades (A/B/0) and evidence levels extracted and stored as Milvus metadata (`src/indexer/metadata.py`)
- LLM enrichment at index time: contextual headers, hypothetical clinician questions, semantic metadata (diseases, drugs, procedures, patient subgroups) — `src/indexer/enricher.py`
- Bibliography parsing links inline references to source citations — `src/indexer/reference.py`

#### Domain-Specific Prompts
- `src/prompts/agent.py` — `AGENT_SYSTEM`: German-language system prompt. Explicitly forbids using training knowledge, mandates tool-first retrieval, defines tool priority order.
- `src/prompts/rewriter.py` — `build_ambiguity_prompt_messages()`: reformulates vague clinical queries, extracts guideline/grade filters, detects when clarification is needed.
- `src/prompts/turn_router.py` — intent detection: classifies turn as clarify/simplify/expand/refine/new_query, decides memory-vs-retrieve routing.
- `src/prompts/answer.py` — dual-mode generation: professional answer for clinicians + plain language summary for patients, with DISCLAIMER appended.

#### Security Measures
**Input Guardrail** (`src/graph/nodes/guardrail_input.py`):
- Prompt injection detection: 9 hardcoded patterns ("ignore previous instructions", "jailbreak", "bypass", etc.)
- Off-topic blocking: keyword list (weather, sports, recipes, politics, music, etc.)
- PII redaction: 4 regex patterns replace dates, full names, phone numbers, German postal codes with `[REDACTED]` before any LLM sees the query

**Output Guardrail** (`src/graph/nodes/guardrail_output.py`):
- Faithfulness check: blocks any answer that has no retrieved chunks (no hallucination without grounding)
- Dosage safety: blocks specific dosing information if dosage figures aren't directly present in retrieved chunks
- Patient-specific warning: adds safety disclaimer when query mentions a specific patient case

**Confidence & Escalation** (`src/graph/nodes/confidence.py`):
- Threshold: mean reranker score of top-3 chunks must be ≥ 0.5, and at least 2 chunks must exist
- Low confidence triggers 4-query fallback escalation (`src/graph/graph.py` → `_multi_query_escalation()`)

---

### 4. Technical Implementation

**Status: ✅ Fully Covered**

#### LangGraph + OpenRouter
- Graph: `src/graph/graph.py` → `build_graph()` — 12-node `StateGraph` with 4 conditional routing decisions
- State: `src/graph/state.py` → `RAGState` TypedDict (~35 fields)
- LLM access: all models accessed via OpenRouter API using the OpenAI-compatible SDK
  - Generation: `GENERATION_MODEL` env var (default: `openai/gpt-4o`)
  - Cheap: `CHEAP_MODEL` env var (default: `google/gemini-2.5-flash`)
  - Embedding: `EMBEDDING_MODEL` env var (default: `openai/text-embedding-3-large`)

#### Error Handling
- Milvus queries: retry logic (5 attempts, 3s exponential backoff) in `src/retrieval/search.py`
- Reranker unavailable: graceful fallback to retrieval scores in `src/retrieval/reranker.py`
- LLM parse failures: all nodes have `except Exception` blocks falling back to safe defaults (e.g. `_default_result()` in `src/graph/nodes/rewriter.py`)
- API exceptions: mapped to HTTP error codes in `src/api/observability.py`

#### Logging & Monitoring
- `src/api/observability.py`: JSON-structured logging via `log_event()`, trace ID per request (`X-Trace-Id` header), middleware logs request start/completion/exception with duration in ms
- `src/telemetry.py`: per-call token usage tracking, cost calculation (USD per 1M tokens), call-level details aggregated into `token_usage` state field
- RAG trace: every node appends a step to `rag_trace` (name, status, summary, details, duration_ms) — full pipeline trace returned in every API response

#### Input Validation
- `src/api/routes/chat.py`: Pydantic `ChatRequest` model
  - `query`: 3–1500 characters
  - `session_id`: 1–120 characters
  - `guideline_id`: whitelist (`mamma`, `krk`, `lunge`, `prosta`, `""`)
  - `grade`: whitelist (`A`, `B`, `0`, `""`)
  - Validation errors return HTTP 422 with detailed field-level messages

#### Rate Limiting
- `src/api/rate_limit.py`: sliding window algorithm with `deque`, thread-safe (`Lock`)
- Config: `src/api/rate_limit.config.json`
  - Chat route: 20 requests / 60 seconds
  - General routes: 60 requests / 60 seconds
- Bucket key: `(route_group, api_key, ip)` — per-user, per-route isolation
- Returns `retry_after_seconds` on limit exceeded

#### API Key Management
- `src/api/auth.py`: `verify_api_key()` checks `X-API-Key` header or `api_key` query param
- Supports multiple keys (`API_KEYS` env var, comma-separated) or single key (`API_KEY` env var)
- LRU-cached verification with explicit `reset_api_key_cache()` function
- Dev mode default: `"dev-secret-key"` — production requires explicit configuration

---

### 5. User Interface

**Status: ✅ Fully Covered (Streamlit)**

#### Framework
- Streamlit app entry point: `src/ui/app.py`
- Main chat component: `src/ui/components/chat_page.py` → `render_chat_page()` (~500 lines)
- Multi-panel layout: sidebar (filters, conversation list) + main chat + optional right panel (insights)

#### Context & Sources Display
- `src/ui/components/source_cards.py` → `render_source_cards()`: displays retrieved chunks with chunk ID, guideline, section, page numbers, recommendation grade/ID, evidence level, citation label `[1]`, `[2]`, etc.
- `src/citations.py` → `format_citation()`: formats full citation strings, e.g. `MAMMA § 4.2.1 (S. 42–45)`
- `src/ui/components/inline_citations.py` → `annotate_citations()`: embeds `[N]` references inline in answer text

#### Tool Call Results Display
- `src/ui/components/source_cards.py` → `render_tool_calls()`: shows each tool call with tool name, input args, result count, and a preview of the first results
- `src/telemetry.py` → `summarize_tool_result()`: converts raw tool output into a user-friendly summary string

#### Progress Indicators
- `src/ui/components/insights_panels.py` → `render_rag_process_panel()`: step-by-step RAG pipeline trace display — each step shows name, status badge (ok/empty/blocked/error), summary text, duration in ms, and collapsible details
- `src/ui/components/insights_panels.py` → `render_token_usage_panel()`: input/output token counts and total cost in USD per response

---

## Optional Tasks Covered

### Easy

All four Easy optional tasks are implemented.

#### 1. Conversation History and Export Functionality ✅

**How:** Conversations are persisted to a SQLite database.
- `src/api/conversation_store.py`: `ConversationStore` class — tables for `conversations` and `messages`, methods: `save_conversation()`, `append_turn()`, `load_session_memory()`, `list_conversations_detailed()`
- Storage path: `CONVERSATION_DB_PATH` env var (default: `data/app_state.db`)
- UI: sidebar shows conversation list, users can load and switch sessions
- Export: `src/api/export_utils.py` — POST `/conversations/{session_id}/export` supports **JSON**, **CSV**, and **PDF** formats

#### 2. Visualisation of RAG Process ✅

**How:** Every API response includes a full `rag_trace` — a list of pipeline steps with name, status, summary, duration, and details. The UI renders this as an interactive step-by-step panel.
- `src/telemetry.py` → `append_rag_step()`: appends a step to the trace at each node
- `src/ui/components/insights_panels.py` → `render_rag_process_panel()`: renders the trace as expandable step cards with status badges and durations

#### 3. Source Citations in Responses ✅

**How:** Every answer includes numbered citations `[1]`, `[2]` inline, linked to source cards.
- `src/citations.py`: formats citation strings with guideline name, section ID, and page range (e.g. `MAMMA § 4.2.1 (S. 42–45)`)
- `src/ui/components/inline_citations.py` → `annotate_citations()`: injects citation markers into answer text
- `src/ui/components/source_cards.py`: renders full source cards beneath the answer for each cited chunk

#### 4. Interactive Help Feature ✅

**How:** The Streamlit sidebar includes:
- Descriptions of all 4 supported guidelines (what they cover, guideline ID)
- Example queries users can run
- Filter controls explained (grade filter, guideline selector)
- The agent system prompt (`src/prompts/agent.py`) includes explicit tool descriptions that help the LLM guide the user when a query is unclear

---

### Medium

#### 5. Calculate and Display Token Usage and Costs ✅

**How:** Every LLM call records token counts and computes cost against a pricing table.
- `src/telemetry.py`: `usage_from_response()` extracts prompt/completion tokens; pricing table maps model IDs to cost per 1M tokens; `merge_token_usage()` aggregates across calls
- `token_usage` is stored in `RAGState` and returned in every API response
- `src/ui/components/insights_panels.py` → `render_token_usage_panel()`: displays input tokens, output tokens, and total cost (USD) per response
- `src/ui/components/analytics_dashboard.py`: shows cumulative token usage and cost aggregated across all conversations

#### 6. Visualisation of Tool Call Results ✅

**How:** Each tool call is logged with input arguments, a summary, result preview, and status.
- `src/graph/nodes/agent.py`: every tool call appends to `tool_calls_log` in state
- `src/telemetry.py` → `summarize_tool_result()`: maps raw tool output to human-readable summary string and preview
- `src/ui/components/source_cards.py` → `render_tool_calls()`: renders each tool call as a card with tool name, args, result count, and first-result preview

#### 7. Conversation Export in Various Formats ✅

**How:** Conversations can be exported in three formats via a single API endpoint.
- `src/api/export_utils.py`: implements `export_conversation()` for:
  - **JSON** — full structured data including messages, metadata, citations, rag_trace
  - **CSV** — tabular rows per message with computed columns (token counts, citation count)
  - **PDF** — formatted text document with conversation transcript
- Route: `POST /conversations/{session_id}/export?format=json|csv|pdf`

#### 8. Multi-Model Support ⚠️ Partial

**How:** Model selection is fully configurable via environment variables:
- `GENERATION_MODEL` (default: `openai/gpt-4o`) — used by agent and answer synthesis nodes
- `CHEAP_MODEL` (default: `google/gemini-2.5-flash`) — used by rewriter, turn router, extraction
- `EMBEDDING_MODEL` (default: `openai/text-embedding-3-large`) — used at index time and query time
- `RERANKER_MODEL` (default: `BAAI/bge-reranker-v2-m3`) — CrossEncoder reranker

**What's missing:** No UI control to switch models at runtime. Switching requires restarting with a different env var. No automatic model selection or fallback chain.

#### 9. Advanced Caching Strategies ⚠️ Partial

**How:** Three caching mechanisms exist:
- `src/retrieval/bm25.py`: BM25 index loaded once as a module-level singleton at startup (not rebuilt per request)
- `src/api/auth.py`: `@lru_cache(maxsize=1)` on the API key set loader
- `src/api/rate_limit.py`: rate limit config loaded once from JSON, `reload_config()` available

**What's missing:** No query-level result caching (same query asked twice hits the database again), no embedding vector caching beyond the Milvus index itself, no distributed cache (Redis, etc.).

#### 10. User Authentication and Personalisation ⚠️ Partial

**How:** API key authentication is implemented:
- `src/api/auth.py`: `verify_api_key()` checks `X-API-Key` header or `api_key` query param against environment-configured keys
- Multiple API keys supported (`API_KEYS` env var, comma-separated)
- Session isolation via `session_id` field in each request

**What's missing:** No user registration, login, or logout. No JWT tokens. No OAuth/SSO. No per-user personalisation (e.g. preferred guideline, saved filters). RBAC exists in `src/graph/permissions.py` but all roles currently have the same tool access.

---

### Hard

#### 11. Implement an Advanced Analytics Dashboard ✅

**How:** A dedicated analytics dashboard aggregates usage data across all conversations.
- `src/api/analytics_service.py`: `AnalyticsService` class — queries conversation store, aggregates metrics
- `src/api/routes/analytics.py`: `GET /analytics/overview` endpoint
- `src/ui/components/analytics_dashboard.py` → `render_analytics_dashboard()`: Streamlit component showing:
  - Total conversations, messages, tokens, cost
  - Conversations and messages per day (time-series)
  - Tool call frequency (which tools were called most)
  - Guideline distribution (which guidelines were queried)
  - Top questions from session summaries

#### 12. Implement an Evaluation of Your RAG System (RAGAs) ✅

**How:** A full evaluation framework is implemented in the `evaluations/` directory.
- `evaluations/scripts/run_eval.py`: runs a test dataset against the live API, collects answers and retrieved chunks
- `evaluations/metrics/ragas_metrics.py`: integrates the `ragas` library — computes `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`, `answer_correctness`
- `evaluations/metrics/retrieval.py`: custom retrieval metrics — `chunk_recall`, `chunk_precision`, `top_gold_chunk_hit`
- `evaluations/metrics/behavioral.py`: behavioral metrics — `answer_length`, `tool_call_count`, `external_search_used`
- `evaluations/metrics/similarity.py`: semantic similarity metrics — `answer_similarity`, `coverage`
- `evaluations/ui/app.py`: Streamlit evaluation dashboard for browsing per-question results and summary stats
- Results stored as: `summary.json`, `item_results.json`, `ragas_records.json`, `metadata.json`

#### 13. A/B Testing for Different RAG Strategies ⚠️ Partial

**How:** An A/B evaluation script exists:
- `evaluations/scripts/run_ab_eval.py`: runs the same evaluation dataset against two different model configurations and compares metrics side-by-side

**What's missing:** No in-production traffic split. No automated winner selection. No statistical significance testing. A/B comparison is only offline (evaluation dataset), not live with real users.

---

## Optional Tasks Not Implemented

| Task | Reason |
|---|---|
| Real-time data updates to knowledge base | Manual re-indexing only via `scripts/run_indexer.py` — no scheduler or crawler |
| Connect to MCP server (Medium) | No MCP protocol integration |
| Implement tools as MCP servers (Hard) | Tools are plain Python functions, not MCP servers |
| Deploy to cloud with scaling (Hard) | FastAPI is cloud-ready but no Dockerfile, no cloud config, no deployment docs |
| Advanced indexing — RAPTOR / ColBERT (Hard) | HNSW + hybrid BM25 only; no recursive abstractive indexing or late-interaction retrieval |
| Fine-tune the model (Hard) | Uses pre-trained GPT-4o and Gemini; no fine-tuning pipeline |
| Multi-language support (Hard) | German only — prompts, guardrails, chunking, and outputs are all German-specific |

---

