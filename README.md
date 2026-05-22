# German Oncology RAG Assistant

A retrieval-augmented generation (RAG) system for querying German S3 oncology clinical guidelines. Ask clinical questions in German; the system retrieves grounded evidence from four major cancer guidelines and generates professional and plain-language answers with full citations.

---

## Overview

The system is built on a **LangGraph state machine** with 12 nodes. Every user query passes through input safety checks, query rewriting, intelligent routing, a GPT-4o tool-calling agent, confidence evaluation, answer generation, and output safety checks — before returning a cited, grounded response.

**Supported guidelines:**

| ID | Guideline | Version |
|---|---|---|
| `mamma` | Mammakarzinom (Breast Cancer) | S3 v4.4 |
| `krk` | Kolorektales Karzinom (Colorectal Cancer) | S3 v3.0 |
| `lunge` | Lungenkarzinom (Lung Cancer) | S3 v4.0 |
| `prosta` | Prostatakarzinom (Prostate Cancer) | S3 v8.0 |

---

## Architecture

```
User Query
    │
    ▼
[guardrail_input]  ── blocked ──► [blocked] ──► END
    │ pass
    ▼
[rewrite]  ── clarification ──► [clarification] ──► END
    │         duplicate ──────► [repeat_answer] ──► END
    │ ok
    ▼
[turn_router]
    │
    ▼
[agent]  ←── GPT-4o tool-calling loop (2 iterations)
    │          6 tools: search_guidelines, lookup_empfehlung,
    │          compare_guidelines, drug_class_lookup,
    │          calculate_bmi, pubmed_search
    ▼
[confidence]  ── low score ──► [escalate] ──┐
    │ ok                                     │
    ▼◄───────────────────────────────────────┘
[answer]  ── Gemini extract → GPT-4o synthesize
    │
    ▼
[guardrail_output]  ── output_blocked ──► END
    │ ok
    ▼
[external_search]  ──► END
```

---

## Features

### Core
- **Hybrid RAG search:** dense vector (Milvus HNSW) + BM25 sparse, fused via Reciprocal Rank Fusion
- **CrossEncoder reranking:** `BAAI/bge-reranker-v2-m3` re-scores top-12 candidates
- **6 domain tools:** guideline search, recommendation lookup, guideline comparison, drug cross-search, BMI calculation, PubMed search
- **Dual-stage answer generation:** Gemini 2.5 Flash extracts verbatim sentences → GPT-4o synthesizes into professional + plain-language answer
- **Full citation chain:** `[1]`, `[2]` inline references linked to source cards with page, section, and grade
- **Conversation memory:** repeated or follow-up queries reuse prior answers without re-querying the database

### Safety
- **Input guardrail:** blocks prompt injections and off-topic queries; redacts PII (dates, names, phone numbers, postal codes) before any LLM sees the query
- **Output guardrail:** blocks answers without retrieved evidence; restricts dosing details unless directly grounded in retrieved chunks; adds patient-specific disclaimers
- **Confidence escalation:** low reranker confidence triggers 4-query multi-strategy fallback retrieval

### Infrastructure
- **Rate limiting:** sliding window, per-user per-route (20 req/60s for chat)
- **API key management:** multi-key support via environment variable
- **Structured logging:** JSON-structured logs with trace IDs and request durations
- **Token tracking:** per-call usage and cost (USD) aggregated across the pipeline
- **Conversation persistence:** SQLite-backed conversation history with session isolation

### UI
- **Streamlit chat interface** with real-time SSE streaming
- **Source cards:** retrieved chunks with guideline, section, page range, grade, evidence level
- **RAG process visualisation:** step-by-step trace with status and duration for each pipeline node
- **Analytics dashboard:** token usage, cost, tool frequency, guideline distribution, time-series
- **Export:** download conversations as JSON, CSV, or PDF

---

## Prerequisites

- Python 3.12+
- An [OpenRouter](https://openrouter.ai) API key (routes to GPT-4o and Gemini 2.5 Flash)
- PDF files of the S3 guidelines (not included in the repository)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Ferid1088/german-oncology-assistant.git
cd german-oncology-assistant

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install the package and dependencies
pip install -e .

# Install dev dependencies (tests)
pip install -e ".[dev]"
```

---

## Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

`.env` reference:

```env
# Required — get a key at https://openrouter.ai
OPENROUTER_API_KEY=your_key_here

# LLM models (change only if you have confirmed the model ID exists on OpenRouter)
GENERATION_MODEL=openai/gpt-4o
CHEAP_MODEL=google/gemini-2.5-flash
EMBEDDING_MODEL=openai/text-embedding-3-large

# Milvus — leave empty to use local milvus-lite (./milvus.db, no server required)
MILVUS_URI=
MILVUS_COLLECTION=oncology_guidelines

# API authentication — change in production
API_KEY=dev-secret-key
# Optionally add multiple comma-separated keys:
# API_KEYS=key1,key2,key3

# Conversation database (SQLite)
CONVERSATION_DB_PATH=data/app_state.db

# Optional: Google Custom Search for external web results
# GOOGLE_SEARCH_API_KEY=
# GOOGLE_SEARCH_ENGINE_ID=

# Logging level: DEBUG | INFO | WARNING | ERROR
LOG_LEVEL=INFO

# Optional: PostgreSQL for LangGraph checkpointing (replaces in-memory)
# DATABASE_URL=postgresql://postgres:postgres@localhost:5432/oncology_rag
```

---

## Indexing the Guidelines

Before running the app, you must index the PDF guidelines into the Milvus vector database. Place your PDF files in the `data/` directory and run:

```bash
# Index all 4 configured guidelines
python scripts/run_indexer.py

# Index a specific guideline
python scripts/run_indexer.py --pdf mammakarzinom_v4.4.pdf

# Dry run (parse and chunk only, no database writes)
python scripts/run_indexer.py --dry-run

# Skip LLM enrichment (faster, lower quality)
python scripts/run_indexer.py --no-enrich
```

Indexing with enrichment enabled (default) calls Gemini 2.5 Flash per chunk to generate:
- A contextual header summarising the chunk in clinical context
- 2–3 hypothetical questions a clinician might ask about this chunk
- Semantic metadata (diseases, drugs, procedures, patient subgroups)

This improves retrieval quality significantly but takes longer and uses tokens.

After indexing, the BM25 sparse index (`bm25_index.pkl`) is rebuilt automatically.

**Expected guideline file names** (configured in `src/indexer/pipeline.py`):

| File | Guideline ID |
|---|---|
| `mammakarzinom_v4.4.pdf` | `mamma` |
| `kolorektales_v3.0.pdf` | `krk` |
| `lungenkarzinom_v4.0.pdf` | `lunge` |
| `prostatakarzinom_v8.0.pdf` | `prosta` |

---

## Running the App

The helper script starts both the FastAPI backend and Streamlit UI, waits for the API health check, and shuts both down cleanly on Ctrl+C.

```bash
python scripts/run_app.py
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

To start services individually:

```bash
# Backend only
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# UI only (requires backend running)
streamlit run src/ui/app.py --server.port 8501
```

---

## API Reference

All endpoints require the `X-API-Key` header (value from `API_KEY` env var).

### Chat

```http
POST /chat
Content-Type: application/json
X-API-Key: dev-secret-key

{
  "query": "Welche Erstlinientherapie empfiehlt die S3-Leitlinie beim HER2+ Mammakarzinom?",
  "session_id": "my-session-123",
  "guideline_id": "mamma",   // optional filter: mamma | krk | lunge | prosta
  "grade": "A"               // optional filter: A | B | 0
}
```

Response is streamed as Server-Sent Events (SSE). Final event contains the full structured payload including `answer_professional`, `answer_plain`, `citations`, `rag_trace`, and `token_usage`.

### Conversations

```http
GET  /conversations                        # list all sessions
GET  /conversations/{session_id}           # load a session
DELETE /conversations/{session_id}         # delete a session
POST /conversations/{session_id}/export?format=json   # export (json | csv | pdf)
```

### Analytics

```http
GET /analytics/overview    # token usage, cost, tool frequency, guideline distribution
```

---

## How It Works

### Query Pipeline (12 nodes)

The graph has **12 nodes** in total, but they are not all defined in the same place. There are two groups:

**Group 1 — 8 nodes with their own dedicated files in `src/graph/nodes/`**

Each of these nodes is complex enough (LLM calls, external APIs, multi-step logic) to justify its own module:

| Node | File | Model | What it does |
|---|---|---|---|
| `guardrail_input` | `src/graph/nodes/guardrail_input.py` | — | Regex: blocks injections, off-topic queries; redacts PII |
| `rewrite` | `src/graph/nodes/rewriter.py` | Gemini 2.5 Flash | Normalises query, extracts guideline/grade filters, detects ambiguity |
| `turn_router` | `src/graph/nodes/turn_router.py` | Gemini 2.5 Flash | Classifies turn intent; routes to memory or full retrieval |
| `agent` | `src/graph/nodes/agent.py` | GPT-4o | 2-iteration tool-calling loop; iteration 1 always calls `search_guidelines` |
| `confidence` | `src/graph/nodes/confidence.py` | — | Mean reranker score of top-3 chunks; escalates if below 0.5 |
| `answer` | `src/graph/nodes/answer.py` | Gemini + GPT-4o | Stage 1: verbatim extraction (Gemini). Stage 2: synthesis (GPT-4o) |
| `guardrail_output` | `src/graph/nodes/guardrail_output.py` | — | Blocks ungrounded answers; restricts dosing; adds patient warnings |
| `external_search` | `src/graph/nodes/external_search.py` | — | Google/DuckDuckGo supplemental snippets appended to response |

**Group 2 — 4 nodes defined as private functions directly inside `src/graph/graph.py`**

These nodes are simple response-formatting functions with no LLM calls. They only read from the existing state and return a formatted dict. Because they are short and self-contained, they live as private functions (prefixed with `_`) inside `graph.py` alongside the graph wiring code:

| Node | Private function in `graph.py` | What it does |
|---|---|---|
| `blocked` | `_blocked_response()` | Reads `input_block_reason` from state and returns it as the final answer. Triggered when the input guardrail detects a prompt injection or off-topic query. |
| `clarification` | `_clarification_response()` | Reads `clarification_rationale` and `expected_clarification` from state and returns a German clarification request. Triggered when the rewriter detects missing clinical dimensions (e.g. tumour stage, therapy line). |
| `repeat_answer` | `_repeat_previous_answer_response()` | Reads all `prior_*` fields from state and returns the previous answer unchanged. Triggered when the current rewritten query exactly matches the previous turn's query. |
| `escalate` | `_multi_query_escalation()` | Generates 4 query variants from the rewritten query and runs each through `search_guidelines_tool`, then merges and deduplicates results. Triggered when the confidence node scores too low. |

All four are registered in `build_graph()` exactly like any other node:

```python
builder.add_node("blocked",       _blocked_response)
builder.add_node("clarification", _clarification_response)
builder.add_node("repeat_answer", _repeat_previous_answer_response)
builder.add_node("escalate",      _multi_query_escalation)
```

The naming convention makes it clear they are internal implementation details, not importable public APIs.

### Retrieval Pipeline

```
Query
  │
  ├─► embed_texts()         OpenAI text-embedding-3-large (3072 dim)
  │
  ├─► Dense search ×2       Milvus HNSW, COSINE metric
  │     • all chunk types   top-20
  │     • recommendations   top-10 (prevents prose burying terse recs)
  │
  ├─► BM25 sparse search    pre-built pkl index, German tokeniser
  │
  ├─► RRF fusion            k=60, 2 passes: rrf(dense_all, dense_rec) → rrf(result, bm25)
  │
  ├─► CrossEncoder rerank   BAAI/bge-reranker-v2-m3, pool=12, batch=8
  │
  └─► Parent expansion      fetch parent chunk text from Milvus, merge page ranges
```

### Answer Generation

**Path A — fresh retrieval:**
```
Gemini extracts verbatim sentences with [N] citation tags
  │
  ├── extraction == "NICHTS" → refuse (anti-hallucination guard)
  └── extraction ok
        │
        GPT-4o synthesises professional answer + plain-language summary
          │
          filter citations to only referenced [N] numbers
```

**Path B — memory reuse (`followup_routing == "memory"`):**
```
Heuristic shortcut? (e.g. "kürzer", "3 Sätze")
  YES → truncate/trim prior answer
  NO  → GPT-4o rewrites prior answer to fit new intent (JSON response)
          │
          fallback to basic text manipulation on parse error
```

---

## Project Structure

```
.
├── src/
│   ├── api/                    FastAPI backend
│   │   ├── main.py             App entry point, middleware
│   │   ├── routes/             chat.py, conversations.py, analytics.py
│   │   ├── auth.py             API key verification
│   │   ├── rate_limit.py       Sliding window rate limiter
│   │   ├── observability.py    JSON logging, trace IDs
│   │   ├── conversation_store.py  SQLite persistence
│   │   ├── export_utils.py     JSON / CSV / PDF export
│   │   └── analytics_service.py
│   │
│   ├── graph/                  LangGraph state machine
│   │   ├── graph.py            build_graph() — 12-node StateGraph
│   │   ├── state.py            RAGState TypedDict (~35 fields)
│   │   ├── permissions.py      RBAC: is_tool_allowed(), is_source_allowed()
│   │   └── nodes/
│   │       ├── guardrail_input.py
│   │       ├── rewriter.py
│   │       ├── turn_router.py
│   │       ├── agent.py        GPT-4o tool-calling loop
│   │       ├── confidence.py
│   │       ├── answer.py       Dual-stage generation
│   │       ├── guardrail_output.py
│   │       └── external_search.py
│   │
│   ├── retrieval/              Search and ranking
│   │   ├── search.py           hybrid_search() — dense + BM25 + RRF
│   │   ├── bm25.py             BM25 index build and load
│   │   ├── reranker.py         CrossEncoder reranking
│   │   ├── expander.py         Parent chunk expansion
│   │   └── postprocess.py      Deduplication
│   │
│   ├── tools/                  Agent tools (6 + web)
│   │   ├── search_guidelines.py
│   │   ├── lookup_empfehlung.py
│   │   ├── compare_guidelines.py
│   │   ├── drug_class_lookup.py
│   │   ├── calculate_bmi.py
│   │   ├── pubmed_search.py
│   │   └── web_search.py
│   │
│   ├── indexer/                PDF ingestion pipeline
│   │   ├── pipeline.py         index_pdf() — main orchestration
│   │   ├── chunker.py          Hierarchical chunking (550 tok, 70 overlap)
│   │   ├── embedder.py         embed_texts() — batch 64
│   │   ├── store.py            MilvusStore — HNSW collection management
│   │   ├── enricher.py         LLM enrichment (headers, HyDE questions, metadata)
│   │   ├── metadata.py         Grade/evidence/section extraction
│   │   └── reference.py        Bibliography parsing
│   │
│   ├── prompts/                System prompts and templates
│   │   ├── agent.py            AGENT_SYSTEM — GPT-4o tool-calling prompt
│   │   ├── rewriter.py         Query rewrite + ambiguity detection
│   │   ├── turn_router.py      Intent classification
│   │   └── answer.py           Extraction and synthesis prompts
│   │
│   ├── ui/                     Streamlit frontend
│   │   ├── app.py              Entry point
│   │   └── components/
│   │       ├── chat_page.py    Main chat interface
│   │       ├── source_cards.py Source chunks + tool call display
│   │       ├── inline_citations.py  [N] annotation
│   │       ├── insights_panels.py   RAG trace + token usage
│   │       ├── analytics_dashboard.py
│   │       └── filters.py      Guideline and grade filters
│   │
│   ├── telemetry.py            Token tracking, cost, tool summarisation
│   └── citations.py            Citation string formatting
│
├── evaluations/                Evaluation framework
│   ├── scripts/
│   │   ├── run_eval.py         Runs test dataset against live API
│   │   └── run_ab_eval.py      A/B comparison between two configs
│   ├── metrics/
│   │   ├── ragas_metrics.py    RAGAs integration (faithfulness, relevancy, etc.)
│   │   ├── retrieval.py        chunk_recall, chunk_precision
│   │   ├── behavioral.py       tool_call_count, external_search_used
│   │   └── similarity.py       answer_similarity, coverage
│   └── ui/
│       └── app.py              Evaluation results dashboard
│
├── scripts/
│   ├── run_app.py              Start API + UI together
│   ├── run_indexer.py          Index PDFs into Milvus
│   └── generate_eval_dataset.py
│
├── tests/                      Pytest test suite
│   ├── api/
│   ├── graph/
│   ├── indexer/
│   ├── retrieval/
│   └── tools/
│
├── data/                       Runtime data (not committed)
│   └── app_state.db            SQLite conversation database
│
├── milvus.db/                  Milvus Lite local database (not committed)
├── bm25_index.pkl              BM25 sparse index (not committed)
├── pyproject.toml
└── .env.example
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run a specific module
pytest tests/retrieval/
pytest tests/graph/

# With verbose output
pytest -v
```

---

## Evaluation

Run the evaluation suite against the live API (requires the app to be running):

```bash
# Full evaluation with RAGAs metrics
python evaluations/scripts/run_eval.py

# A/B comparison between two model configs
python evaluations/scripts/run_ab_eval.py

# View results in the evaluation dashboard
streamlit run evaluations/ui/app.py
```

Metrics computed:
- **Retrieval:** `chunk_recall`, `chunk_precision`, `top_gold_chunk_hit`
- **RAGAs:** `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`, `answer_correctness`
- **Behavioral:** `answer_length`, `tool_call_count`, `external_search_used`
- **Similarity:** `answer_similarity`, `coverage`

Results are saved as `summary.json`, `item_results.json`, `ragas_records.json`, and `metadata.json`.

---

## Key Design Decisions

**Why LangGraph?**
State-machine architecture makes each pipeline step independently testable, traceable, and replaceable. Every node reads from and writes to a single `RAGState` TypedDict, making data flow explicit.

**Why hybrid search?**
Dense vectors capture semantic similarity; BM25 catches exact keyword matches (drug names, recommendation IDs, section numbers). RRF fusion combines them without needing calibrated scores.

**Why dual dense search?**
A single dense search over all chunk types allows long prose sections to outrank short, terse recommendation chunks. A second search restricted to `chunk_type=recommendation` ensures terse clinical recommendations always appear in the candidate pool.

**Why two-stage answer generation?**
Stage 1 (Gemini extraction) forces the model to copy sentences verbatim from retrieved text — no paraphrasing, no knowledge from training data. Stage 2 (GPT-4o synthesis) only rephrases what was already extracted. This makes hallucination structurally hard.

**Why no LLM in the input guardrail?**
Using an LLM to detect prompt injections creates a circular vulnerability — the attacker can craft text that tricks the LLM into approving it. Regex and keyword matching are deterministic and cannot be jailbroken.

---

## Tech Stack

| Component | Technology |
|---|---|
| Graph orchestration | LangGraph |
| LLM — generation | GPT-4o via OpenRouter |
| LLM — cheap tasks | Gemini 2.5 Flash via OpenRouter |
| Embeddings | OpenAI `text-embedding-3-large` (3072 dim) |
| Vector database | Milvus Lite (in-process, `./milvus.db`) |
| Sparse index | BM25 (`rank-bm25`) |
| Reranker | `BAAI/bge-reranker-v2-m3` (sentence-transformers) |
| PDF parsing | PyMuPDF |
| Backend API | FastAPI + Uvicorn |
| Streaming | Server-Sent Events (sse-starlette) |
| Frontend | Streamlit |
| Persistence | SQLite (conversations) |
| Evaluation | RAGAs |

---

## Environment Notes

- **Milvus Lite** runs in-process — no Docker or external Milvus server required for local development. The database lives in `./milvus.db/`.
- **OpenRouter** provides a single API-compatible endpoint for both GPT-4o and Gemini. Only one API key is needed.
- The `MILVUS_URI` env var should be left empty (or unset) when using Milvus Lite. Setting it to an HTTP address enables a full Milvus server.
- For production, set `DATABASE_URL` to a PostgreSQL connection string to enable LangGraph graph checkpointing across server restarts.
