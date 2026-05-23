# German Oncology Assistant

A retrieval-augmented generation (RAG) system for querying German S3 oncology clinical guidelines. Clinical questions are asked in German; the system retrieves evidence-based answers from four cancer guidelines and generates professional and plain-language answers with full citations.

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

### Main Pipeline

```
                              User Query
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │       GUARDRAIL INPUT        │
                   │  no LLM · rule-based only    │
                   │  ① prompt injection detect   │──── blocked ──► [BLOCKED] ──► END
                   │  ② off-topic keyword block   │
                   │  ③ PII redaction (no block)  │
                   └──────────────┬───────────────┘
                                  │ pass
                                  ▼
                   ┌──────────────────────────────┐
                   │            REWRITE           │
                   │  Gemini 2.5 Flash · 400 tok  │──── clarification ──► [CLARIFICATION] ──► END
                   │  → rewritten_query           │
                   │  → filters {guideline, grade}│──── duplicate ──────► [REPEAT ANSWER]  ──► END
                   │  → intent classification     │
                   │  → query decomposition       │
                   └──────────────┬───────────────┘
                                  │ ok
                                  ▼
                   ┌──────────────────────────────┐
                   │          TURN ROUTER         │
                   │  ① heuristic keyword match   │
                   │    vereinfach→simplify        │
                   │    kürzer→shorten  etc.       │
                   │  ② Gemini (only if needed)   │
                   │  → followup_routing:         │
                   │    "retrieve" | "memory"     │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │             AGENT            │──── memory ──► reuse prior chunks (no LLM)
                   │  GPT-4o · max 2 iterations   │
                   │  iter 1: FORCED search       │
                   │  iter 2: auto (any or stop)  │
                   │  RBAC checked per tool call  │
                   └──────────────┬───────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
 ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
 │search_guidelines│   │lookup_empfehlung │   │ compare_guidelines  │
 │  hybrid search  │   │  exact filter    │   │ search × 4 guidel.  │
 │  (see below)    │   │ recommendation_id│   │ results grouped     │
 └────────┬────────┘   └────────┬─────────┘   └──────────┬──────────┘
          │                     │                         │
          ▼                     ▼                         ▼
 ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────────┐
 │drug_class_lookup│   │  calculate_bmi   │   │   pubmed_search     │
 │ drug name search│   │ BMI = kg/m²      │   │ esearch → esummary  │
 │ across 4 guides │   │ + WHO category   │   │ 5 structured results│
 └────────┬────────┘   └────────┬─────────┘   └──────────┬──────────┘
          └───────────────────────┴───────────────────────┘
                                  │  retrieved_chunks
                                  ▼
                   ┌──────────────────────────────┐
                   │          CONFIDENCE          │
                   │  score = mean(top-3 reranker)│──── score<0.5 ──────────────────────────┐
                   │  threshold : 0.5             │          or                             │
                   │  min chunks: 2               │──── chunks<2  ──────────────────────────┤
                   └──────────────┬───────────────┘                                        │
                                  │ ok                                          ┌───────────▼──────────┐
                                  │                                             │        ESCALATE      │
                                  │                                             │  4 query variants:   │
                                  │                                             │  · rewritten_query   │
                                  │                                             │  · decomposed        │
                                  │                                             │    items             │
                                  │                                             │  · "Leitlinienempf.."│
                                  │◄────────────────────────────────────────────  → merged top-10     │
                                  ▼                                             └──────────────────────┘
                   ┌──────────────────────────────┐
                   │             ANSWER           │
                   │  Path A — fresh retrieval    │
                   │  ┌──────────────────────┐   │
                   │  │ ① Gemini EXTRACT     │   │
                   │  │   verbatim [N] tags  │   │
                   │  │   "NICHTS" → skip ②  │   │
                   │  └──────────┬───────────┘   │
                   │             ▼               │
                   │  ┌──────────────────────┐   │
                   │  │ ② GPT-4o SYNTHESIZE  │   │
                   │  │   {professional,     │   │
                   │  │    plain} + citations│   │
                   │  │   + DISCLAIMER       │   │
                   │  └──────────────────────┘   │
                   │  Path B — memory reuse       │
                   │  heuristic (shorten/simplify)│
                   │  or GPT-4o rewrite of prior  │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │       GUARDRAIL OUTPUT       │
                   │  no LLM · rule-based only    │──── blocked ──► END  (safety fields set)
                   │  ① patient-specific detect   │
                   │  ② dosage query detect       │
                   │  ③ grounding verify          │
                   │    (lifts block if grounded) │
                   └──────────────┬───────────────┘
                                  │ pass
                                  ▼
                   ┌──────────────────────────────┐
                   │        EXTERNAL SEARCH       │
                   │  Google CSE → DuckDuckGo     │
                   │  → [] on failure             │
                   │  skip if: blocked / clarif / │
                   │   no-web-perm / prior exists │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                                 END
                         SSE → Streamlit UI
                   data: { answer, citations, tool_calls,
                           rag_trace, token_usage, trace_id }
                   data: [DONE]
```

### Hybrid Search *(called by `search_guidelines`)*

```
          query string
               │
               ▼
   ┌───────────────────────┐
   │  text-embedding-3-    │   3072 dim · batch 64
   │  large  (OpenAI)      │
   └─────────┬─────────────┘
             │  query vector
     ┌───────┴────────┐
     │                │
     ▼                ▼
┌──────────┐   ┌──────────────┐   ┌──────────────────────┐
│ Milvus   │   │   Milvus     │   │   BM25 sparse        │
│ ANN #1   │   │   ANN #2     │   │   rank_bm25.         │
│ all types│   │ recomm. only │   │   OkapiBM25          │
│ top-20   │   │ top-10       │   │   bm25_index.pkl     │
│ COSINE   │   │ COSINE/HNSW  │   │   top-20 matches     │
│ HNSW     │   │              │   │                      │
└────┬─────┘   └──────┬───────┘   └──────────┬───────────┘
     │                │                       │
     └────────┬───────┘                       │
              ▼                               │
   ┌─────────────────────┐                    │
   │   RRF  round 1      │                    │
   │  fuse(ANN#1, ANN#2) │                    │
   │  k = 60             │                    │
   └──────────┬──────────┘                    │
              └──────────────┬────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │   RRF  round 2      │
                  │ fuse(round1, BM25)  │
                  │ k = 60              │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │   CrossEncoder      │
                  │ BAAI/bge-reranker   │
                  │ -v2-m3 (local)      │
                  │ up to 30 candidates │
                  │ fallback: RRF order │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │  Parent Expansion   │
                  │ fetch parent chunk  │
                  │ prepend to leaf text│
                  │ merge page ranges   │
                  │ silent on failure   │
                  └──────────┬──────────┘
                             ▼
                      top-5 chunks
                   (RetrievedChunk[])
```

### LLM Calls per Request *(happy path)*

```
  Node            Model                  Purpose                  Tokens
  ──────────────────────────────────────────────────────────────────────
  rewrite         Gemini 2.5 Flash       expand + classify query    400
  turn_router     Gemini 2.5 Flash       intent classification      200  ← only if heuristic fails
  agent iter 1    GPT-4o                 forced search call          —
  agent iter 2    GPT-4o                 optional extra tools        —
  answer extract  Gemini 2.5 Flash       verbatim extraction       1500
  answer synth    GPT-4o                 professional + plain JSON  1200
  ──────────────────────────────────────────────────────────────────────
  All via OpenRouter  (one API key · one base URL)
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
git clone https://github.com/Ferid1088/german-oncology-assistant.git
cd german-oncology-assistant

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -e .
pip install -e ".[dev]"         # dev dependencies (tests)
```

---

## Configuration

```bash
cp .env.example .env
```

`.env` reference:

```env
# Required — get a key at https://openrouter.ai
OPENROUTER_API_KEY=your_key_here

# LLM models
GENERATION_MODEL=openai/gpt-4o
CHEAP_MODEL=google/gemini-2.5-flash
EMBEDDING_MODEL=openai/text-embedding-3-large

# Milvus — leave empty to use Milvus Lite (./milvus.db, no server required)
MILVUS_URI=
MILVUS_COLLECTION=oncology_guidelines

# API authentication — change in production
API_KEY=dev-secret-key
# Multiple comma-separated keys:
# API_KEYS=key1,key2,key3

# Conversation database (SQLite)
CONVERSATION_DB_PATH=data/app_state.db

# Optional: Google Custom Search for external web results
# GOOGLE_SEARCH_API_KEY=
# GOOGLE_SEARCH_ENGINE_ID=

# Logging level: DEBUG | INFO | WARNING | ERROR
LOG_LEVEL=INFO

# Optional: PostgreSQL for LangGraph checkpointing
# DATABASE_URL=postgresql://postgres:postgres@localhost:5432/oncology_rag
```

---

## Indexing the Guidelines

Place PDF files in the `data/` directory and run:

```bash
python scripts/run_indexer.py                        # All 4 guidelines
python scripts/run_indexer.py --pdf mammakarzinom_v4.4.pdf  # Single guideline
python scripts/run_indexer.py --dry-run              # Parse only, no database writes
python scripts/run_indexer.py --no-enrich            # Faster, without LLM enrichment
```

With enrichment enabled (default), Gemini 2.5 Flash generates per chunk:
- A contextual header (clinical context and content)
- 2–3 hypothetical questions a clinician might ask about this chunk
- Semantic metadata (diseases, drugs, procedures, patient subgroups)

**Expected file names** (configured in `src/indexer/pipeline.py`):

| File | Guideline ID |
|---|---|
| `mammakarzinom_v4.4.pdf` | `mamma` |
| `kolorektales_v3.0.pdf` | `krk` |
| `lungenkarzinom_v4.0.pdf` | `lunge` |
| `prostatakarzinom_v8.0.pdf` | `prosta` |

---
## Indexing Pipeline  *(`python scripts/run_indexer.py`)*

```
   PDF  (docs/knowledge_base/)
        │
        ▼
   ┌─────────┐     extract pages · detect printed page numbers
   │  PARSE  │     skip TOC · strip copyright header
   └────┬────┘     repair hyphenation · merge continuation lines
        │
        ▼
   ┌──────────┐    single-pass state machine → StructuralUnit:
   │ DETECT   │    heading | empfehlung | bibliography | prose
   └────┬─────┘
        │
        ▼
   ┌───────┐       heading    → parent Chunk  (is_leaf=False)
   │ CHUNK │       empfehlung → leaf   Chunk  (chunk_type=recommendation)
   └───┬───┘       prose      → sliding-window leaves
        │               TARGET_TOKENS=550 · OVERLAP_TOKENS=70
        ▼
   ┌──────────┐    attach: source_filename · is_current
   │ METADATA │    assign: chunk_index_in_parent
   └────┬─────┘
        │
        ▼
   ┌────────┐      3 LLM calls per chunk  (Gemini 2.5 Flash)
   │ ENRICH │  ①  contextual header  (prepended before embed)
   │optional│  ②  hypothetical questions  (HyDE)
   └────┬───┘  ③  semantic metadata  {diseases, drugs, …}
        │
        ▼
   ┌───────┐       text-embedding-3-large · 3072 dim
   │ EMBED │       input: header + questions + chunk_text
   └───┬───┘       batch size: 64
        │
        ▼
   ┌────────┐      Milvus Lite  (./milvus.db)
   │ UPSERT │      HNSW: M=16 · efConstruction=200 · COSINE
   └────┬───┘      dynamic fields for enricher metadata
        │
        ▼
   ┌─────────────┐  query all is_leaf chunks (paginated 1000/page)
   │ BM25 REBUILD│  build OkapiBM25 → bm25_index.pkl
   └─────────────┘  reload in-process singleton
```

---
## Running the App

```bash
python scripts/run_app.py
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI backend | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

Start services individually:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
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
  "guideline_id": "mamma",   // optional: mamma | krk | lunge | prosta
  "grade": "A"               // optional: A | B | 0
}
```

Response is streamed as Server-Sent Events (SSE). The final event contains `answer_professional`, `answer_plain`, `citations`, `rag_trace` and `token_usage`.

### Conversations

```http
GET    /conversations                             # list all sessions
GET    /conversations/{session_id}                # load a session
DELETE /conversations/{session_id}                # delete a session
POST   /conversations/{session_id}/export?format=json   # export (json | csv | pdf)
```

### Analytics

```http
GET /analytics/overview    # token usage, cost, tool frequency, guideline distribution
```

---

## Node Overview (12 nodes)

**Group 1 — 8 nodes with dedicated files in `src/graph/nodes/`:**

| Node | File | Model | What it does |
|---|---|---|---|
| `guardrail_input` | `guardrail_input.py` | — | Regex: blocks injections, off-topic; redacts PII |
| `rewrite` | `rewriter.py` | Gemini 2.5 Flash | Normalises query, extracts filters, detects ambiguity |
| `turn_router` | `turn_router.py` | Gemini 2.5 Flash | Classifies turn intent; routes to memory or retrieval |
| `agent` | `agent.py` | GPT-4o | 2-iteration tool-calling loop; iter 1 always calls `search_guidelines` |
| `confidence` | `confidence.py` | — | Mean reranker score top-3; escalates if score < 0.5 |
| `answer` | `answer.py` | Gemini + GPT-4o | Stage 1: verbatim extraction (Gemini). Stage 2: synthesis (GPT-4o) |
| `guardrail_output` | `guardrail_output.py` | — | Blocks ungrounded answers; restricts dosing |
| `external_search` | `external_search.py` | — | Google/DuckDuckGo supplemental snippets |

**Group 2 — 4 nodes as private functions in `src/graph/graph.py`:**

| Node | Function | What it does |
|---|---|---|
| `blocked` | `_blocked_response()` | Returns `input_block_reason` as the final answer |
| `clarification` | `_clarification_response()` | Returns a German clarification request |
| `repeat_answer` | `_repeat_previous_answer_response()` | Returns the previous answer unchanged |
| `escalate` | `_multi_query_escalation()` | Generates 4 query variants and runs fallback retrieval |

---

## Key Design Decisions

**Why LangGraph?**
Every pipeline step is independently testable, traceable, and replaceable. Each node reads from and writes to a single `RAGState` TypedDict.

**Why hybrid search?**
Dense vectors capture semantic similarity; BM25 catches exact keyword matches (drug names, recommendation IDs, section numbers). RRF fusion combines both without needing calibrated scores.

**Why dual dense search?**
A single dense search over all chunk types allows long prose sections to outrank short recommendation chunks. A second search restricted to `chunk_type=recommendation` ensures clinical recommendations always appear in the candidate pool.

**Why two-stage answer generation?**
Stage 1 (Gemini extraction) forces the model to copy sentences verbatim from retrieved text — no paraphrasing, no training data knowledge. Stage 2 (GPT-4o synthesis) only rephrases what was already extracted. This makes hallucination structurally hard.

**Why no LLM in the input guardrail?**
Using an LLM to detect prompt injections creates a circular vulnerability. Regex and keyword matching are deterministic and cannot be jailbroken.

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
│   │   └── nodes/              8 node modules (see above)
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
│   ├── ui/                     Streamlit frontend
│   │   ├── app.py              Entry point
│   │   └── components/
│   │       ├── chat_page.py
│   │       ├── source_cards.py
│   │       ├── inline_citations.py
│   │       ├── insights_panels.py
│   │       ├── analytics_dashboard.py
│   │       └── filters.py
│   │
│   ├── telemetry.py            Token tracking, cost, tool summarisation
│   └── citations.py            Citation string formatting
│
├── evaluations/                Evaluation framework
│   ├── scripts/
│   │   ├── run_eval.py         Runs test dataset against live API
│   │   └── run_ab_eval.py      A/B comparison between two configs
│   ├── metrics/
│   │   ├── ragas_metrics.py    RAGAs integration
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
pytest                    # All tests
pytest tests/retrieval/   # Single module
pytest -v                 # Verbose output
```

---

## Evaluation

Run the evaluation suite against the live API:

```bash
python evaluations/scripts/run_eval.py        # Full evaluation with RAGAs metrics
python evaluations/scripts/run_ab_eval.py     # A/B comparison between two model configs
streamlit run evaluations/ui/app.py           # View results in the dashboard
```

Metrics computed:
- **Retrieval:** `chunk_recall`, `chunk_precision`, `top_gold_chunk_hit`
- **RAGAs:** `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`, `answer_correctness`
- **Behavioral:** `answer_length`, `tool_call_count`, `external_search_used`
- **Similarity:** `answer_similarity`, `coverage`

Results are saved as `summary.json`, `item_results.json`, `ragas_records.json` and `metadata.json`.

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

## Security Layers

```
  INPUT                              OUTPUT
  ─────────────────────────────────  ────────────────────────────────────
  prompt injection → substring block patient-specific → regex → blocked
  off-topic        → keyword block   dosage w/o source → blocked
  PII              → regex redact    dosage w/ source  → allowed (grounded)
  auth             → API key check   hallucination     → NICHTS guard
  rate limit       → sliding window  training data     → GPT-4o never sees
                     per(route·key·ip)                   raw chunks
```

---
## Environment Notes

- **Milvus Lite** runs in-process — no Docker or external Milvus server required for local development. The database lives in `./milvus.db/`.
- **OpenRouter** provides a single API-compatible endpoint for both GPT-4o and Gemini. Only one API key is needed.
- Leave `MILVUS_URI` empty (or unset) for Milvus Lite. Setting it to an HTTP address enables a full Milvus server.
- For production, set `DATABASE_URL` to a PostgreSQL connection string to enable LangGraph graph checkpointing across server restarts.


---

## Key Abbreviations & Concepts



| Term | Meaning in this project |
|---|---|
| **Leitlinie** | German word for clinical guideline. S3 is the highest evidence level (systematic evidence review). |
| **Empfehlung** | Recommendation — a specific clinical action statement inside a Leitlinie, e.g. "Empfehlung 4.2.1". |
| **Empfehlungsgrad** | Recommendation grade: A (strong), B (moderate), 0 (open). |
| **Evidenzlevel** | Evidence level: 1a/1b (RCTs), 2a/2b (cohort studies), 3/4/5 (lower evidence). |
| **Leaf chunk** | A chunk with no children — the actual content chunk that gets embedded and searched. Parent chunks are sections that contain leaf chunks. |
| **Dense vector / embedding** | A 3072-dimensional float array representing the semantic meaning of a text. Computed by OpenAI's `text-embedding-3-large`. Semantically similar texts have similar vectors. |
| **ANN (Approximate Nearest Neighbor)** | Algorithm to find the closest vectors to a query vector without comparing to every vector. Milvus uses HNSW for this. |
| **HNSW** | Hierarchical Navigable Small World. A graph-based ANN index that allows very fast approximate nearest-neighbor search on high-dimensional vectors. Parameters: M=16 (connections per node), efConstruction=200 (build-time quality). |
| **BM25** | Best Match 25. A classical keyword-based ranking function. Unlike dense search, it ranks documents by exact term frequency and inverse document frequency. Good for exact drug names, recommendation IDs. BM25Okapi is a popular variant with parameter tuning. |
| **Sparse retrieval** | Search based on keyword matching (BM25). Returns only chunks that contain the exact query terms. |
| **Dense retrieval** | Search based on embedding similarity (Milvus). Returns chunks that are semantically similar even if they use different words. |
| **RRF (Reciprocal Rank Fusion)** | A score fusion formula: `score = Σ 1/(k + rank + 1)` for each ranked list. Combines multiple ranked lists into one without needing calibrated scores. k=60 is the standard smoothing constant. |
| **Hybrid search** | Combination of dense + sparse retrieval fused via RRF. Gets the best of both worlds: semantic matching from dense, exact term matching from sparse. |
| **Cross-encoder / Reranker** | A model (`BAAI/bge-reranker-v2-m3`) that takes a (query, passage) pair as input and outputs a relevance score. More accurate than embedding similarity but slower — used after an initial fast retrieval to re-sort the top candidates. |
| **Parent expansion** | Each leaf chunk has a `parent_chunk_id`. Before returning results, the system fetches the parent chunk text and prepends it to the leaf text. This gives the LLM more context about where the sentence comes from.|
| **TypedDict** | Python type hint for dict with known keys and typed values. `RAGState` is a TypedDict containing all fields that flow through the graph. |
| **OpenRouter** | API proxy that routes calls to multiple LLM providers (OpenAI, Google, Anthropic) using a single API key and unified OpenAI-compatible interface. |
| **Gemini 2.5 Flash** | Google's fast, cheap model. Used here for operations that need to be quick and cost-efficient: query rewriting, turn classification, verbatim extraction. |
| **GPT-4o** | OpenAI's high-capability model. Used here for: running the tool-calling agent loop, synthesizing the final answer, memory-based rewrites. |
| **SSE (Server-Sent Events)** | HTTP protocol for server-to-client streaming. Used by the FastAPI backend to stream partial answers to the Streamlit UI as they are generated. |
| **RBAC** | Role-Based Access Control. Users have a `user_role` in state (user/professional/admin) and this determines which tools they can call. |
| **PII** | Personally Identifiable Information. Patient names, birthdates, phone numbers, postal codes. The input guardrail redacts these before the query is processed further. |
