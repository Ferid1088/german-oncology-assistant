# Project Concept: German Oncology Guideline RAG Agent

## What I'm building

A retrieval-augmented generation (RAG) agent that answers questions about German S3 oncology guidelines (Mammakarzinom, Kolorektales Karzinom, Lungenkarzinom, Prostatakarzinom — published by the Leitlinienprogramm Onkologie). The corpus is ~2,000 pages of high-quality clinical German across 4–5 PDFs.

Every answer is delivered in two layers:
1. **Fachliche Antwort** — professional German with formal terminology, Empfehlungsgrade (A/B/0), evidence levels, and inline citations.
2. **In einfachen Worten** — same content in plain German for non-experts, no jargon.

Plus a mandatory medical disclaimer and source list.

The goal is a learning/portfolio project, not a clinical product. All disclaimers reflect this.

---

## Domain choices and why

- **Domain**: German clinical oncology guidelines.
- **Why this domain**: well-structured documents (hierarchical numbered sections, formal recommendations with grades), real-world value, distinctively German (rare in RAG demos), forces honest engineering on faithfulness/citations/refusal.
- **User persona**: oncology professionals as primary, with a lay explanation always appended so non-experts can also understand.

---

## Stack

| Layer | Choice |
|---|---|
| Orchestration | **LangGraph** (stateful graph; LangChain ecosystem) |
| LLM access | **OpenRouter** via OpenAI-compatible SDK |
| Generation model | `openai/gpt-5` |
| Cheap transform model | `google/gemini-2.5-flash` (query rewriting, self-query, contextual headers, hypothetical Qs) |
| Embedding model | `openai/text-embedding-3-large` (3072 dim, multilingual) |
| Reranker | `BAAI/bge-reranker-v2-m3` (local, free, multilingual) |
| Vector store | **Milvus** (dense + native BM25 sparse + metadata filters) |
| Backend | **FastAPI** (mandatory) |
| Frontend | **Streamlit** |
| Auth | OAuth via Authlib — **Google + GitHub** |
| Persistent state | **Postgres** (auth, sessions, messages, feedback, LangGraph checkpoints, Langfuse) |
| Cache + rate limit | **Redis** (GPTCache semantic cache + slowapi rate limit + sessions) |
| Observability | **Langfuse** self-hosted (tracing, traces feed A/B + Ragas) |
| Evaluation | **Ragas** + A/B comparison runner |
| PII redaction | Microsoft **Presidio** |
| Deployment | **Docker Compose** all-in-one |

---

## RAG pipeline (locked design)

Although the system is described as one end-to-end RAG pipeline, the implementation should follow a deep architecture rather than a shallow one. Major functional areas such as ingestion/indexing, retrieval, recommendation lookup, answer generation, guardrails, memory, evaluation, and analytics should remain distinct implementation domains with localized logic and tests, instead of being flattened into generic shared modules.

### Indexing (offline, one-time)
1. **Parse** — PyMuPDF for text extraction from PDFs.
2. **Clean** — strip repeated page headers/footers, de-hyphenate German line breaks, preserve section numbering and Empfehlung blocks, footnote removal, table detection.
3. **Chunk** — hierarchical:
   - Small chunks (400–700 tokens) for retrieval.
   - Parent chunks (full section) for context return.
   - Each formal `Empfehlung X.Y` block is a dedicated standalone chunk.
4. **Enrich**:
   - Metadata: guideline, version, section_path, page range, chunk_type, recommendation_id, recommendation_grade, evidence_level, LLM-extracted entities (drugs, diseases, procedures).
   - **Anthropic-style contextual retrieval**: Gemini Flash generates a short context header per chunk explaining where it sits in the guideline; prepended before embedding.
   - **Hypothetical questions**: Gemini Flash generates 2–3 likely questions per chunk; embedded alongside for retrieval lift.
5. **Embed** — `text-embedding-3-large` on the contextualized text.
6. **Index** — push to Milvus with both dense and BM25 sparse vectors plus all metadata.

### Query-time (LangGraph state machine)
```
input → input guardrail (PII redact, prompt-injection detect, off-topic classifier)
      → conversational rewrite + self-query metadata extraction [Gemini Flash]
      → intent router (factual / comparative / recommendation / external)
      → tool-calling agent loop [GPT-5]
            ↓ calls one or more of:
            - search_guidelines  (the core hybrid retrieval)
            - lookup_empfehlung
            - compare_guidelines
            - drug_class_lookup
            - calculate_bmi
            - pubmed_search
      → confidence check
            ↓ if low: escalate with HyDE + multi-query fusion + decomposition,
                      then re-retrieve
      → answer generation [GPT-5]
            - produces dual-layer (Fachlich + In einfachen Worten)
            - inline citation IDs
            - mandatory disclaimer
            - PubMed disclosure if pubmed_search was used
      → output guardrail (faithfulness check, PII scan)
      → persist to memory (PostgresSaver) + log to Langfuse
      → stream to user
```

### Retrieval sub-pipeline (inside `search_guidelines` tool)
```
query
  → embed (text-embedding-3-large)
  → dense search Milvus top-20
  → BM25 sparse search Milvus top-20
  → RRF fusion → top-20
  → metadata filter (guideline, grade, chunk_type — auto-extracted + UI-explicit)
  → bge-reranker-v2-m3 → top-5
  → parent-document expansion (retrieve small, return parent section context)
  → return ranked chunks with full metadata for citation
```

### Query transformation cascade (conditional)
- **Always**: conversational rewrite + self-query metadata extraction (cheap).
- **On low confidence**: HyDE + multi-query fusion + decomposition (expensive, only when needed).

---

## Six tools (LangGraph-bound functions)

1. **`search_guidelines(query, guideline?, grade?, top_k=5)`** — core hybrid retrieval; returns chunks with text + section_path + page + grade.
2. **`lookup_empfehlung(guideline, recommendation_id)`** — fetch a specific Empfehlung X.Y verbatim with grade and evidence level.
3. **`compare_guidelines(topic, guideline_a, guideline_b)`** — side-by-side comparison of two guidelines on a topic.
4. **`drug_class_lookup(substance_name)`** — find all mentions of a drug across guidelines, grouped by indication and grade.
5. **`calculate_bmi(weight_kg, height_cm)`** — BMI + WHO category in German; input validation enforced.
6. **`pubmed_search(query, max_results=5)`** — searches PubMed E-utilities for literature beyond the corpus. Every PubMed result carries a mandatory disclosure: *"Quelle: U.S. National Library of Medicine (NLM) – PubMed. Diese Ergebnisse stammen aus externen Datenquellen außerhalb der deutschen S3-Leitlinien."*

---

## Memory and personalization

- **LangGraph PostgresSaver** for per-user persistent conversation state.
- **Summary buffer** rolls over long sessions.
- Per-user history, saved searches, feedback persisted in Postgres.
- Auth identity (Google/GitHub OAuth) is the memory partition key.

---

## Guardrails

- **Input**: PII redaction via Presidio; prompt-injection patterns flagged; off-topic classifier refuses non-oncology questions.
- **Retrieval**: source whitelisting (only indexed guidelines + optional PubMed), permission checks tied to user role.
- **Output**: faithfulness check against retrieved context; PII scan; mandatory disclaimer; no individual treatment recommendations or dosages.
- **Roles** (from OAuth): `user` (default), `professional` (manual upgrade), `admin` (eval + A/B controls).

---

## Evaluation and A/B

- **Eval set**: ~30 hand-curated German Q&A pairs derived from the corpus (LLM-generated, manually filtered).
- **Ragas metrics**: context precision, context recall, faithfulness, answer relevancy, answer correctness.
- **A/B variants**:
  - A = hybrid retrieval + reranker
  - B = hybrid retrieval, no reranker
- Variant tag stored per message; eval runner produces per-variant Ragas scores.
- Results surfaced in Streamlit Evaluation page.

Detailed execution artifacts for this layer are documented separately in:
- `docs/testing-and-evaluation-strategy.md`
- `docs/evaluation-dataset.schema.json`

Evaluation should exist at both the system level and the feature level. In addition to the shared benchmark and A/B comparisons, major feature areas such as retrieval, answer generation, guardrails, recommendation lookup, and external literature search should each have focused tests so that failures can be localized to a specific domain rather than only observed in end-to-end runs.

### Citation model

The app uses multiple citation/provenance layers that should stay conceptually distinct:

- **Chunk provenance**: chunk metadata tracks guideline, section, page range, and source file.
- **Answer citation label**: a stable citation label can be shown in the UI and inline answers.
- **Bibliography linkage**: in-text references such as `[1189]` can be linked from chunk metadata to structured reference entries parsed from the guideline bibliography.

This separation is important because a source card citation and a bibliography reference are not the same thing.

---

## Observability and analytics

- **Langfuse self-hosted** traces every LangGraph run, tool call, LLM call, latency, cost.
- Streamlit Analytics page embeds Langfuse dashboards: queries/day, latency, cost, top guidelines, top tools, feedback ratios.

---

## Caching, rate limiting, cost control

- **Semantic cache** via GPTCache on Redis: caches near-duplicate queries.
- **Rate limiting**: per-user requests/minute + daily cost cap, Redis-backed via slowapi.
- **API key management**: env vars only, never committed.

---

## UI (Streamlit)

- Chat page with SSE streaming.
- Source cards: guideline + section + page + excerpt + open-PDF link.
- Tool-call cards (expandable).
- Filter panel: guideline / recommendation grade.
- Model picker (for A/B).
- Feedback buttons (thumbs up/down → Postgres + Langfuse).
- History, Saved searches, Analytics, Evaluation, Admin pages.

---

## Deployment

Docker Compose with:
- `milvus-standalone` + `etcd` + `minio`
- `postgres` (app DB + Langfuse DB)
- `redis`
- `langfuse-server`
- `backend` (FastAPI + LangGraph)
- `frontend` (Streamlit)
- `indexer` (one-shot job)

Single `docker compose up` launches everything; `.env` carries all secrets.

---

## What I want to discuss

I'd like another AI agent's perspective on:

1. Whether the locked design is sound for this domain, or if any stage is over- or under-engineered.
2. Specific risks: PDF parsing quality on these guidelines, embedding quality on German medical text, faithfulness of the dual-layer answer.
3. How to construct the 30-question Ragas eval set efficiently and with adequate coverage.
4. Prompt design for the dual-layer (Fachlich + In einfachen Worten) answer — how to avoid the lay version being a useless rephrase and the professional version being verbose.
5. The LangGraph state machine — node boundaries, when escalation should fire, how the confidence check should be implemented.
6. PubMed integration — handling rate limits and aligning external results with the guideline citations cleanly.
7. Anything I've overlooked (compliance, GDPR for OAuth user data, copyright on guideline excerpts in answers).

The project is for personal learning; nothing will be deployed clinically or published.
