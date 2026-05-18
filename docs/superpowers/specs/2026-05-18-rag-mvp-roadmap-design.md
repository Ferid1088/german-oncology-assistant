# Design Spec: German Oncology Guideline RAG App — MVP & Phased Roadmap

**Date:** 2026-05-18
**Status:** Approved in brainstorming

---

## Context

A retrieval-augmented generation app for German S3 oncology guidelines (Mammakarzinom, Kolorektales Karzinom, Lungenkarzinom, Prostatakarzinom). The corpus is ~2,000 pages across 4 PDFs. Every answer delivers two layers: a professional Fachliche Antwort and a plain-language In einfachen Worten explanation, with inline citations and a mandatory disclaimer.

This is a portfolio and learning project, not a clinical product. No Docker/container orchestration — all external services (Milvus, Postgres, Redis, Langfuse) run locally and are managed separately. The app connects to localhost.

Full stack is defined in `docs/project_concept.md`. This spec focuses on MVP definition, feature decomposition, risks, and the phased implementation roadmap.

---

## MVP Definition

**The MVP is complete when:** a user can type a German oncology question in the Streamlit chat UI, the system retrieves relevant guideline chunks via hybrid search, and returns a dual-layer answer (Fachlich + plain German) with inline citations and a disclaimer. Source cards (guideline, section, page, excerpt) are shown alongside the answer.

MVP uses one tool (`search_guidelines`) and one LangGraph path (no HyDE fallback, no multi-query). All other features build on this foundation.

MVP does NOT require: OAuth, Redis cache, Ragas runner, A/B eval page, analytics page, Presidio (regex fallback acceptable), PubMed integration, or the full 6-tool suite.

---

## Features (F1–F11)

| # | Feature | Covers |
|---|---|---|
| F1 | Indexing Pipeline | Parse → clean → detect → chunk → metadata → LLM enrich → embed → Milvus |
| F2 | Retrieval Engine | Dense + BM25 + RRF + metadata filter + reranker + parent expansion |
| F3 | LangGraph State Machine | All nodes: guardrail, rewrite, route, tool loop, confidence, answer, output guard, persist |
| F4 | Six Tools | search_guidelines, lookup_empfehlung, compare_guidelines, drug_class_lookup, calculate_bmi, pubmed_search |
| F5 | FastAPI Backend | /chat SSE streaming, /feedback, auth middleware (delivery layer only — no business logic) |
| F6 | Auth + User Roles | Google + GitHub OAuth via Authlib; user/professional/admin roles |
| F7 | Memory + State | LangGraph PostgresSaver, summary buffer, per-user history/saved searches/feedback |
| F8 | Observability | Langfuse local (or Cloud): traces, tool/LLM calls, latency, cost; Analytics page |
| F9 | Evaluation Pipeline | 30-item Q&A eval set, guardrail dataset, Ragas runner, A/B comparison (reranker on/off) |
| F10 | Streamlit UI | Chat, source cards, tool-call cards, filter, feedback, history, eval, analytics, admin pages |
| F11 | Caching + Rate Limiting | GPTCache on Redis, slowapi per-user rate limits + daily cost cap |

**Excluded:** Docker Compose orchestration — services are managed locally, outside the project.

---

## Dependency Map

```
F1: Indexing Pipeline
    └── F2: Retrieval Engine
            ├── F4 (retrieval tools): search_guidelines, lookup_empfehlung,
            │   compare_guidelines, drug_class_lookup
            │       └── F3: LangGraph State Machine ←── also receives:
            │               │   F4 (standalone tools): calculate_bmi, pubmed_search
            │               │   (these do not depend on F2)
            │               ├── F5: FastAPI Backend
            │               │       ├── F6: Auth + User Roles
            │               │       ├── F7: Memory + State  (needs Postgres)
            │               │       ├── F8: Observability   (needs Langfuse)
            │               │       ├── F10: Streamlit UI
            │               │       └── F11: Caching + Rate Limiting (needs Redis)
            │               └── F9: Evaluation Pipeline
            └── F9: Evaluation Pipeline  (also needs indexed corpus)
```

**Critical path:** F1 → F2 → F4 (search_guidelines) → F3 → F5 → F10

**Architecture constraint:** Business logic lives in the LangGraph/core layer only. FastAPI is a delivery layer (routes delegate, no logic). Streamlit is a UI layer (callbacks delegate, no logic). This enforces the deep architecture principle from `docs/project_concept.md`.

---

## Phased Roadmap (Depth-First)

### Phase 0A — Parse + Clean + Detect + Chunk (one PDF)

One guideline only (start with Mammakarzinom). Validate output before proceeding.

- PyMuPDF extraction, page by page
- Text cleaning: dehyphenation, header/footer removal, paragraph merging
- Structural detection: numbered headings (`^\d+(\.\d+)*\s+`), Empfehlung blocks, bibliography markers
- Hierarchical chunking: leaf 400–700 tokens + parent (full subsection); each `Empfehlung X.Y` = standalone chunk
- All **reliably extractable structural fields** from `docs/guideline-chunk-metadata.schema.json` (structural pass only — not all schema fields)
- Reference extraction: in-text citation markers + bibliography → reference store

**Milestone:** Manual inspection confirms chunk boundaries respected, Empfehlung blocks isolated, structural metadata present.

---

### Phase 0B — Enrichment + Embed + Milvus (same one PDF)

- LLM enrichment via Gemini Flash: diseases, drugs, procedures, patient_subgroups, contextual headers, hypothetical questions
- Embedding: `text-embedding-3-large` (3072 dim)
- Milvus configured for hybrid search: dense vector similarity + sparse keyword (BM25) search over the same corpus, with metadata filtering
- Push enriched chunks + vectors + metadata to Milvus

**Milestone:** 10 test queries return relevant chunks with correct metadata. BM25 confirmed working.

---

### Phase 0C — Rollout to All 4 PDFs

- Apply validated pipeline to Kolorektales, Lungenkarzinom, Prostatakarzinom
- Per-guideline heuristic tuning where layouts differ
- Full validation pass across all guidelines

**Milestone:** All ~2,000 pages indexed. Metadata coverage acceptable across all 4 guidelines.

---

### Phase 1 — Retrieval Engine `[F2]`

- Dense top-20 + BM25 top-20 + RRF fusion → top-20
- Metadata filtering (guideline, grade, chunk_type)
- `bge-reranker-v2-m3` local reranking → top-5
- Parent-document expansion
- **Early eval artifact:** 5–10 smoke queries, manual check of retrieved chunks vs expected sections

**Milestone:** `search_guidelines` returns correct top-5 reranked chunks for smoke queries.

---

### Phase 2 — Core LangGraph: MVP Path First, Then Full Nodes `[F3 partial, F4 partial]`

*Risk-heavy phase. Build the minimal path first, then extend within the same phase.*

**Minimal path first:**

- LangGraph state schema + project setup
- Query rewriting node (Gemini Flash)
- `search_guidelines` tool wired to retrieval engine
- Tool-calling agent loop (GPT-5 via OpenRouter — verify model ID before writing generation code)
- Answer generation: dual-layer (Fachlich + plain German) + inline citations + disclaimer
- **Early eval artifact:** 5 smoke questions; manually inspect dual-layer quality and citation grounding

**Then add within Phase 2:**

- Input guardrail: off-topic classifier + Presidio PII redaction (regex fallback acceptable for MVP)
- Self-query metadata extraction (Gemini Flash)
- Intent router (factual / recommendation / comparison / external)
- Output guardrail: faithfulness check, PII scan
- Confidence check: lightweight signal only — based on reranker score threshold or retrieved-chunk count, not an additional LLM call. Triggers multi-query escalation. HyDE deferred to Phase 11A.
- `lookup_empfehlung` tool — pulled forward from Phase 5; directly tied to indexed Empfehlung blocks, useful for early validation
- In-memory state (Postgres persistence deferred to Phase 6)

**Milestone:** CLI test — type a German question, receive a cited dual-layer answer. `lookup_empfehlung` also functional.

---

### Phase 3 — FastAPI Backend `[F5]`

*Delivery layer only. No business logic in routes.*

- `/chat` SSE streaming endpoint (delegates entirely to LangGraph)
- `/feedback` endpoint (writes to store, no logic)
- API-key auth placeholder (OAuth deferred to Phase 7)
- Health check endpoint

**Milestone:** `curl /chat` streams a dual-layer answer. LangGraph logic is untouched by route code.

---

### Phase 4 — Streamlit Chat UI `[F10 partial]` ← MVP complete

*UI layer only. No business logic in callbacks.*

- Chat page with SSE streaming
- Source cards (guideline + section + page + excerpt + PDF link)
- Tool-call cards (expandable)
- Filter panel (guideline, recommendation grade)
- Feedback buttons (thumbs up/down)

**Milestone:** Open browser, ask a question, see dual-layer answer with source cards. **MVP done.**

---

### Phase 5 — Remaining Tools `[F4]`

- `compare_guidelines` — side-by-side topic comparison across two guidelines
- `drug_class_lookup` — drug mentions grouped by indication + grade across all guidelines
- `calculate_bmi` — BMI + WHO category in German, validated input
- `pubmed_search` — PubMed E-utilities, rate limit handling, mandatory disclosure label on every result
- Wire all tools into agent loop; test intent routing to each tool

**Milestone:** Each tool invokable via the chat UI with correct output.

---

### Phase 6 — Postgres + Memory `[F7]`

- Postgres schema: users, sessions, messages, feedback
- LangGraph `PostgresSaver` replacing in-memory state
- Summary buffer for long sessions
- Per-user history, saved searches, feedback persistence
- Before OAuth exists (Phase 7), the session key is the API key from Phase 3 — conversation state is already partitioned per key. Phase 7 replaces that placeholder identity with a real OAuth-derived user record without changing the persistence model.

**Milestone:** Conversation history persists across sessions, keyed by API key.

---

### Phase 7 — Auth + User Roles `[F6]`

- Authlib: Google OAuth + GitHub OAuth flows
- Roles: `user` / `professional` / `admin`
- Auth middleware in FastAPI; login UI in Streamlit

**Milestone:** Login with Google or GitHub; protected endpoints enforce roles.

---

### Phase 8 — Observability `[F8]`

- Langfuse local setup (fall back to Langfuse Cloud if local install is friction)
- Instrument LangGraph: traces, tool calls, LLM calls, latency, cost per run
- Streamlit Analytics page (embed Langfuse dashboards)

**Milestone:** Every query visible in Langfuse with full trace and cost breakdown.

---

### Phase 9 — Caching + Rate Limiting `[F11]`

- Redis local setup
- GPTCache semantic cache on Redis (near-duplicate query deduplication)
- `slowapi` per-user rate limiting: requests/min + daily cost cap

**Milestone:** Near-duplicate queries served from cache; excess requests blocked.

---

### Phase 10 — Formal Evaluation Pipeline `[F9]`

- Construct 30 Q&A eval set using source-first method from `docs/testing-and-evaluation-strategy.md`
- Guardrail dataset (10–15 prompts), smoke-test dataset (4 representative items)
- Ragas runner: context precision, recall, faithfulness, answer relevancy, answer correctness
- A/B runner: variant A (reranker on) vs variant B (reranker off); same stable eval set for both
- Streamlit Evaluation page: per-variant Ragas scores

*Small eval artifacts begin earlier — smoke queries in Phase 0/1, answer-quality checks in Phase 2. This phase formalizes the full Ragas and A/B infrastructure.*

**Milestone:** Both variants evaluated; scores visible in Streamlit Evaluation page.

---

### Phase 11A — Advanced Retrieval Transforms `[F3 completion — retrieval track]`

- HyDE (hypothetical document embeddings) triggered on low-confidence
- Multi-query fusion + query decomposition for complex questions

**Milestone:** Low-confidence queries escalate correctly and return improved chunks.

---

### Phase 11B — Advanced Guardrails + Security `[F3 completion — security track]`

- Prompt injection detection in input guardrail
- Source whitelisting enforcement in retrieval
- Permission checks tied to user role

**Milestone:** Full guardrail dataset passes at 100% refusal/redaction rate.

---

### Phase 12 — Remaining UI Pages `[F10 completion]`

- History page, Saved Searches page
- Admin page (user role management, A/B controls, eval controls)
- Model picker for A/B testing in chat UI

**Milestone:** All planned Streamlit pages functional.

---

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| PDF parsing quality on German oncology layouts (tables, multi-column, footnotes) | HIGH | Parse one PDF first in Phase 0A; inspect raw output before committing to full pipeline |
| LLM enrichment cost/time for ~2,000 pages via Gemini Flash | HIGH | Run on one guideline first to estimate cost; make indexer restartable from a checkpoint |
| GPT-5 model ID on OpenRouter may not exist or behave differently than expected | HIGH | Verify exact model ID before writing generation code; have fallback ready (e.g. `openai/gpt-4o`) |
| Milvus native BM25 sparse schema and query API (newer feature) | MEDIUM | Test on small batch before committing full schema |
| LangGraph tool-calling loop reliability | MEDIUM | Build with one tool (`search_guidelines`) and test fully before adding remaining 5 |
| Dual-layer answer faithfulness — plain-language layer risks being a shallow rephrase | MEDIUM | Invest in prompt design in Phase 2; smoke-test both layers against the same source chunks |
| Langfuse local setup (Next.js app with its own Postgres) | MEDIUM | Use Langfuse Cloud if local install adds friction |
| Business logic drifting into FastAPI routes or Streamlit callbacks | MEDIUM | Enforce layering from Phase 3 onward: routes delegate, callbacks delegate |
| Bibliography/reference-link accuracy — in-text markers (`[1189]`) may fail to resolve to parsed bibliography entries due to PDF formatting inconsistencies | MEDIUM | Validate reference linking on one guideline before Phase 0C rollout; store unresolved markers with a flag rather than dropping them |

---

## Key Files

| File | Purpose |
|---|---|
| `docs/project_concept.md` | Locked stack and full pipeline design |
| `docs/parsing-and-chunking-strategy.md` | Chunking heuristics and fallback rules |
| `docs/testing-and-evaluation-strategy.md` | Eval set construction and Ragas metrics |
| `docs/guideline-chunk-metadata.schema.json` | Canonical chunk metadata schema |
| `docs/evaluation-dataset.schema.json` | Evaluation item schema |
| `docs/knowledge_base/` | 4 guideline PDFs |
| `config/sources.json` | PDF source URLs |
