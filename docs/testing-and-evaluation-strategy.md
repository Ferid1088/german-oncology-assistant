# Testing and Evaluation Strategy

## Goal

This document defines the recommended testing and evaluation strategy for the German oncology guideline RAG app described in `docs/project_concept.md`.

The purpose is to test not only whether the app answers correctly, but also whether it:

- retrieves the right guideline content
- cites evidence faithfully
- handles comparison and recommendation queries correctly
- applies guardrails reliably
- supports A/B evaluation of retrieval strategies

---

## 1. What needs to be tested

According to the project concept, the app should be evaluated across several layers.

### 1.1 Retrieval quality

The system must retrieve the correct guideline chunks, sections, and pages.

This includes testing:

- dense + BM25 hybrid retrieval
- metadata filtering
- reranking quality
- parent-section expansion

### 1.2 Answer quality

The system must produce:

- a professional answer in formal German
- a plain-language explanation
- inline citations grounded in retrieved content
- a medical disclaimer

This includes testing:

- answer correctness
- answer relevancy
- faithfulness to the retrieved evidence
- usefulness of the plain-language layer

### 1.3 Tool and routing behavior

The app uses intent routing and tool calls.

This means evaluation should include questions that trigger:

- `search_guidelines`
- `lookup_empfehlung`
- `compare_guidelines`
- `drug_class_lookup`
- `pubmed_search`

### 1.4 Guardrails and refusal behavior

The project concept also requires testing:

- off-topic refusal
- prompt-injection detection
- PII redaction
- refusal of dosage or patient-specific treatment advice

### 1.5 A/B comparison

The concept defines two retrieval variants:

- **A** = hybrid retrieval + reranker
- **B** = hybrid retrieval without reranker

The same stable evaluation set should be reusable for both variants.

---

## 2. Datasets you need

The recommended evaluation setup uses multiple datasets rather than a single Q&A list.

### 2.1 Main evaluation dataset

This is the core benchmark set of approximately **30 hand-curated German Q&A items** derived from the guideline corpus.

Each item should contain:

- question
- reference answer
- gold chunk IDs
- gold sections
- gold pages
- question type
- difficulty
- guideline scope

This dataset is used for:

- Ragas evaluation
- regression testing
- A/B comparison

### 2.2 Retrieval relevance dataset

This can be stored together with the main evaluation items or separately.

Its purpose is to define which chunks, sections, and pages count as relevant evidence for each question.

This is required for measuring:

- context precision
- context recall
- citation quality

### 2.3 Guardrail dataset

This dataset contains unsafe or unsupported prompts such as:

- prompt injection attempts
- off-topic questions
- dosage requests
- patient-specific treatment requests
- PII-containing inputs

This dataset is used to verify refusal and redaction behavior.

### 2.4 Smoke-test dataset

This is a very small set of representative prompts used during development for quick checks.

It should include:

- one factual question
- one recommendation question
- one comparison question
- one guardrail prompt

---

## 3. Best way to extract the evaluation set

The best approach is **source-first dataset construction**.

### Step 1: identify high-value source chunks

Start from the parsed/chunked corpus and select source chunks such as:

- recommendation chunks
- evidence chunks
- rationale chunks
- chunks with explicit patient subgroup language
- chunks useful for cross-guideline comparison

This is the best starting point because every evaluation item should have traceable gold evidence.

### Step 2: generate candidate questions

For each selected source chunk, create 2–4 candidate questions.

This can be done:

- manually
- or with LLM assistance followed by human review

Question categories should include:

- factual
- recommendation
- evidence-oriented
- comparison
- drug/entity lookup
- external-literature trigger cases

### Step 3: manually curate and label items

Each candidate question should be manually checked before it enters the final dataset.

Manual review should confirm:

- the answer is grounded in the selected chunk(s)
- the gold citations are correct
- the question type is correct
- the difficulty label is reasonable
- the question is realistic and useful

---

## 4. Recommended composition of the first benchmark set

For an initial set of **30 evaluation items**, the following distribution is recommended:

- **10 recommendation questions**
- **6 factual questions**
- **5 evidence/rationale questions**
- **4 comparison questions**
- **3 drug/entity questions**
- **2 PubMed-trigger questions**

Separately, create **10–15 guardrail prompts**.

This gives good coverage of the pipeline described in the project concept.

---

## 5. Recommended data model

Use the JSON schema in:

- `docs/evaluation-dataset.schema.json`

This schema supports:

- evaluation items
- A/B testing items
- guardrail items
- smoke-test items

Important fields include:

- `question`
- `question_type`
- `difficulty`
- `guideline_scope`
- `source_chunk_ids`
- `expected_answer`
- `expected_answer_plain_language`
- `gold_chunk_ids`
- `gold_sections`
- `gold_pages`
- `expected_filters`
- `requires_comparison`
- `needs_pubmed`
- `should_refuse`
- `should_redact_pii`

---

## 6. Evaluation metrics

The project concept already identifies **Ragas** as the main evaluation framework.

Recommended metrics:

- **context precision**
- **context recall**
- **faithfulness**
- **answer relevancy**
- **answer correctness**

In addition, for app-level testing, manually inspect:

- citation formatting
- disclaimer presence
- quality of the plain-language layer
- whether the correct tool was invoked
- whether guardrails fired correctly

### 6.1 Automatic vs manual evaluation

Separate evaluation modes clearly:

- **Automatic evaluation**: Ragas metrics, retrieval overlap with gold chunks, latency comparisons, refusal/redaction rates.
- **Manual evaluation**: expert review of answer usefulness, citation appropriateness, plain-language quality, and overall medical phrasing.
- **Smoke/regression checks**: a very small stable subset run frequently during development.

This separation helps avoid mixing exploratory review with regression testing.

### 6.2 Provisional pass/fail targets

At the beginning, use provisional thresholds rather than aiming for perfect scores.

Suggested initial targets:

- **Faithfulness**: no obvious contradiction with retrieved evidence in reviewed samples
- **Citation presence**: 100% of answerable in-domain responses should include at least one source citation
- **Guardrail behavior**: 100% refusal or safe redirection for clearly unsafe patient-specific dosage/treatment requests in the guardrail set
- **PII behavior**: all test prompts containing explicit PII should trigger redaction before downstream processing
- **A/B comparison**: variant changes should be evaluated on both quality and latency, not only one dimension

These are starting thresholds and should be tightened after you collect real baseline runs.

---

## 7. A/B testing strategy

Use the same stable benchmark set for both retrieval variants:

- **Variant A**: hybrid retrieval + reranker
- **Variant B**: hybrid retrieval without reranker

Store a variant tag per run and compare:

- Ragas metrics
- retrieval quality
- latency
- citation quality

Do not create separate question sets for A and B initially. The same benchmark is more useful for fair comparison.

---

## 8. Guardrail testing guidance

Guardrail prompts should test:

- prompt injection attempts
- off-topic requests
- patient-specific treatment advice
- dosage questions
- PII in the prompt

Each guardrail item should define whether the app is expected to:

- refuse
- redact
- warn
- continue with a safe alternative

---

## 9. Recommended workflow

The recommended workflow for building the evaluation dataset is:

1. parse and chunk the guideline corpus
2. filter high-value chunks by metadata (`chunk_type`, guideline, recommendation grade)
3. sample source chunks from each guideline
4. draft candidate questions and reference answers
5. manually validate and label each item
6. save the dataset in structured JSON following `evaluation-dataset.schema.json`
7. run baseline evaluation
8. reuse the same dataset for regression testing and A/B comparison

### 9.1 Annotation protocol

Even if you are the only annotator, define a simple annotation protocol.

For each item:

1. confirm the gold evidence really supports the intended answer
2. verify the retrieval labels are neither too narrow nor too broad
3. assign a question type and difficulty label
4. note any ambiguity in `notes`
5. version the dataset whenever items are added, removed, or relabeled

This will make later evaluation runs more trustworthy.

### 9.2 Suggested file-level organization

Recommended future organization:

- `data/eval/evaluation-dataset.json`
- `data/eval/guardrail-dataset.json`
- `data/eval/smoke-dataset.json`
- `docs/evaluation-dataset.schema.json`

Keeping the schema in `docs/` and the actual datasets in `data/` helps separate documentation from artifacts.

---

## Final Recommendation

The most important rule is:

> **Build the evaluation set from known source chunks first, then derive questions from them, and only then finalize gold answers and retrieval labels.**

This produces a stronger benchmark than writing free-form questions without traceable evidence, and it aligns well with the retrieval, citation, and faithfulness goals of the project.