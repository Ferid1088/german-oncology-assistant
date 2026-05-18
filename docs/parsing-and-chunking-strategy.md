# Parsing and Chunking Strategy for the Knowledge Base

## Goal

This document defines the recommended strategy for parsing, chunking, and metadata extraction for the German oncology guideline knowledge base used in the RAG system.

The main design objective is to preserve the semantic structure of the guidelines while producing retrieval-friendly chunks and reliable metadata for filtering, reranking, citations, and answer generation.

---

## Core Strategy

The best approach for this knowledge base is **structure-first hierarchical chunking with two-stage metadata extraction**.

This means:

1. Parse documents according to their real document structure, not only by token length.
2. Build a hierarchy of sections and recommendation blocks.
3. Create small retrieval chunks and larger parent context chunks.
4. Extract metadata in two stages:
   - deterministic structural extraction first
   - LLM-assisted semantic extraction second

This is more appropriate than naive fixed-size chunking because oncology guidelines are highly structured and meaning often depends on section boundaries, recommendation blocks, evidence statements, and patient subgroup context.

---

## 1. Parsing Strategy

### 1.1 PDF text extraction

Use a PDF parser such as **PyMuPDF** to extract text page by page.

The parser should preserve:

- page boundaries
- line order
- heading markers
- recommendation numbering
- local section structure

### 1.2 Text cleaning

After extraction, normalize the text before chunking.

Recommended cleaning steps:

- remove repeated headers and footers
- de-hyphenate words split across line breaks
- merge broken lines within paragraphs
- preserve section numbering such as `1`, `1.2`, `3.4.1`
- preserve formal recommendation labels such as `Empfehlung 4.3`
- remove or isolate footnotes where appropriate
- detect tables and figure-like regions when possible

### 1.3 Structural detection

Before any token-based splitting, detect the document hierarchy.

Important structural units:

- major sections
- subsections
- recommendation blocks
- evidence or rationale blocks
- summaries
- tables

This structural detection step is critical because recommendation and evidence content should not be cut apart arbitrarily.

### 1.4 Implementation heuristics

To reduce implementation drift, use explicit heuristics where possible.

Suggested heuristics:

- **Numbered headings**: detect lines beginning with patterns such as `^\d+(\.\d+)*\s+`.
- **Recommendation blocks**: detect markers like `Empfehlung`, optionally followed by hierarchical numbering.
- **Bibliography entries**: detect lines in the reference section using patterns such as `^\[?\d+\]?\s*[\.|:]` or equivalent normalized variants.
- **Merged paragraph lines**: join adjacent lines when the first line does not end with sentence-final punctuation and the next line does not look like a heading or list item.
- **Hyphenation repair**: only remove line-break hyphens when the surrounding tokens form a plausible German word.

These heuristics should be treated as defaults and refined after inspecting real extraction output from the PDFs.

---

## 2. Chunking Strategy

### 2.1 Use hierarchical chunking

The recommended design uses **two chunk levels**:

#### A. Leaf chunks for retrieval

These are the chunks stored and retrieved at the fine-grained level.

Recommended settings:

- target size: **400–700 tokens**
- overlap: **10–15% maximum**, only when needed for long prose sections
- each formal recommendation block should become its **own standalone chunk**, even when it is short

These chunks are optimized for relevance matching and reranking.

#### B. Parent chunks for answer context

These are larger chunks used after retrieval to provide broader context.

Recommended structure:

- usually a full section or subsection
- linked from each leaf chunk via `parent_chunk_id`
- optionally linked to a root ancestor via `root_chunk_id`

This supports the retrieval strategy:

> retrieve small, return parent context

which improves citation quality and answer faithfulness.

---

### 2.2 Chunk by semantic boundaries before size boundaries

Chunking should follow this priority order:

1. recommendation block
2. evidence or rationale block
3. subsection
4. paragraph group
5. token-length fallback split

This means token-based splitting should happen only when a structurally valid unit is too long.

### 2.3 Avoid naive fixed window chunking

Do **not** chunk the PDFs only by character count or token count without structural awareness.

That approach would:

- split recommendations from their grades
- separate evidence statements from the recommendation they support
- damage citations
- reduce reranker quality
- hurt answer faithfulness

### 2.4 Fallback and failure-handling rules

Parsing and chunking will not always succeed cleanly. Define fallback behavior explicitly.

Recommended fallback rules:

- If a section title is missing, inherit the nearest valid ancestor title and mark the chunk for review in logs.
- If a recommendation block spans pages awkwardly, preserve it as one semantic chunk even if page metadata spans multiple pages.
- If table extraction is unreliable, store the content as `chunk_type = table` or `note` with raw text preserved rather than forcing a misleading structure.
- If bibliography parsing fails for a cited reference, preserve the in-text `reference_ids` and flag the missing reference record for later repair.
- If semantic metadata extraction is uncertain, store empty arrays rather than hallucinated labels and rely on logging or confidence tracking.

---

## 3. Metadata Extraction Strategy

The recommended approach is **two-stage metadata extraction**.

### 3.1 Stage A: deterministic structural metadata

Use parser logic, regex, and document structure rules to extract highly reliable metadata.

Recommended deterministic fields:

- `chunk_id`
- `parent_chunk_id`
- `root_chunk_id`
- `guideline_id`
- `guideline_version`
- `guideline_title`
- `source_filename`
- `section_path`
- `section_title`
- `chunk_type`
- `chunk_index_in_parent`
- `recommendation_id`
- `recommendation_grade`
- `evidence_level` where formatting is consistent
- `page_start`
- `page_end`
- `is_leaf_chunk`
- `is_current`

These fields should be extracted reproducibly and should not depend on an LLM whenever avoidable.

Recommended deterministic validation checks outside the schema:

- `page_end >= page_start` when both are present
- `char_end >= char_start` when both are present
- `references_cited[].reference_id` should be represented in `reference_ids`
- `recommendation_grade` should normally appear only on recommendation-related chunks

### 3.2 Stage B: LLM-assisted semantic metadata

Use an LLM for semantic enrichment where rules alone are insufficient.

Recommended semantic fields for v1:

- `diseases`
- `drugs`
- `procedures`
- `patient_subgroups`
- `risk_category`

Recommended semantic fields for v2:

- `tumor_stage`
- `therapy_setting`
- `biomarkers`
- `clinical_intent`

This division keeps structural metadata stable while allowing richer semantic indexing over time.

---

## 4. Metadata Normalization Principles

Semantic metadata should be normalized as much as possible.

For example:

- use lowercase canonical labels where practical
- unify common synonyms
- avoid storing only surface-form mentions

Example normalization issue:

- `NSCLC`
- `nicht-kleinzelliges Lungenkarzinom`
- `nicht kleinzellig`

These should map to a consistent canonical value if they are intended for filtering or structured retrieval.

Recommended practice:

- store canonical values for filtering
- optionally keep raw mentions separately later if needed

---

## 5. Enrichment and Retrieval Metadata

Because the retrieval pipeline uses contextual enrichment, chunk metadata should also capture indexing-time enrichment state.

Important fields:

- `has_contextual_header`
- `has_hypothetical_questions`
- `enrichment_version`

For expanded lifecycle tracking, also consider:

- `parser_version`
- `chunking_version`
- `metadata_extraction_version`
- `indexed_at`
- `source_hash`

These fields are especially useful for debugging, evaluation, A/B testing, and re-indexing.

---

## 6. Reference Extraction and Citation Linking

The guideline PDFs can contain bibliography-style internal references where a chunk cites a study in-text and the full bibliographic entry appears elsewhere in the document.

Example pattern:

- body text contains a citation marker such as `[1189]`
- the references section later contains an entry such as `1189 . Han S, Woo S, ... URL: https://pubmed.ncbi.nlm.nih.gov/...`

This should be handled explicitly during parsing.

### 6.1 What should be extracted

Add a reference extraction pass that detects:

- in-text numeric citation markers such as `[1189]`
- multiple citations such as `[45, 46]`
- bibliography entries in the reference list
- external identifiers when available, such as PubMed URLs or PMIDs

### 6.2 How to represent references in chunk metadata

At the chunk level, extract and store:

- `reference_ids` for simple linkage
- optionally `references_cited` for richer mention-level details

Example:

```json
{
  "reference_ids": ["1189"],
  "references_cited": [
    {
      "reference_id": "1189",
      "mention_text": "[1189]"
    }
  ]
}
```

### 6.3 Recommended storage model

Use a two-layer model:

1. **chunk metadata** stores cited reference IDs
2. **a separate reference-entry store** holds full parsed bibliography records

This is cleaner than duplicating full study metadata in every chunk.

In the current documentation set, the chunk-side contract is defined in:

- `docs/guideline-chunk-metadata.schema.json`

A separate reference-entry schema is still recommended as a future addition.

### 6.4 Why this matters

Reference extraction improves:

- provenance and explainability
- evidence-chain tracing
- integration with PubMed-linked sources
- future evaluation of whether answers preserve cited evidence correctly

### 6.5 Parsing workflow addition

Add these steps to the indexing pipeline:

1. detect in-text citation markers in chunk text
2. parse the document reference list separately
3. create structured reference records
4. link chunk-level `reference_ids` to parsed bibliography entries

This is especially useful in oncology guidelines where recommendations may cite external studies that should remain traceable.

---

## 7. Recommended End-to-End Pipeline

The best practical pipeline for this knowledge base is:

1. parse PDF page-by-page
2. clean extracted text
3. detect headings, sections, and recommendation blocks
4. build parent section nodes
5. create leaf chunks from semantic units
6. attach deterministic structural metadata
7. run LLM-based semantic metadata extraction
8. generate contextual headers for retrieval
9. generate hypothetical questions for retrieval lift
10. embed the enriched text
11. store vectors, metadata, and hierarchy links in the vector store

---

## 8. Recommended Defaults for This Project

For this oncology guideline corpus, the recommended defaults are:

- **parent chunk granularity:** full subsection
- **leaf chunk size:** 400–700 tokens
- **overlap:** 10–15% maximum when necessary
- **special rule:** every formal `Empfehlung X.Y` block becomes its own chunk
- **metadata extraction:** deterministic first, LLM second
- **initial semantic metadata scope:** diseases, drugs, procedures, patient subgroups, risk categories
- **later semantic expansion:** tumor stage, therapy setting, biomarkers, clinical intent

---

## 9. What to Avoid

Avoid the following mistakes:

- naive fixed-size chunking with no structural awareness
- large overlaps that create redundant retrieval noise
- storing unstable experimental analytics directly inside chunk metadata
- implementing chunk-level supersession logic before building a reliable version-diff process
- overemphasizing dosage and contraindication extraction in v1 for a system that is not meant to give dosing advice

---

## Final Recommendation

The best strategy for this project is:

> **Use structure-first hierarchical chunking around sections, recommendation blocks, and evidence units, then extract metadata in two stages: deterministic structural metadata first, and LLM-assisted semantic metadata second.**

This gives the best balance of:

- retrieval precision
- answer faithfulness
- clean citations
- metadata-based filtering
- future extensibility

for a German oncology guideline RAG system.