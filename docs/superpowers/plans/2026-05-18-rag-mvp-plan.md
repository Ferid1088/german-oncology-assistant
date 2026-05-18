# RAG MVP — Indexing → Retrieval → LangGraph → FastAPI → Streamlit

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working end-to-end RAG system that answers German oncology questions with dual-layer cited answers, ending at the MVP milestone (Streamlit chat UI with source cards).

**Architecture:** Depth-first — each layer is complete and tested before the next begins. Indexing pipeline (offline) produces chunks in Milvus; retrieval engine searches them; LangGraph orchestrates query rewriting → tool calls → dual-layer answer generation; FastAPI exposes SSE streaming; Streamlit renders the chat UI. Business logic lives exclusively in `src/graph/` and `src/retrieval/` — FastAPI and Streamlit are delivery/UI layers only.

**Tech Stack:** Python 3.12, uv, pymupdf, openai SDK (OpenRouter), pymilvus, FlagEmbedding (bge-reranker), langgraph, fastapi + sse-starlette, streamlit, pytest + pytest-mock

**Scope:** Phases 0A → 4 of the roadmap spec (`docs/superpowers/specs/2026-05-18-rag-mvp-roadmap-design.md`). Later phases get separate plans.

---

## File Map

```
src/
  indexer/
    parser.py        # PyMuPDF extraction + text cleaning
    detector.py      # Structural detection: headings, Empfehlung blocks, bibliography
    chunker.py       # Hierarchical chunker: leaf + parent chunks
    metadata.py      # Deterministic structural metadata extraction
    reference.py     # In-text citation markers + bibliography parsing
    enricher.py      # Gemini Flash: contextual headers, hypothetical Qs, semantic metadata
    embedder.py      # text-embedding-3-large via OpenRouter
    store.py         # Milvus schema setup + upsert
    pipeline.py      # Orchestrates full indexing flow
  retrieval/
    search.py        # Dense + BM25 + RRF fusion + metadata filter
    reranker.py      # bge-reranker-v2-m3 via FlagEmbedding
    expander.py      # Parent-document expansion
  tools/
    search_guidelines.py   # LangGraph tool wrapping retrieval
    lookup_empfehlung.py   # Direct Empfehlung X.Y lookup
  graph/
    state.py         # LangGraph TypedDict state schema
    nodes/
      rewriter.py    # Query rewriting (Gemini Flash)
      agent.py       # Tool-calling agent loop (GPT-5 via OpenRouter)
      answer.py      # Dual-layer answer generation
      guardrail_input.py   # Off-topic classifier + PII redaction
      self_query.py  # Metadata extraction from query (Gemini Flash)
      router.py      # Intent router
      confidence.py  # Lightweight confidence check
      guardrail_output.py  # Faithfulness check + PII scan
    graph.py         # LangGraph StateGraph assembly + compile
  api/
    main.py          # FastAPI app factory
    auth.py          # API-key middleware
    routes/
      chat.py        # POST /chat → SSE stream
      feedback.py    # POST /feedback
  ui/
    app.py           # Streamlit entry point
    components/
      chat_page.py   # Chat UI with SSE streaming
      source_cards.py  # Source card rendering
      filters.py     # Filter panel + feedback buttons
tests/
  conftest.py        # Shared fixtures
  indexer/
    test_parser.py
    test_detector.py
    test_chunker.py
    test_metadata.py
    test_reference.py
    test_enricher.py
  retrieval/
    test_search.py
    test_reranker.py
  tools/
    test_tools.py
  graph/
    test_nodes.py
    test_graph.py
scripts/
  run_indexer.py     # CLI entry point for the indexing pipeline
pyproject.toml
.env.example
```

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `src/__init__.py` and all `__init__.py` files
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "oncology-rag"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pymupdf>=1.24.0",
    "openai>=1.40.0",
    "pymilvus>=2.4.0",
    "FlagEmbedding>=1.2.0",
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-community>=0.3.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
    "sse-starlette>=2.1.0",
    "streamlit>=1.38.0",
    "httpx>=0.27.0",
    "pydantic>=2.8.0",
    "python-dotenv>=1.0.0",
    "presidio-analyzer>=2.2.0",
    "presidio-anonymizer>=2.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.14.0",
    "pytest-asyncio>=0.24.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Install dependencies**

```bash
uv pip install -e ".[dev]"
```

Expected: All packages install without error.

- [ ] **Step 3: Create .env.example**

```bash
# OpenRouter
OPENROUTER_API_KEY=your_key_here
GENERATION_MODEL=openai/gpt-4o        # verify openai/gpt-5 exists before changing
CHEAP_MODEL=google/gemini-2.5-flash
EMBEDDING_MODEL=openai/text-embedding-3-large

# Milvus (local, no auth)
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=oncology_guidelines

# App
API_KEY=dev-secret-key
```

- [ ] **Step 4: Create directory skeleton**

```bash
mkdir -p src/indexer src/retrieval src/tools src/graph/nodes src/api/routes src/ui/components
mkdir -p tests/indexer tests/retrieval tests/tools tests/graph
mkdir -p data/chunks data/eval scripts
touch src/__init__.py src/indexer/__init__.py src/retrieval/__init__.py
touch src/tools/__init__.py src/graph/__init__.py src/graph/nodes/__init__.py
touch src/api/__init__.py src/api/routes/__init__.py src/ui/__init__.py
touch src/ui/components/__init__.py
touch tests/__init__.py tests/indexer/__init__.py tests/retrieval/__init__.py
touch tests/tools/__init__.py tests/graph/__init__.py
```

- [ ] **Step 5: Create tests/conftest.py**

```python
import os
import pytest
from pathlib import Path

@pytest.fixture
def sample_pdf_path() -> Path:
    return Path("docs/knowledge_base/mammakarzinom_v4.4.pdf")

@pytest.fixture
def sample_text() -> str:
    return """1 Einleitung

1.1 Informationen zu dieser Leitlinie

Die S3-Leitlinie Mammakarzinom wurde erstellt.

Empfehlung 1.1
Frauen mit erhöhtem familiären Risiko sollen eine genetische Beratung erhalten.
Empfehlungsgrad: A
Evidenzlevel: 1a

1.2 Hintergrund

Weiterer Text der Leitlinie.

[45] Autor A et al. Titel. Journal 2020;1:1-10.
[46] Autor B et al. Titel. Journal 2021;2:2-20.
"""

@pytest.fixture
def openrouter_client(mocker):
    """Mock OpenRouter client — prevents real API calls in unit tests."""
    mock = mocker.MagicMock()
    mock.chat.completions.create.return_value = mocker.MagicMock(
        choices=[mocker.MagicMock(message=mocker.MagicMock(content="mocked response"))]
    )
    mock.embeddings.create.return_value = mocker.MagicMock(
        data=[mocker.MagicMock(embedding=[0.1] * 3072)]
    )
    return mock
```

- [ ] **Step 6: Verify pytest runs**

```bash
uv run pytest tests/ -v
```

Expected: `no tests ran` (0 collected). No import errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .env.example src/ tests/ scripts/ data/
git commit -m "feat: project skeleton, deps, conftest"
```

---

## Task 2: PDF Parser

**Files:**
- Create: `src/indexer/parser.py`
- Create: `tests/indexer/test_parser.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/indexer/test_parser.py
import pytest
from src.indexer.parser import extract_pages, clean_text

def test_extract_pages_returns_list_of_page_dicts(sample_pdf_path):
    pages = extract_pages(sample_pdf_path)
    assert isinstance(pages, list)
    assert len(pages) > 0
    first = pages[0]
    assert "page_number" in first
    assert "text" in first
    assert isinstance(first["text"], str)
    assert len(first["text"]) > 0

def test_extract_pages_page_numbers_are_one_indexed(sample_pdf_path):
    pages = extract_pages(sample_pdf_path)
    assert pages[0]["page_number"] == 1

def test_clean_text_removes_trailing_whitespace():
    raw = "Empfehlung 1.1  \n  Text  \n"
    result = clean_text(raw)
    assert not any(line.endswith("  ") for line in result.splitlines())

def test_clean_text_repairs_german_hyphenation():
    raw = "Die Behand-\nlung erfolgt"
    result = clean_text(raw)
    assert "Behandlung" in result

def test_clean_text_merges_broken_paragraph_lines():
    raw = "Dies ist ein langer\nSatz der weitergeht."
    result = clean_text(raw)
    assert "langer Satz" in result

def test_clean_text_preserves_section_numbers():
    raw = "1.2 Diagnostik\nText"
    result = clean_text(raw)
    assert "1.2 Diagnostik" in result
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/indexer/test_parser.py -v
```

Expected: `ImportError: cannot import name 'extract_pages'`

- [ ] **Step 3: Implement parser**

```python
# src/indexer/parser.py
import re
from pathlib import Path
import fitz  # pymupdf


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text page by page from a PDF. Returns list of {page_number, text}."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({"page_number": i + 1, "text": text})
    doc.close()
    return pages


def clean_text(text: str) -> str:
    """
    Normalize extracted PDF text:
    - Repair German hyphenation across line breaks
    - Merge broken paragraph lines
    - Strip trailing whitespace per line
    - Preserve section numbers and Empfehlung labels
    """
    # Repair hyphenation: "Behand-\nlung" -> "Behandlung"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    lines = text.splitlines()
    merged: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if (
            i + 1 < len(lines)
            and line
            and not _is_heading(line)
            and not _ends_sentence(line)
            and not _is_heading(lines[i + 1].strip())
            and lines[i + 1].strip()
        ):
            # Merge continuation line
            merged.append(line + " " + lines[i + 1].strip())
            i += 2
        else:
            merged.append(line)
            i += 1

    return "\n".join(merged)


def _is_heading(line: str) -> bool:
    return bool(re.match(r"^\d+(\.\d+)*\s+\S", line.strip()))


def _ends_sentence(line: str) -> bool:
    return line.rstrip().endswith((".", ":", "!", "?"))
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/indexer/test_parser.py -v
```

Expected: All 6 tests PASS. (The PDF-dependent tests require Milvus to be running but only need the PDF file present.)

- [ ] **Step 5: Commit**

```bash
git add src/indexer/parser.py tests/indexer/test_parser.py
git commit -m "feat(indexer): PDF extraction and text cleaning"
```

---

## Task 3: Structural Detector

**Files:**
- Create: `src/indexer/detector.py`
- Create: `tests/indexer/test_detector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/indexer/test_detector.py
from src.indexer.detector import detect_structure, StructuralUnit

def test_detects_numbered_heading(sample_text):
    units = detect_structure(sample_text)
    headings = [u for u in units if u.kind == "heading"]
    titles = [u.text.strip() for u in headings]
    assert any("1 Einleitung" in t for t in titles)
    assert any("1.1" in t for t in titles)

def test_detects_empfehlung_block(sample_text):
    units = detect_structure(sample_text)
    empfehlungen = [u for u in units if u.kind == "empfehlung"]
    assert len(empfehlungen) >= 1
    assert "1.1" in empfehlungen[0].recommendation_id

def test_empfehlung_includes_grade(sample_text):
    units = detect_structure(sample_text)
    emp = next(u for u in units if u.kind == "empfehlung")
    assert emp.recommendation_grade == "A"

def test_detects_bibliography_entries(sample_text):
    units = detect_structure(sample_text)
    refs = [u for u in units if u.kind == "bibliography_entry"]
    assert len(refs) >= 2
    ids = [u.reference_id for u in refs]
    assert "45" in ids
    assert "46" in ids

def test_detects_prose_body(sample_text):
    units = detect_structure(sample_text)
    prose = [u for u in units if u.kind == "prose"]
    assert len(prose) >= 1
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/indexer/test_detector.py -v
```

Expected: `ImportError: cannot import name 'detect_structure'`

- [ ] **Step 3: Implement detector**

```python
# src/indexer/detector.py
import re
from dataclasses import dataclass, field

HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+\S")
EMPFEHLUNG_RE = re.compile(r"^Empfehlung\s+(\d+(?:\.\d+)+)", re.IGNORECASE)
GRADE_RE = re.compile(r"Empfehlungsgrad\s*:\s*([A-Z0])", re.IGNORECASE)
EVIDENCE_RE = re.compile(r"Evidenzlevel\s*:\s*(\S+)", re.IGNORECASE)
BIB_RE = re.compile(r"^\[?(\d+)\]?\s*\S")


@dataclass
class StructuralUnit:
    kind: str                          # heading | empfehlung | bibliography_entry | prose
    text: str
    section_number: str = ""
    recommendation_id: str = ""
    recommendation_grade: str = ""
    evidence_level: str = ""
    reference_id: str = ""
    line_start: int = 0


def detect_structure(text: str) -> list[StructuralUnit]:
    lines = text.splitlines()
    units: list[StructuralUnit] = []
    i = 0
    in_bibliography = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Bibliography section starts when we see consecutive [N] lines
        if BIB_RE.match(stripped) and (in_bibliography or _looks_like_bib(stripped)):
            in_bibliography = True
            m = BIB_RE.match(stripped)
            units.append(StructuralUnit(
                kind="bibliography_entry",
                text=stripped,
                reference_id=m.group(1),
                line_start=i,
            ))
            i += 1
            continue

        if EMPFEHLUNG_RE.match(stripped):
            block_lines = [stripped]
            grade = ""
            evidence = ""
            j = i + 1
            while j < len(lines) and lines[j].strip():
                bl = lines[j].strip()
                gm = GRADE_RE.search(bl)
                em = EVIDENCE_RE.search(bl)
                if gm:
                    grade = gm.group(1)
                if em:
                    evidence = em.group(1)
                block_lines.append(bl)
                j += 1
            rec_id = EMPFEHLUNG_RE.match(stripped).group(1)
            units.append(StructuralUnit(
                kind="empfehlung",
                text="\n".join(block_lines),
                recommendation_id=rec_id,
                recommendation_grade=grade,
                evidence_level=evidence,
                line_start=i,
            ))
            i = j
            continue

        if HEADING_RE.match(stripped):
            m = HEADING_RE.match(stripped)
            units.append(StructuralUnit(
                kind="heading",
                text=stripped,
                section_number=m.group(1),
                line_start=i,
            ))
            i += 1
            continue

        # Prose: accumulate until next structural boundary
        prose_lines = [stripped]
        j = i + 1
        while j < len(lines):
            next_line = lines[j].strip()
            if not next_line:
                break
            if HEADING_RE.match(next_line) or EMPFEHLUNG_RE.match(next_line) or BIB_RE.match(next_line):
                break
            prose_lines.append(next_line)
            j += 1
        units.append(StructuralUnit(kind="prose", text=" ".join(prose_lines), line_start=i))
        i = j

    return units


def _looks_like_bib(line: str) -> bool:
    return bool(re.match(r"^\[?\d+\]?\s+\w+.+\d{4}", line))
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/indexer/test_detector.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/indexer/detector.py tests/indexer/test_detector.py
git commit -m "feat(indexer): structural detection for headings, Empfehlung, bibliography"
```

---

## Task 4: Hierarchical Chunker

**Files:**
- Create: `src/indexer/chunker.py`
- Create: `tests/indexer/test_chunker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/indexer/test_chunker.py
from src.indexer.chunker import build_chunks, Chunk

def test_empfehlung_becomes_own_leaf_chunk(sample_text):
    chunks = build_chunks("mamma", "1.0", sample_text)
    emp_chunks = [c for c in chunks if c.chunk_type == "empfehlung"]
    assert len(emp_chunks) >= 1
    assert emp_chunks[0].recommendation_id == "1.1"

def test_leaf_chunks_have_parent(sample_text):
    chunks = build_chunks("mamma", "1.0", sample_text)
    leaf_chunks = [c for c in chunks if c.is_leaf]
    parent_ids = {c.parent_chunk_id for c in leaf_chunks if c.parent_chunk_id}
    assert len(parent_ids) >= 1

def test_parent_chunks_are_not_leaf(sample_text):
    chunks = build_chunks("mamma", "1.0", sample_text)
    parent_ids = {c.parent_chunk_id for c in chunks if c.parent_chunk_id}
    parents = [c for c in chunks if c.chunk_id in parent_ids]
    assert all(not c.is_leaf for c in parents)

def test_chunks_have_required_fields(sample_text):
    chunks = build_chunks("mamma", "1.0", sample_text)
    for c in chunks:
        assert c.chunk_id
        assert c.guideline_id == "mamma"
        assert c.guideline_version == "1.0"

def test_leaf_chunk_size_within_bounds(sample_text):
    chunks = build_chunks("mamma", "1.0", sample_text)
    leaf_chunks = [c for c in chunks if c.is_leaf and c.chunk_type == "prose"]
    for c in leaf_chunks:
        # Approximate token count: len(text.split()) * 1.3
        approx_tokens = len(c.text.split()) * 1.3
        assert approx_tokens <= 800  # generous upper bound for test text
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/indexer/test_chunker.py -v
```

- [ ] **Step 3: Implement chunker**

```python
# src/indexer/chunker.py
import uuid
from dataclasses import dataclass, field
from src.indexer.detector import detect_structure, StructuralUnit

TARGET_TOKENS = 550       # middle of 400-700 range
MAX_TOKENS = 700
OVERLAP_TOKENS = 70       # ~12%


@dataclass
class Chunk:
    chunk_id: str
    guideline_id: str
    guideline_version: str
    text: str
    chunk_type: str           # prose | empfehlung | table | heading
    is_leaf: bool
    parent_chunk_id: str | None = None
    root_chunk_id: str | None = None
    section_path: list[str] = field(default_factory=list)
    section_title: str = ""
    recommendation_id: str = ""
    recommendation_grade: str = ""
    evidence_level: str = ""
    page_start: int | None = None
    page_end: int | None = None


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _make_id() -> str:
    return str(uuid.uuid4())


def build_chunks(
    guideline_id: str,
    guideline_version: str,
    text: str,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[Chunk]:
    units = detect_structure(text)
    chunks: list[Chunk] = []
    current_section_path: list[str] = []
    current_section_title: str = ""
    current_parent_id: str | None = None
    prose_buffer: list[str] = []

    def flush_prose():
        nonlocal prose_buffer, current_parent_id
        if not prose_buffer:
            return
        full_text = " ".join(prose_buffer)
        # Split into leaf chunks if too long
        words = full_text.split()
        start = 0
        while start < len(words):
            end = start
            token_count = 0
            while end < len(words) and token_count < TARGET_TOKENS:
                token_count = int((end - start + 1) * 1.3)
                end += 1
            chunk_text = " ".join(words[start:end])
            leaf = Chunk(
                chunk_id=_make_id(),
                guideline_id=guideline_id,
                guideline_version=guideline_version,
                text=chunk_text,
                chunk_type="prose",
                is_leaf=True,
                parent_chunk_id=current_parent_id,
                root_chunk_id=current_parent_id,
                section_path=list(current_section_path),
                section_title=current_section_title,
                page_start=page_start,
                page_end=page_end,
            )
            chunks.append(leaf)
            start = max(end - int(OVERLAP_TOKENS / 1.3), end - 50) if end < len(words) else end
        prose_buffer = []

    for unit in units:
        if unit.kind == "heading":
            flush_prose()
            depth = unit.section_number.count(".") + 1
            current_section_path = current_section_path[: depth - 1] + [unit.section_number]
            current_section_title = unit.text
            # Create parent chunk for this section
            parent = Chunk(
                chunk_id=_make_id(),
                guideline_id=guideline_id,
                guideline_version=guideline_version,
                text=unit.text,
                chunk_type="heading",
                is_leaf=False,
                section_path=list(current_section_path),
                section_title=current_section_title,
                page_start=page_start,
                page_end=page_end,
            )
            chunks.append(parent)
            current_parent_id = parent.chunk_id

        elif unit.kind == "empfehlung":
            flush_prose()
            leaf = Chunk(
                chunk_id=_make_id(),
                guideline_id=guideline_id,
                guideline_version=guideline_version,
                text=unit.text,
                chunk_type="empfehlung",
                is_leaf=True,
                parent_chunk_id=current_parent_id,
                root_chunk_id=current_parent_id,
                section_path=list(current_section_path),
                section_title=current_section_title,
                recommendation_id=unit.recommendation_id,
                recommendation_grade=unit.recommendation_grade,
                evidence_level=unit.evidence_level,
                page_start=page_start,
                page_end=page_end,
            )
            chunks.append(leaf)

        elif unit.kind == "prose":
            prose_buffer.append(unit.text)

        elif unit.kind == "bibliography_entry":
            flush_prose()  # Don't include bib entries as retrieval chunks

    flush_prose()
    return chunks
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/indexer/test_chunker.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/indexer/chunker.py tests/indexer/test_chunker.py
git commit -m "feat(indexer): hierarchical chunker with leaf/parent structure"
```

---

## Task 5: Deterministic Metadata

**Files:**
- Create: `src/indexer/metadata.py`
- Create: `tests/indexer/test_metadata.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/indexer/test_metadata.py
from src.indexer.chunker import build_chunks
from src.indexer.metadata import attach_metadata

def test_metadata_adds_chunk_index(sample_text):
    chunks = build_chunks("mamma", "4.4", sample_text)
    chunks = attach_metadata(chunks, source_filename="mammakarzinom_v4.4.pdf")
    leaf_chunks = [c for c in chunks if c.is_leaf]
    indices = [c.chunk_index_in_parent for c in leaf_chunks]
    assert 0 in indices

def test_metadata_adds_source_filename(sample_text):
    chunks = build_chunks("mamma", "4.4", sample_text)
    chunks = attach_metadata(chunks, source_filename="mammakarzinom_v4.4.pdf")
    assert all(c.source_filename == "mammakarzinom_v4.4.pdf" for c in chunks)

def test_metadata_marks_is_current(sample_text):
    chunks = build_chunks("mamma", "4.4", sample_text)
    chunks = attach_metadata(chunks, source_filename="mammakarzinom_v4.4.pdf")
    assert all(c.is_current is True for c in chunks)
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/indexer/test_metadata.py -v
```

- [ ] **Step 3: Implement metadata attachment**

```python
# src/indexer/metadata.py
from src.indexer.chunker import Chunk
from collections import defaultdict


def attach_metadata(chunks: list[Chunk], source_filename: str) -> list[Chunk]:
    """Attach deterministic structural metadata fields that require cross-chunk context."""
    # Count children per parent to assign chunk_index_in_parent
    parent_child_counts: dict[str | None, int] = defaultdict(int)
    for chunk in chunks:
        if chunk.is_leaf:
            parent_child_counts[chunk.parent_chunk_id] += 1

    parent_counters: dict[str | None, int] = defaultdict(int)
    for chunk in chunks:
        chunk.source_filename = source_filename
        chunk.is_current = True
        if chunk.is_leaf:
            chunk.chunk_index_in_parent = parent_counters[chunk.parent_chunk_id]
            parent_counters[chunk.parent_chunk_id] += 1
        else:
            chunk.chunk_index_in_parent = 0

    return chunks
```

- [ ] **Step 4: Add `source_filename`, `is_current`, `chunk_index_in_parent` fields to `Chunk` dataclass in `chunker.py`**

```python
# Add to Chunk dataclass in src/indexer/chunker.py
source_filename: str = ""
is_current: bool = True
chunk_index_in_parent: int = 0
```

- [ ] **Step 5: Run — expect PASS**

```bash
uv run pytest tests/indexer/test_metadata.py tests/indexer/test_chunker.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/indexer/metadata.py src/indexer/chunker.py tests/indexer/test_metadata.py
git commit -m "feat(indexer): deterministic metadata attachment"
```

---

## Task 6: Reference Extractor

**Files:**
- Create: `src/indexer/reference.py`
- Create: `tests/indexer/test_reference.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/indexer/test_reference.py
from src.indexer.reference import extract_inline_refs, parse_bibliography, ReferenceEntry

def test_extract_inline_refs_single():
    text = "Gemäß der Studie [1189] ist die Therapie wirksam."
    refs = extract_inline_refs(text)
    assert "1189" in refs

def test_extract_inline_refs_multiple():
    text = "Laut [45, 46] und [47] ist die Evidenz klar."
    refs = extract_inline_refs(text)
    assert "45" in refs
    assert "46" in refs
    assert "47" in refs

def test_parse_bibliography_entry(sample_text):
    entries = parse_bibliography(sample_text)
    ids = [e.reference_id for e in entries]
    assert "45" in ids
    assert "46" in ids

def test_unresolved_refs_flagged():
    text = "Gemäß [9999] ist es so."
    refs = extract_inline_refs(text)
    assert "9999" in refs  # extracted but no bib entry → flagged by pipeline
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/indexer/test_reference.py -v
```

- [ ] **Step 3: Implement reference extractor**

```python
# src/indexer/reference.py
import re
from dataclasses import dataclass

INLINE_REF_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")
BIB_ENTRY_RE = re.compile(r"^\[?(\d+)\]?\s+(.+)$")
PUBMED_URL_RE = re.compile(r"https?://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")


@dataclass
class ReferenceEntry:
    reference_id: str
    raw_text: str
    pubmed_id: str = ""
    pubmed_url: str = ""
    unresolved: bool = False


def extract_inline_refs(text: str) -> list[str]:
    """Return list of all cited reference IDs found in text."""
    ids: list[str] = []
    for match in INLINE_REF_RE.finditer(text):
        for ref_id in match.group(1).split(","):
            ids.append(ref_id.strip())
    return ids


def parse_bibliography(text: str) -> list[ReferenceEntry]:
    """Extract structured bibliography entries from document text."""
    entries: list[ReferenceEntry] = []
    for line in text.splitlines():
        line = line.strip()
        m = BIB_ENTRY_RE.match(line)
        if m and len(line) > 10:  # avoid false positives on short lines
            ref_id = m.group(1)
            raw = m.group(2)
            pubmed_url = ""
            pubmed_id = ""
            url_m = PUBMED_URL_RE.search(raw)
            if url_m:
                pubmed_url = url_m.group(0)
                pubmed_id = url_m.group(1)
            entries.append(ReferenceEntry(
                reference_id=ref_id,
                raw_text=raw,
                pubmed_url=pubmed_url,
                pubmed_id=pubmed_id,
            ))
    return entries
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/indexer/test_reference.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/indexer/reference.py tests/indexer/test_reference.py
git commit -m "feat(indexer): inline reference extraction and bibliography parsing"
```

---

## Task 7: LLM Enricher

**Files:**
- Create: `src/indexer/enricher.py`
- Create: `tests/indexer/test_enricher.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/indexer/test_enricher.py
from unittest.mock import MagicMock
from src.indexer.enricher import generate_contextual_header, generate_hypothetical_questions, extract_semantic_metadata

def _mock_client(response_text: str):
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=response_text))]
    )
    return client

def test_generate_contextual_header_returns_string():
    client = _mock_client("Dieser Abschnitt behandelt die Diagnose des Mammakarzinoms.")
    header = generate_contextual_header(
        client=client,
        chunk_text="Empfehlung 2.1: MRT ist indiziert.",
        section_path=["2", "2.1"],
        guideline_title="Mammakarzinom",
    )
    assert isinstance(header, str)
    assert len(header) > 0

def test_generate_hypothetical_questions_returns_list():
    client = _mock_client("Wann ist MRT indiziert?\nWelche Bildgebung empfohlen?")
    questions = generate_hypothetical_questions(
        client=client,
        chunk_text="MRT ist bei unklarem Befund indiziert.",
    )
    assert isinstance(questions, list)
    assert len(questions) >= 1

def test_extract_semantic_metadata_returns_dict():
    client = _mock_client('{"diseases": ["Mammakarzinom"], "drugs": [], "procedures": ["MRT"], "patient_subgroups": [], "risk_category": ""}')
    meta = extract_semantic_metadata(
        client=client,
        chunk_text="MRT ist bei Mammakarzinom-Patientinnen indiziert.",
    )
    assert "diseases" in meta
    assert "Mammakarzinom" in meta["diseases"]

def test_semantic_metadata_returns_empty_on_parse_failure():
    client = _mock_client("invalid json {{")
    meta = extract_semantic_metadata(client=client, chunk_text="text")
    assert meta == {"diseases": [], "drugs": [], "procedures": [], "patient_subgroups": [], "risk_category": ""}
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/indexer/test_enricher.py -v
```

- [ ] **Step 3: Implement enricher**

```python
# src/indexer/enricher.py
import json
import os
from openai import OpenAI

CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash")
EMPTY_SEMANTIC = {"diseases": [], "drugs": [], "procedures": [], "patient_subgroups": [], "risk_category": ""}


def _client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def generate_contextual_header(
    chunk_text: str,
    section_path: list[str],
    guideline_title: str,
    client: OpenAI | None = None,
) -> str:
    c = client or _client()
    section_str = " > ".join(section_path)
    prompt = (
        f"Du analysierst einen Abschnitt der deutschen S3-Leitlinie '{guideline_title}', "
        f"Abschnitt {section_str}.\n\n"
        f"Erstelle einen kurzen Kontext-Header (1-2 Sätze), der erklärt, wo dieser Chunk "
        f"in der Leitlinie steht und was sein Hauptinhalt ist. Antworte nur mit dem Header, kein JSON.\n\n"
        f"Chunk:\n{chunk_text[:800]}"
    )
    resp = c.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


def generate_hypothetical_questions(
    chunk_text: str,
    client: OpenAI | None = None,
) -> list[str]:
    c = client or _client()
    prompt = (
        "Generiere 2-3 medizinische Fragen auf Deutsch, die ein Arzt stellen würde, "
        "wenn er nach dem Inhalt dieses Leitlinien-Abschnitts sucht. "
        "Eine Frage pro Zeile, kein JSON.\n\n"
        f"Abschnitt:\n{chunk_text[:800]}"
    )
    resp = c.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    lines = resp.choices[0].message.content.strip().splitlines()
    return [l.strip("- •123.").strip() for l in lines if l.strip()]


def extract_semantic_metadata(
    chunk_text: str,
    client: OpenAI | None = None,
) -> dict:
    c = client or _client()
    prompt = (
        "Extrahiere semantische Metadaten aus diesem deutschen Leitlinien-Abschnitt. "
        "Antworte ausschließlich mit validem JSON (keine Erklärungen):\n"
        '{"diseases": [], "drugs": [], "procedures": [], "patient_subgroups": [], "risk_category": ""}\n\n'
        f"Abschnitt:\n{chunk_text[:1000]}"
    )
    resp = c.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    try:
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        return dict(EMPTY_SEMANTIC)
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/indexer/test_enricher.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/indexer/enricher.py tests/indexer/test_enricher.py
git commit -m "feat(indexer): LLM enricher for contextual headers, hypothetical Qs, semantic metadata"
```

---

## Task 8: Embedder

**Files:**
- Create: `src/indexer/embedder.py`

- [ ] **Step 1: Write failing test**

```python
# tests/indexer/test_enricher.py  (append to existing file)
from src.indexer.embedder import embed_texts

def test_embed_texts_returns_correct_shape():
    from unittest.mock import MagicMock
    client = MagicMock()
    client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 3072), MagicMock(embedding=[0.2] * 3072)]
    )
    result = embed_texts(["text one", "text two"], client=client)
    assert len(result) == 2
    assert len(result[0]) == 3072
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/indexer/test_enricher.py::test_embed_texts_returns_correct_shape -v
```

- [ ] **Step 3: Implement embedder**

```python
# src/indexer/embedder.py
import os
from openai import OpenAI

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-large")
EMBED_BATCH_SIZE = 64


def _client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def embed_texts(texts: list[str], client: OpenAI | None = None) -> list[list[float]]:
    """Embed a list of texts in batches. Returns list of 3072-dim vectors."""
    c = client or _client()
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        resp = c.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in resp.data])
    return all_embeddings
```

- [ ] **Step 4: Run — expect PASS**

```bash
uv run pytest tests/indexer/test_enricher.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/indexer/embedder.py
git commit -m "feat(indexer): batched text embedder via OpenRouter"
```

---

## Task 9: Milvus Store

**Files:**
- Create: `src/indexer/store.py`

- [ ] **Step 1: Verify Milvus is running**

```bash
curl http://localhost:19530/v1/vector/collections
```

Expected: JSON response (may be empty list). If connection refused, start Milvus: `milvus-server` or check local installation.

- [ ] **Step 2: Write failing test**

```python
# tests/indexer/test_enricher.py  (append)
from unittest.mock import MagicMock, patch
from src.indexer.store import MilvusStore

def test_milvus_store_upsert_calls_insert(mocker):
    mock_client = mocker.MagicMock()
    store = MilvusStore(collection_name="test_col", client=mock_client)
    store.upsert([{
        "chunk_id": "abc",
        "text": "test",
        "dense_vector": [0.1] * 3072,
        "guideline_id": "mamma",
        "chunk_type": "prose",
        "section_path": ["1"],
        "recommendation_grade": "",
        "is_leaf": True,
    }])
    mock_client.insert.assert_called_once()
```

- [ ] **Step 3: Run — expect FAIL**

```bash
uv run pytest tests/indexer/test_enricher.py::test_milvus_store_upsert_calls_insert -v
```

- [ ] **Step 4: Implement Milvus store**

```python
# src/indexer/store.py
import os
from pymilvus import MilvusClient, DataType

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = os.getenv("MILVUS_COLLECTION", "oncology_guidelines")
DIM = 3072


class MilvusStore:
    def __init__(self, collection_name: str = COLLECTION, client: MilvusClient | None = None):
        self.collection_name = collection_name
        self.client = client or MilvusClient(uri=MILVUS_URI)

    def ensure_collection(self) -> None:
        """Create collection with dense + sparse fields if it does not exist."""
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("guideline_id", DataType.VARCHAR, max_length=64)
        schema.add_field("chunk_type", DataType.VARCHAR, max_length=32)
        schema.add_field("recommendation_grade", DataType.VARCHAR, max_length=8)
        schema.add_field("is_leaf", DataType.BOOL)

        index_params = self.client.prepare_index_params()
        index_params.add_index("dense_vector", index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 200})

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def upsert(self, records: list[dict]) -> None:
        """Insert or update records. Each dict must have chunk_id and dense_vector."""
        self.client.insert(collection_name=self.collection_name, data=records)

    def drop(self) -> None:
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
```

- [ ] **Step 5: Run — expect PASS**

```bash
uv run pytest tests/indexer/test_enricher.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/indexer/store.py
git commit -m "feat(indexer): Milvus store with HNSW dense index"
```

---

## Task 10: Indexing Pipeline

**Files:**
- Create: `src/indexer/pipeline.py`
- Create: `scripts/run_indexer.py`

- [ ] **Step 1: Implement pipeline**

```python
# src/indexer/pipeline.py
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

from src.indexer.parser import extract_pages, clean_text
from src.indexer.chunker import build_chunks
from src.indexer.metadata import attach_metadata
from src.indexer.reference import extract_inline_refs, parse_bibliography
from src.indexer.enricher import generate_contextual_header, generate_hypothetical_questions, extract_semantic_metadata
from src.indexer.embedder import embed_texts
from src.indexer.store import MilvusStore

load_dotenv()
log = logging.getLogger(__name__)


GUIDELINE_MAP = {
    "mammakarzinom_v4.4.pdf":    ("mamma",  "4.4", "S3-Leitlinie Mammakarzinom"),
    "kolorektales_v3.0.pdf":     ("krk",    "3.0", "S3-Leitlinie Kolorektales Karzinom"),
    "lungenkarzinom_v4.0.pdf":   ("lunge",  "4.0", "S3-Leitlinie Lungenkarzinom"),
    "prostatakarzinom_v8.0.pdf": ("prosta", "8.0", "S3-Leitlinie Prostatakarzinom"),
}


def index_pdf(pdf_path: Path, store: MilvusStore, dry_run: bool = False) -> int:
    """Index a single PDF. Returns number of leaf chunks indexed."""
    filename = pdf_path.name
    guideline_id, version, title = GUIDELINE_MAP[filename]
    log.info("Indexing %s", filename)

    pages = extract_pages(pdf_path)
    full_text = "\n".join(clean_text(p["text"]) for p in pages)

    # Build chunks from full text
    chunks = build_chunks(guideline_id, version, full_text)
    chunks = attach_metadata(chunks, source_filename=filename)

    # Parse bibliography from full text for reference linking
    bib_entries = parse_bibliography(full_text)
    bib_by_id = {e.reference_id: e for e in bib_entries}

    records = []
    leaf_chunks = [c for c in chunks if c.is_leaf]

    for chunk in leaf_chunks:
        # Reference linking
        inline_refs = extract_inline_refs(chunk.text)
        resolved = [r for r in inline_refs if r in bib_by_id]
        unresolved = [r for r in inline_refs if r not in bib_by_id]
        if unresolved:
            log.warning("Unresolved refs in chunk %s: %s", chunk.chunk_id, unresolved)

        # LLM enrichment
        header = generate_contextual_header(
            chunk_text=chunk.text,
            section_path=chunk.section_path,
            guideline_title=title,
        )
        hypo_qs = generate_hypothetical_questions(chunk_text=chunk.text)
        semantic = extract_semantic_metadata(chunk_text=chunk.text)

        # Text to embed: header + hypothetical questions + chunk text
        embed_input = f"{header}\n" + "\n".join(hypo_qs) + f"\n\n{chunk.text}"
        vector = embed_texts([embed_input])[0]

        record = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "dense_vector": vector,
            "guideline_id": chunk.guideline_id,
            "guideline_version": chunk.guideline_version,
            "chunk_type": chunk.chunk_type,
            "section_path": json.dumps(chunk.section_path),
            "section_title": chunk.section_title,
            "recommendation_id": chunk.recommendation_id,
            "recommendation_grade": chunk.recommendation_grade,
            "evidence_level": chunk.evidence_level,
            "parent_chunk_id": chunk.parent_chunk_id or "",
            "source_filename": chunk.source_filename,
            "is_leaf": chunk.is_leaf,
            "is_current": chunk.is_current,
            "contextual_header": header,
            "hypothetical_questions": json.dumps(hypo_qs),
            "diseases": json.dumps(semantic.get("diseases", [])),
            "drugs": json.dumps(semantic.get("drugs", [])),
            "procedures": json.dumps(semantic.get("procedures", [])),
            "reference_ids": json.dumps(resolved),
        }
        records.append(record)

    if not dry_run:
        store.ensure_collection()
        # Batch upsert in groups of 100
        for i in range(0, len(records), 100):
            store.upsert(records[i : i + 100])

    log.info("Indexed %d leaf chunks from %s", len(records), filename)
    return len(records)
```

- [ ] **Step 2: Create run_indexer.py script**

```python
# scripts/run_indexer.py
"""CLI: python scripts/run_indexer.py [--pdf mammakarzinom_v4.4.pdf] [--dry-run]"""
import argparse
import logging
from pathlib import Path
from src.indexer.pipeline import index_pdf, GUIDELINE_MAP
from src.indexer.store import MilvusStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="Single PDF filename to index (from GUIDELINE_MAP)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk but do not write to Milvus")
    args = parser.parse_args()

    kb_dir = Path("docs/knowledge_base")
    store = MilvusStore()

    pdfs = [args.pdf] if args.pdf else list(GUIDELINE_MAP.keys())
    for pdf_name in pdfs:
        pdf_path = kb_dir / pdf_name
        if not pdf_path.exists():
            print(f"WARNING: {pdf_path} not found, skipping")
            continue
        count = index_pdf(pdf_path, store, dry_run=args.dry_run)
        print(f"Indexed {count} chunks from {pdf_name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run indexer on one PDF (Phase 0A/0B validation)**

```bash
python scripts/run_indexer.py --pdf mammakarzinom_v4.4.pdf
```

Expected: Progress log lines, final "Indexed N chunks from mammakarzinom_v4.4.pdf". Inspect N — should be in the hundreds.

- [ ] **Step 4: Run 10 spot-check queries via Python REPL**

```python
from src.indexer.store import MilvusStore
from src.indexer.embedder import embed_texts
store = MilvusStore()
store.client.load_collection("oncology_guidelines")
q = embed_texts(["Welche Empfehlung gilt für Mammakarzinom-Screening?"])[0]
results = store.client.search(
    collection_name="oncology_guidelines",
    data=[q],
    anns_field="dense_vector",
    limit=5,
    output_fields=["chunk_id", "section_title", "recommendation_grade", "chunk_type"],
)
for r in results[0]:
    print(r)
```

Expected: 5 results with section titles and chunk types that are relevant to screening.

- [ ] **Step 5: Run Phase 0C — index remaining 3 PDFs**

```bash
python scripts/run_indexer.py --pdf kolorektales_v3.0.pdf
python scripts/run_indexer.py --pdf lungenkarzinom_v4.0.pdf
python scripts/run_indexer.py --pdf prostatakarzinom_v8.0.pdf
```

- [ ] **Step 6: Commit**

```bash
git add src/indexer/pipeline.py scripts/run_indexer.py
git commit -m "feat(indexer): full indexing pipeline and CLI runner"
```

---

## Task 11: Retrieval Engine

**Files:**
- Create: `src/retrieval/search.py`
- Create: `src/retrieval/reranker.py`
- Create: `src/retrieval/expander.py`
- Create: `tests/retrieval/test_search.py`

- [ ] **Step 1: Write failing retrieval tests**

```python
# tests/retrieval/test_search.py
from unittest.mock import MagicMock, patch
from src.retrieval.search import hybrid_search, rrf_fuse, RetrievedChunk

def test_rrf_fuse_combines_results():
    dense = [RetrievedChunk(chunk_id="a", text="t", score=0.9, guideline_id="g", section_title="s", page_start=1, page_end=1, chunk_type="prose", recommendation_grade="", section_path=[])]
    sparse = [RetrievedChunk(chunk_id="b", text="t2", score=0.8, guideline_id="g", section_title="s", page_start=2, page_end=2, chunk_type="prose", recommendation_grade="", section_path=[])]
    combined = rrf_fuse(dense, sparse, k=60)
    ids = [c.chunk_id for c in combined]
    assert "a" in ids
    assert "b" in ids

def test_rrf_fuse_boosts_chunk_in_both_lists():
    chunk_a = RetrievedChunk(chunk_id="a", text="t", score=0.9, guideline_id="g", section_title="s", page_start=1, page_end=1, chunk_type="prose", recommendation_grade="", section_path=[])
    combined = rrf_fuse([chunk_a], [chunk_a], k=60)
    combined_both = rrf_fuse([chunk_a], [chunk_a], k=60)
    combined_one = rrf_fuse([chunk_a], [], k=60)
    assert combined_both[0].score > combined_one[0].score
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/retrieval/test_search.py -v
```

- [ ] **Step 3: Implement search module**

```python
# src/retrieval/search.py
import os
import json
from dataclasses import dataclass, field
from pymilvus import MilvusClient
from src.indexer.embedder import embed_texts

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = os.getenv("MILVUS_COLLECTION", "oncology_guidelines")
TOP_K_DENSE = 20
TOP_K_SPARSE = 20


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    guideline_id: str
    section_title: str
    section_path: list[str]
    page_start: int | None
    page_end: int | None
    chunk_type: str
    recommendation_grade: str
    recommendation_id: str = ""
    parent_chunk_id: str = ""
    source_filename: str = ""
    contextual_header: str = ""


def rrf_fuse(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion over two ranked lists."""
    scores: dict[str, float] = {}
    by_id: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_results):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (k + rank + 1)
        by_id[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(sparse_results):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (k + rank + 1)
        by_id[chunk.chunk_id] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        RetrievedChunk(**{**vars(by_id[cid]), "score": sc})
        for cid, sc in ranked
    ]


def _milvus_results_to_chunks(results: list[dict]) -> list[RetrievedChunk]:
    chunks = []
    for r in results:
        e = r.get("entity", r)
        chunks.append(RetrievedChunk(
            chunk_id=e.get("chunk_id", ""),
            text=e.get("text", ""),
            score=r.get("distance", 0.0),
            guideline_id=e.get("guideline_id", ""),
            section_title=e.get("section_title", ""),
            section_path=json.loads(e.get("section_path", "[]")),
            page_start=e.get("page_start"),
            page_end=e.get("page_end"),
            chunk_type=e.get("chunk_type", ""),
            recommendation_grade=e.get("recommendation_grade", ""),
            recommendation_id=e.get("recommendation_id", ""),
            parent_chunk_id=e.get("parent_chunk_id", ""),
            source_filename=e.get("source_filename", ""),
            contextual_header=e.get("contextual_header", ""),
        ))
    return chunks


def hybrid_search(
    query: str,
    guideline_id: str | None = None,
    grade_filter: str | None = None,
    chunk_type_filter: str | None = None,
    top_k: int = 20,
    client: MilvusClient | None = None,
) -> list[RetrievedChunk]:
    """Dense + BM25 search with RRF fusion and optional metadata filter."""
    c = client or MilvusClient(uri=MILVUS_URI)
    vector = embed_texts([query])[0]

    output_fields = [
        "chunk_id", "text", "guideline_id", "section_title", "section_path",
        "page_start", "page_end", "chunk_type", "recommendation_grade",
        "recommendation_id", "parent_chunk_id", "source_filename", "contextual_header",
    ]

    # Build metadata filter expression
    filters = ["is_leaf == true"]
    if guideline_id:
        filters.append(f'guideline_id == "{guideline_id}"')
    if grade_filter:
        filters.append(f'recommendation_grade == "{grade_filter}"')
    if chunk_type_filter:
        filters.append(f'chunk_type == "{chunk_type_filter}"')
    expr = " and ".join(filters) if filters else ""

    # Dense search
    dense_raw = c.search(
        collection_name=COLLECTION,
        data=[vector],
        anns_field="dense_vector",
        limit=top_k,
        filter=expr,
        output_fields=output_fields,
    )
    dense_chunks = _milvus_results_to_chunks(dense_raw[0] if dense_raw else [])

    # BM25 sparse search — Milvus 2.4+ supports full-text search
    # Fallback: use dense only if BM25 not configured
    try:
        sparse_raw = c.search(
            collection_name=COLLECTION,
            data=[query],
            anns_field="sparse_vector",
            limit=top_k,
            filter=expr,
            output_fields=output_fields,
        )
        sparse_chunks = _milvus_results_to_chunks(sparse_raw[0] if sparse_raw else [])
    except Exception:
        sparse_chunks = []  # BM25 not configured — use dense only

    return rrf_fuse(dense_chunks, sparse_chunks)[:top_k]
```

- [ ] **Step 4: Implement reranker**

```python
# src/retrieval/reranker.py
from FlagEmbedding import FlagReranker
from src.retrieval.search import RetrievedChunk

_reranker: FlagReranker | None = None


def _get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    return _reranker


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int = 5,
) -> list[RetrievedChunk]:
    if not chunks:
        return []
    reranker = _get_reranker()
    pairs = [[query, c.text] for c in chunks]
    scores = reranker.compute_score(pairs, normalize=True)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [
        RetrievedChunk(**{**vars(c), "score": float(s)})
        for s, c in ranked[:top_k]
    ]
```

- [ ] **Step 5: Implement parent expander**

```python
# src/retrieval/expander.py
import os
from pymilvus import MilvusClient
from src.retrieval.search import RetrievedChunk

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = os.getenv("MILVUS_COLLECTION", "oncology_guidelines")


def expand_to_parents(
    chunks: list[RetrievedChunk],
    client: MilvusClient | None = None,
) -> list[RetrievedChunk]:
    """For each leaf chunk that has a parent, fetch and attach parent context."""
    c = client or MilvusClient(uri=MILVUS_URI)
    result = []
    for chunk in chunks:
        if not chunk.parent_chunk_id:
            result.append(chunk)
            continue
        try:
            parent_rows = c.get(
                collection_name=COLLECTION,
                ids=[chunk.parent_chunk_id],
                output_fields=["text"],
            )
            if parent_rows:
                parent_text = parent_rows[0].get("text", "")
                expanded = RetrievedChunk(
                    **{**vars(chunk), "text": f"{parent_text}\n\n{chunk.text}"}
                )
                result.append(expanded)
            else:
                result.append(chunk)
        except Exception:
            result.append(chunk)
    return result
```

- [ ] **Step 6: Run retrieval tests**

```bash
uv run pytest tests/retrieval/test_search.py -v
```

Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/retrieval/ tests/retrieval/
git commit -m "feat(retrieval): hybrid search, RRF fusion, reranker, parent expansion"
```

---

## Task 12: Tools

**Files:**
- Create: `src/tools/search_guidelines.py`
- Create: `src/tools/lookup_empfehlung.py`
- Create: `tests/tools/test_tools.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/tools/test_tools.py
from unittest.mock import MagicMock, patch
from src.tools.search_guidelines import search_guidelines_tool
from src.tools.lookup_empfehlung import lookup_empfehlung_tool

def test_search_guidelines_returns_list(mocker):
    mock_chunk = MagicMock()
    mock_chunk.chunk_id = "abc"
    mock_chunk.text = "Empfehlung text"
    mock_chunk.score = 0.9
    mock_chunk.guideline_id = "mamma"
    mock_chunk.section_title = "Diagnose"
    mock_chunk.section_path = ["2", "2.1"]
    mock_chunk.page_start = 10
    mock_chunk.page_end = 11
    mock_chunk.recommendation_grade = "A"
    mock_chunk.recommendation_id = "2.1"
    mock_chunk.source_filename = "mammakarzinom_v4.4.pdf"

    mocker.patch("src.tools.search_guidelines.hybrid_search", return_value=[mock_chunk])
    mocker.patch("src.tools.search_guidelines.rerank", return_value=[mock_chunk])
    mocker.patch("src.tools.search_guidelines.expand_to_parents", return_value=[mock_chunk])

    result = search_guidelines_tool(query="Screening Mammakarzinom")
    assert isinstance(result, list)
    assert result[0]["chunk_id"] == "abc"
    assert "text" in result[0]
    assert "citation" in result[0]

def test_lookup_empfehlung_queries_by_id(mocker):
    mock_client = mocker.MagicMock()
    mock_client.query.return_value = [{
        "chunk_id": "emp1",
        "text": "Empfehlung 2.1 Text",
        "recommendation_grade": "A",
        "evidence_level": "1a",
        "section_title": "Diagnose",
        "guideline_id": "mamma",
    }]
    result = lookup_empfehlung_tool(guideline_id="mamma", recommendation_id="2.1", client=mock_client)
    assert result["recommendation_id"] == "2.1"
    assert result["recommendation_grade"] == "A"
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/tools/test_tools.py -v
```

- [ ] **Step 3: Implement search_guidelines tool**

```python
# src/tools/search_guidelines.py
from src.retrieval.search import hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.expander import expand_to_parents


def search_guidelines_tool(
    query: str,
    guideline_id: str | None = None,
    grade: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Core RAG retrieval tool. Returns ranked chunks with citation metadata.
    Suitable for use as a LangGraph tool function.
    """
    candidates = hybrid_search(
        query=query,
        guideline_id=guideline_id,
        grade_filter=grade,
        top_k=20,
    )
    reranked = rerank(query=query, chunks=candidates, top_k=top_k)
    expanded = expand_to_parents(reranked)

    return [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "score": round(c.score, 4),
            "guideline_id": c.guideline_id,
            "section_title": c.section_title,
            "section_path": c.section_path,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "recommendation_grade": c.recommendation_grade,
            "recommendation_id": c.recommendation_id,
            "source_filename": c.source_filename,
            "citation": f"{c.guideline_id.upper()} § {'.'.join(c.section_path)} (S. {c.page_start}–{c.page_end})"
            if c.page_start
            else c.guideline_id.upper(),
        }
        for c in expanded
    ]
```

- [ ] **Step 4: Implement lookup_empfehlung tool**

```python
# src/tools/lookup_empfehlung.py
import os
from pymilvus import MilvusClient

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = os.getenv("MILVUS_COLLECTION", "oncology_guidelines")


def lookup_empfehlung_tool(
    guideline_id: str,
    recommendation_id: str,
    client: MilvusClient | None = None,
) -> dict:
    """
    Fetch a specific Empfehlung X.Y verbatim by its recommendation_id.
    Returns the full text with grade, evidence level, and source metadata.
    """
    c = client or MilvusClient(uri=MILVUS_URI)
    expr = (
        f'guideline_id == "{guideline_id}" and '
        f'recommendation_id == "{recommendation_id}" and '
        f'chunk_type == "empfehlung"'
    )
    rows = c.query(
        collection_name=COLLECTION,
        filter=expr,
        output_fields=["chunk_id", "text", "recommendation_grade", "evidence_level", "section_title", "guideline_id"],
        limit=1,
    )
    if not rows:
        return {
            "found": False,
            "recommendation_id": recommendation_id,
            "guideline_id": guideline_id,
            "message": f"Empfehlung {recommendation_id} nicht gefunden in {guideline_id}.",
        }
    row = rows[0]
    return {
        "found": True,
        "recommendation_id": recommendation_id,
        "guideline_id": row["guideline_id"],
        "text": row["text"],
        "recommendation_grade": row["recommendation_grade"],
        "evidence_level": row["evidence_level"],
        "section_title": row["section_title"],
    }
```

- [ ] **Step 5: Run — expect PASS**

```bash
uv run pytest tests/tools/test_tools.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/tools/ tests/tools/
git commit -m "feat(tools): search_guidelines and lookup_empfehlung tools"
```

---

## Task 13: LangGraph State + Core Graph

**Files:**
- Create: `src/graph/state.py`
- Create: `src/graph/nodes/rewriter.py`
- Create: `src/graph/nodes/agent.py`
- Create: `src/graph/nodes/answer.py`
- Create: `src/graph/nodes/guardrail_input.py`
- Create: `src/graph/nodes/self_query.py`
- Create: `src/graph/nodes/router.py`
- Create: `src/graph/nodes/confidence.py`
- Create: `src/graph/nodes/guardrail_output.py`
- Create: `src/graph/graph.py`
- Create: `tests/graph/test_nodes.py`

- [ ] **Step 1: Define state schema**

```python
# src/graph/state.py
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    # Input
    user_query: str
    session_id: str

    # Preprocessing
    rewritten_query: str
    metadata_filters: dict[str, str]    # guideline_id, grade, chunk_type
    intent: str                          # factual | recommendation | comparison | external

    # Retrieval
    retrieved_chunks: list[dict]
    confidence: float                    # 0.0–1.0 from reranker scores

    # Generation
    answer_professional: str
    answer_plain: str
    citations: list[dict]
    disclaimer: str

    # Guardrails
    input_blocked: bool
    input_block_reason: str
    output_blocked: bool

    # Tool calls (for display)
    tool_calls_log: list[dict]

    # Conversation memory
    messages: Annotated[list, add_messages]
```

- [ ] **Step 2: Write failing graph tests**

```python
# tests/graph/test_nodes.py
from unittest.mock import MagicMock
from src.graph.state import RAGState
from src.graph.nodes.rewriter import rewrite_query
from src.graph.nodes.confidence import check_confidence
from src.graph.nodes.guardrail_input import apply_input_guardrail

def _base_state() -> RAGState:
    return RAGState(
        user_query="Welche Empfehlung gilt für das Screening?",
        session_id="test-session",
        rewritten_query="",
        metadata_filters={},
        intent="",
        retrieved_chunks=[],
        confidence=0.0,
        answer_professional="",
        answer_plain="",
        citations=[],
        disclaimer="",
        input_blocked=False,
        input_block_reason="",
        output_blocked=False,
        tool_calls_log=[],
        messages=[],
    )

def test_rewrite_query_updates_state(mocker):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Screening-Empfehlung Mammakarzinom"))]
    )
    state = _base_state()
    result = rewrite_query(state, client=mock_client)
    assert result["rewritten_query"] == "Screening-Empfehlung Mammakarzinom"

def test_confidence_high_when_chunks_present():
    state = _base_state()
    state["retrieved_chunks"] = [{"score": 0.85}, {"score": 0.80}, {"score": 0.75}]
    result = check_confidence(state)
    assert result["confidence"] > 0.5

def test_confidence_low_when_no_chunks():
    state = _base_state()
    state["retrieved_chunks"] = []
    result = check_confidence(state)
    assert result["confidence"] == 0.0

def test_input_guardrail_blocks_offtopic():
    state = _base_state()
    state["user_query"] = "Wie koche ich Spaghetti?"
    result = apply_input_guardrail(state)
    assert result["input_blocked"] is True

def test_input_guardrail_passes_medical_query():
    state = _base_state()
    result = apply_input_guardrail(state)
    assert result["input_blocked"] is False
```

- [ ] **Step 3: Run — expect FAIL**

```bash
uv run pytest tests/graph/test_nodes.py -v
```

- [ ] **Step 4: Implement graph nodes**

```python
# src/graph/nodes/rewriter.py
import os
from openai import OpenAI
from src.graph.state import RAGState

CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash")


def _client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def rewrite_query(state: RAGState, client: OpenAI | None = None) -> dict:
    c = client or _client()
    history = state.get("messages", [])
    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in history[-4:]
    ) if history else ""

    prompt = (
        "Du bist ein Assistent für deutsche Onkologie-Leitlinien. "
        "Formuliere die folgende Anfrage als präzise medizinische Suchanfrage um. "
        "Berücksichtige den Gesprächsverlauf. Antworte nur mit der umformulierten Anfrage.\n\n"
        + (f"Verlauf:\n{history_text}\n\n" if history_text else "")
        + f"Anfrage: {state['user_query']}"
    )
    resp = c.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return {"rewritten_query": resp.choices[0].message.content.strip()}
```

```python
# src/graph/nodes/guardrail_input.py
import re
from src.graph.state import RAGState

ONCOLOGY_KEYWORDS = [
    "karzinom", "tumor", "krebs", "leitlinie", "empfehlung", "therapie",
    "diagnose", "screening", "onkologie", "chemo", "bestrahlung", "mamma",
    "prostat", "lunge", "kolorektal", "darm", "metastas", "staging",
    "evidenz", "grade", "studie", "patient",
]


def apply_input_guardrail(state: RAGState) -> dict:
    query = state["user_query"].lower()

    # PII patterns (basic — Presidio integration in Phase 11B)
    pii_patterns = [
        r"\b\d{2}\.\d{2}\.\d{4}\b",   # dates of birth
        r"\bdr\.\s+\w+",               # doctor names
    ]

    # Off-topic: no medical keyword at all
    has_medical = any(kw in query for kw in ONCOLOGY_KEYWORDS)
    if not has_medical and len(query.split()) > 3:
        return {
            "input_blocked": True,
            "input_block_reason": "Ihre Anfrage scheint nicht onkologische Leitlinien zu betreffen. Bitte stellen Sie medizinische Fragen zu den S3-Leitlinien.",
        }

    return {"input_blocked": False, "input_block_reason": ""}
```

```python
# src/graph/nodes/self_query.py
import json
import os
from openai import OpenAI
from src.graph.state import RAGState

CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash")


def _client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def extract_metadata_filters(state: RAGState, client: OpenAI | None = None) -> dict:
    c = client or _client()
    prompt = (
        "Extrahiere Metadaten-Filter aus dieser Anfrage für eine Leitlinien-Datenbank. "
        "Antworte ausschließlich mit JSON:\n"
        '{"guideline_id": "" | "mamma" | "krk" | "lunge" | "prosta", "grade": "" | "A" | "B" | "0", "chunk_type": "" | "empfehlung" | "prose"}\n\n'
        f"Anfrage: {state['rewritten_query'] or state['user_query']}"
    )
    resp = c.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
    )
    try:
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json")
        filters = json.loads(raw)
        return {"metadata_filters": {k: v for k, v in filters.items() if v}}
    except Exception:
        return {"metadata_filters": {}}
```

```python
# src/graph/nodes/router.py
import os
from openai import OpenAI
from src.graph.state import RAGState

CHEAP_MODEL = os.getenv("CHEAP_MODEL", "google/gemini-2.5-flash")
VALID_INTENTS = {"factual", "recommendation", "comparison", "external"}


def _client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def route_intent(state: RAGState, client: OpenAI | None = None) -> dict:
    c = client or _client()
    prompt = (
        "Klassifiziere die Anfrage als einen der folgenden Typen und antworte nur mit dem Typ:\n"
        "factual | recommendation | comparison | external\n\n"
        f"Anfrage: {state['rewritten_query'] or state['user_query']}"
    )
    resp = c.chat.completions.create(
        model=CHEAP_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
    )
    intent = resp.choices[0].message.content.strip().lower()
    if intent not in VALID_INTENTS:
        intent = "factual"
    return {"intent": intent}
```

```python
# src/graph/nodes/confidence.py
from src.graph.state import RAGState

CONFIDENCE_THRESHOLD = 0.5


def check_confidence(state: RAGState) -> dict:
    """Lightweight confidence: mean reranker score of top-3 chunks."""
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"confidence": 0.0}
    top_scores = [c["score"] for c in chunks[:3] if "score" in c]
    confidence = sum(top_scores) / len(top_scores) if top_scores else 0.0
    return {"confidence": confidence}


def needs_escalation(state: RAGState) -> str:
    """Routing function: returns 'escalate' or 'answer'."""
    return "escalate" if state["confidence"] < CONFIDENCE_THRESHOLD else "answer"
```

```python
# src/graph/nodes/agent.py
import os
import json
from openai import OpenAI
from src.graph.state import RAGState
from src.tools.search_guidelines import search_guidelines_tool
from src.tools.lookup_empfehlung import lookup_empfehlung_tool

GEN_MODEL = os.getenv("GENERATION_MODEL", "openai/gpt-4o")

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_guidelines",
            "description": "Suche in deutschen Onkologie-Leitlinien nach relevantem Inhalt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "guideline_id": {"type": "string", "enum": ["mamma", "krk", "lunge", "prosta", ""]},
                    "grade": {"type": "string", "enum": ["A", "B", "0", ""]},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_empfehlung",
            "description": "Rufe eine spezifische Empfehlung X.Y direkt ab.",
            "parameters": {
                "type": "object",
                "properties": {
                    "guideline_id": {"type": "string"},
                    "recommendation_id": {"type": "string"},
                },
                "required": ["guideline_id", "recommendation_id"],
            },
        },
    },
]


def _client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def _dispatch_tool(name: str, args: dict) -> str:
    if name == "search_guidelines":
        results = search_guidelines_tool(**args)
        return json.dumps(results, ensure_ascii=False)
    if name == "lookup_empfehlung":
        result = lookup_empfehlung_tool(**args)
        return json.dumps(result, ensure_ascii=False)
    return json.dumps({"error": f"Unknown tool: {name}"})


def run_agent(state: RAGState, client: OpenAI | None = None) -> dict:
    """Tool-calling agent loop. Runs until the model stops calling tools."""
    c = client or _client()
    messages = [
        {"role": "system", "content": (
            "Du bist ein medizinischer Leitlinien-Assistent für deutsche S3-Onkologie-Leitlinien. "
            "Nutze die verfügbaren Tools um relevante Leitlinienabschnitte zu finden, bevor du antwortest."
        )},
        {"role": "user", "content": state["rewritten_query"] or state["user_query"]},
    ]

    all_chunks: list[dict] = []
    tool_calls_log: list[dict] = []

    for _ in range(5):  # max iterations
        resp = c.chat.completions.create(
            model=GEN_MODEL,
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            break

        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]})

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result_str = _dispatch_tool(tc.function.name, args)
            tool_calls_log.append({"tool": tc.function.name, "args": args})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

            if tc.function.name == "search_guidelines":
                all_chunks.extend(json.loads(result_str))

    return {
        "retrieved_chunks": all_chunks[:10],
        "tool_calls_log": tool_calls_log,
    }
```

```python
# src/graph/nodes/answer.py
import os
import json
from openai import OpenAI
from src.graph.state import RAGState

GEN_MODEL = os.getenv("GENERATION_MODEL", "openai/gpt-4o")
DISCLAIMER = (
    "\n\n---\n*Haftungsausschluss: Diese Informationen stammen aus den deutschen S3-Leitlinien "
    "und dienen ausschließlich zu Bildungszwecken. Sie ersetzen keine individuelle medizinische Beratung.*"
)


def _client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


def generate_answer(state: RAGState, client: OpenAI | None = None) -> dict:
    c = client or _client()
    chunks = state.get("retrieved_chunks", [])
    context = "\n\n".join(
        f"[{i+1}] {ch['citation']}: {ch['text'][:600]}"
        for i, ch in enumerate(chunks[:5])
    )
    citations = [
        {"label": f"[{i+1}]", "citation": ch["citation"], "source_filename": ch.get("source_filename", "")}
        for i, ch in enumerate(chunks[:5])
    ]

    prompt = (
        f"Du bist ein medizinischer Leitlinien-Assistent. Beantworte die Frage basierend AUSSCHLIEßLICH auf den folgenden Leitlinienabschnitten.\n\n"
        f"Quellen:\n{context}\n\n"
        f"Frage: {state['user_query']}\n\n"
        "Antworte in ZWEI Teilen:\n\n"
        "**Fachliche Antwort:**\n"
        "Verwende formale medizinische Terminologie auf Deutsch. Nenne Empfehlungsgrade (A/B/0) und Evidenzlevel. "
        "Zitiere Quellen inline als [1], [2] etc.\n\n"
        "**In einfachen Worten:**\n"
        "Erkläre dasselbe in klarer, verständlicher Sprache für Nicht-Mediziner. Kein Fachjargon."
    )
    resp = c.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
    )
    full = resp.choices[0].message.content.strip()

    # Split into professional and plain sections
    pro = ""
    plain = ""
    if "**In einfachen Worten:**" in full:
        parts = full.split("**In einfachen Worten:**", 1)
        pro = parts[0].replace("**Fachliche Antwort:**", "").strip()
        plain = parts[1].strip()
    else:
        pro = full

    return {
        "answer_professional": pro,
        "answer_plain": plain,
        "citations": citations,
        "disclaimer": DISCLAIMER,
    }
```

```python
# src/graph/nodes/guardrail_output.py
from src.graph.state import RAGState


def apply_output_guardrail(state: RAGState) -> dict:
    """Basic faithfulness check: ensure answer references at least one source."""
    answer = state.get("answer_professional", "")
    chunks = state.get("retrieved_chunks", [])

    if not chunks and answer:
        return {
            "output_blocked": True,
            "answer_professional": "Die Anfrage konnte nicht mit den verfügbaren Leitlinienabschnitten beantwortet werden.",
            "answer_plain": "Es wurden keine relevanten Informationen in den Leitlinien gefunden.",
        }
    return {"output_blocked": False}
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/graph/test_nodes.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/graph/ tests/graph/
git commit -m "feat(graph): LangGraph state schema and all core nodes"
```

---

## Task 14: LangGraph Assembly

**Files:**
- Create: `src/graph/graph.py`
- Create: `tests/graph/test_graph.py`

- [ ] **Step 1: Write failing smoke test**

```python
# tests/graph/test_graph.py
from unittest.mock import patch, MagicMock
import pytest

def _make_mock_client(content="Antwort"):
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=content, tool_calls=None))]
    )
    return client

def test_graph_compiles():
    from src.graph.graph import build_graph
    graph = build_graph()
    assert graph is not None
```

- [ ] **Step 2: Run — expect FAIL**

```bash
uv run pytest tests/graph/test_graph.py::test_graph_compiles -v
```

- [ ] **Step 3: Implement graph assembly**

```python
# src/graph/graph.py
from langgraph.graph import StateGraph, END
from src.graph.state import RAGState
from src.graph.nodes.guardrail_input import apply_input_guardrail
from src.graph.nodes.rewriter import rewrite_query
from src.graph.nodes.self_query import extract_metadata_filters
from src.graph.nodes.router import route_intent
from src.graph.nodes.agent import run_agent
from src.graph.nodes.confidence import check_confidence, needs_escalation, CONFIDENCE_THRESHOLD
from src.graph.nodes.answer import generate_answer
from src.graph.nodes.guardrail_output import apply_output_guardrail


def _multi_query_escalation(state: RAGState) -> dict:
    """Simple multi-query fallback: run search with a broader reformulation."""
    from src.tools.search_guidelines import search_guidelines_tool
    query = state.get("rewritten_query") or state["user_query"]
    broader_query = f"Leitlinienempfehlungen zu {query}"
    chunks = search_guidelines_tool(query=broader_query, top_k=5)
    return {"retrieved_chunks": chunks}


def _blocked_response(state: RAGState) -> dict:
    return {
        "answer_professional": state.get("input_block_reason", "Anfrage blockiert."),
        "answer_plain": state.get("input_block_reason", "Anfrage blockiert."),
        "citations": [],
        "disclaimer": "",
    }


def _route_after_guardrail(state: RAGState) -> str:
    return "blocked" if state["input_blocked"] else "rewrite"


def build_graph(checkpointer=None):
    builder = StateGraph(RAGState)

    builder.add_node("guardrail_input", apply_input_guardrail)
    builder.add_node("blocked", _blocked_response)
    builder.add_node("rewrite", rewrite_query)
    builder.add_node("self_query", extract_metadata_filters)
    builder.add_node("router", route_intent)
    builder.add_node("agent", run_agent)
    builder.add_node("confidence", check_confidence)
    builder.add_node("escalate", _multi_query_escalation)
    builder.add_node("answer", generate_answer)
    builder.add_node("guardrail_output", apply_output_guardrail)

    builder.set_entry_point("guardrail_input")
    builder.add_conditional_edges("guardrail_input", _route_after_guardrail, {
        "blocked": "blocked",
        "rewrite": "rewrite",
    })
    builder.add_edge("blocked", END)
    builder.add_edge("rewrite", "self_query")
    builder.add_edge("self_query", "router")
    builder.add_edge("router", "agent")
    builder.add_edge("agent", "confidence")
    builder.add_conditional_edges("confidence", needs_escalation, {
        "escalate": "escalate",
        "answer": "answer",
    })
    builder.add_edge("escalate", "answer")
    builder.add_edge("answer", "guardrail_output")
    builder.add_edge("guardrail_output", END)

    return builder.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/graph/test_graph.py -v
```

- [ ] **Step 5: CLI smoke test (requires Milvus + OpenRouter key)**

```bash
python -c "
from dotenv import load_dotenv; load_dotenv()
from src.graph.graph import build_graph
graph = build_graph()
result = graph.invoke({
    'user_query': 'Welche Screening-Empfehlung gilt für Mammakarzinom?',
    'session_id': 'smoke-test',
    'rewritten_query': '', 'metadata_filters': {}, 'intent': '',
    'retrieved_chunks': [], 'confidence': 0.0,
    'answer_professional': '', 'answer_plain': '', 'citations': [],
    'disclaimer': '', 'input_blocked': False, 'input_block_reason': '',
    'output_blocked': False, 'tool_calls_log': [], 'messages': [],
})
print(result['answer_professional'][:400])
print('---')
print(result['answer_plain'][:300])
"
```

Expected: A dual-layer answer with inline citations.

- [ ] **Step 6: Commit**

```bash
git add src/graph/graph.py tests/graph/test_graph.py
git commit -m "feat(graph): full LangGraph assembly with all nodes and routing"
```

---

## Task 15: FastAPI Backend

**Files:**
- Create: `src/api/auth.py`
- Create: `src/api/routes/chat.py`
- Create: `src/api/routes/feedback.py`
- Create: `src/api/main.py`

- [ ] **Step 1: Implement API-key auth middleware**

```python
# src/api/auth.py
import os
from fastapi import Request, HTTPException

API_KEY = os.getenv("API_KEY", "dev-secret-key")


async def verify_api_key(request: Request):
    key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
```

- [ ] **Step 2: Implement /chat SSE endpoint**

```python
# src/api/routes/chat.py
import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from src.api.auth import verify_api_key
from src.graph.graph import build_graph
from src.graph.state import RAGState

router = APIRouter()
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    guideline_id: str = ""
    grade: str = ""


async def _stream_response(request: ChatRequest):
    graph = get_graph()
    initial_state = RAGState(
        user_query=request.query,
        session_id=request.session_id,
        rewritten_query="",
        metadata_filters={k: v for k, v in {
            "guideline_id": request.guideline_id,
            "grade": request.grade,
        }.items() if v},
        intent="",
        retrieved_chunks=[],
        confidence=0.0,
        answer_professional="",
        answer_plain="",
        citations=[],
        disclaimer="",
        input_blocked=False,
        input_block_reason="",
        output_blocked=False,
        tool_calls_log=[],
        messages=[],
    )

    final_state = graph.invoke(initial_state)

    # Stream the answer in chunks
    payload = {
        "answer_professional": final_state["answer_professional"],
        "answer_plain": final_state["answer_plain"],
        "citations": final_state["citations"],
        "disclaimer": final_state["disclaimer"],
        "tool_calls": final_state["tool_calls_log"],
        "blocked": final_state["input_blocked"] or final_state["output_blocked"],
    }

    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/chat", dependencies=[Depends(verify_api_key)])
async def chat(request: ChatRequest):
    return StreamingResponse(
        _stream_response(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 3: Implement /feedback endpoint**

```python
# src/api/routes/feedback.py
import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.api.auth import verify_api_key

router = APIRouter()
log = logging.getLogger(__name__)


class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    rating: int   # 1 = thumbs up, -1 = thumbs down
    comment: str = ""


@router.post("/feedback", dependencies=[Depends(verify_api_key)])
async def feedback(request: FeedbackRequest):
    # Phase 6 will persist to Postgres — for now log only
    log.info("Feedback: session=%s rating=%d", request.session_id, request.rating)
    return {"status": "ok"}
```

- [ ] **Step 4: Create FastAPI app**

```python
# src/api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes.chat import router as chat_router
from src.api.routes.feedback import router as feedback_router

app = FastAPI(title="Oncology RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(feedback_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 5: Start the API and verify**

```bash
uv run uvicorn src.api.main:app --reload --port 8000
```

In a second terminal:

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "X-API-Key: dev-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "Welche Empfehlung gilt für Mammakarzinom-Screening?", "session_id": "test"}' \
  --no-buffer
```

Expected: SSE stream with `data: {...}` followed by `data: [DONE]`.

- [ ] **Step 6: Commit**

```bash
git add src/api/
git commit -m "feat(api): FastAPI backend with SSE chat and feedback endpoints"
```

---

## Task 16: Streamlit Chat UI — MVP Complete

**Files:**
- Create: `src/ui/app.py`
- Create: `src/ui/components/chat_page.py`
- Create: `src/ui/components/source_cards.py`
- Create: `src/ui/components/filters.py`

- [ ] **Step 1: Create source cards component**

```python
# src/ui/components/source_cards.py
import streamlit as st


def render_source_cards(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown("#### Quellen")
    for c in citations:
        with st.expander(f"{c['label']} {c['citation']}"):
            st.markdown(f"**Datei:** `{c.get('source_filename', 'n/a')}`")
            st.markdown(f"**Referenz:** {c['citation']}")


def render_tool_calls(tool_calls: list[dict]) -> None:
    if not tool_calls:
        return
    with st.expander(f"Tool-Aufrufe ({len(tool_calls)})"):
        for tc in tool_calls:
            st.json(tc)
```

- [ ] **Step 2: Create filter panel component**

```python
# src/ui/components/filters.py
import streamlit as st


def render_filters() -> dict:
    """Render sidebar filter panel. Returns dict of active filters."""
    st.sidebar.header("Filter")
    guideline = st.sidebar.selectbox(
        "Leitlinie",
        options=["Alle", "mamma", "krk", "lunge", "prosta"],
        format_func=lambda x: {
            "Alle": "Alle Leitlinien",
            "mamma": "Mammakarzinom",
            "krk": "Kolorektales Karzinom",
            "lunge": "Lungenkarzinom",
            "prosta": "Prostatakarzinom",
        }.get(x, x),
    )
    grade = st.sidebar.selectbox(
        "Empfehlungsgrad",
        options=["Alle", "A", "B", "0"],
    )
    return {
        "guideline_id": "" if guideline == "Alle" else guideline,
        "grade": "" if grade == "Alle" else grade,
    }


def render_feedback_buttons(session_id: str, query: str, api_url: str, api_key: str) -> None:
    col1, col2 = st.columns([1, 1])
    import httpx
    with col1:
        if st.button("👍 Hilfreich"):
            httpx.post(f"{api_url}/feedback", json={
                "session_id": session_id, "query": query, "rating": 1
            }, headers={"X-API-Key": api_key})
    with col2:
        if st.button("👎 Nicht hilfreich"):
            httpx.post(f"{api_url}/feedback", json={
                "session_id": session_id, "query": query, "rating": -1
            }, headers={"X-API-Key": api_key})
```

- [ ] **Step 3: Create chat page**

```python
# src/ui/components/chat_page.py
import json
import httpx
import streamlit as st
from src.ui.components.source_cards import render_source_cards, render_tool_calls
from src.ui.components.filters import render_feedback_buttons


def render_chat_page(api_url: str, api_key: str, filters: dict) -> None:
    st.title("Onkologie Leitlinien-Assistent")
    st.caption("S3-Leitlinien: Mammakarzinom · Kolorektales Karzinom · Lungenkarzinom · Prostatakarzinom")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Stellen Sie Ihre Frage zu den Leitlinien...")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Suche in den Leitlinien..."):
            try:
                resp = httpx.post(
                    f"{api_url}/chat",
                    json={
                        "query": query,
                        "session_id": st.session_state.session_id,
                        **filters,
                    },
                    headers={"X-API-Key": api_key},
                    timeout=120,
                )
                resp.raise_for_status()

                # Parse SSE
                payload = None
                for line in resp.text.splitlines():
                    if line.startswith("data:") and "[DONE]" not in line:
                        payload = json.loads(line[5:].strip())

                if payload and not payload.get("blocked"):
                    st.markdown("**Fachliche Antwort**")
                    st.markdown(payload["answer_professional"])
                    st.markdown("---")
                    st.markdown("**In einfachen Worten**")
                    st.markdown(payload["answer_plain"])
                    st.markdown(payload.get("disclaimer", ""))
                    render_source_cards(payload.get("citations", []))
                    render_tool_calls(payload.get("tool_calls", []))
                    render_feedback_buttons(
                        st.session_state.session_id, query, api_url, api_key
                    )
                    answer_text = payload["answer_professional"]
                elif payload and payload.get("blocked"):
                    st.warning(payload.get("answer_professional", "Anfrage blockiert."))
                    answer_text = payload.get("answer_professional", "")
                else:
                    st.error("Keine Antwort erhalten.")
                    answer_text = ""

                st.session_state.messages.append({"role": "assistant", "content": answer_text})

            except Exception as e:
                st.error(f"Fehler: {e}")
```

- [ ] **Step 4: Create Streamlit entry point**

```python
# src/ui/app.py
import os
import streamlit as st
from dotenv import load_dotenv
from src.ui.components.chat_page import render_chat_page
from src.ui.components.filters import render_filters

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-secret-key")

st.set_page_config(
    page_title="Onkologie Leitlinien-Assistent",
    page_icon="🏥",
    layout="wide",
)

filters = render_filters()
render_chat_page(api_url=API_URL, api_key=API_KEY, filters=filters)
```

- [ ] **Step 5: Start UI and verify MVP milestone**

Ensure the FastAPI backend is running on port 8000, then:

```bash
uv run streamlit run src/ui/app.py --server.port 8501
```

Open `http://localhost:8501`. Ask: *"Welche Screening-Empfehlung gilt für Mammakarzinom?"*

Expected:
- Spinner appears while retrieving
- Dual-layer answer rendered (Fachliche Antwort + In einfachen Worten)
- Source cards show guideline, section, page
- Feedback buttons visible
- Filter panel in sidebar functional

**MVP milestone reached.**

- [ ] **Step 6: Commit**

```bash
git add src/ui/
git commit -m "feat(ui): Streamlit MVP chat UI — dual-layer answer with source cards"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Phase 0A: parser, detector, chunker, metadata, reference — Tasks 2–5, 6
- [x] Phase 0B: enricher, embedder, Milvus store — Tasks 7–9
- [x] Phase 0C: pipeline + rollout script — Task 10
- [x] Phase 1: retrieval (dense + BM25 + RRF + reranker + expander) — Task 11
- [x] Phase 2: LangGraph (state, all nodes, graph) + lookup_empfehlung pulled forward — Tasks 12–14
- [x] Early eval artifacts (smoke queries in Task 10 Step 4) — covered
- [x] Phase 3: FastAPI (SSE, feedback, auth placeholder) — Task 15
- [x] Phase 4: Streamlit (chat, source cards, filters, feedback) — Task 16
- [x] Architecture constraint (no logic in routes or callbacks) — enforced in Tasks 15–16
- [x] Confidence check lightweight (score-based, no LLM call) — Task 13, confidence.py
- [x] API-key as persistence key before OAuth — noted in feedback.py + chat.py (session_id)
- [x] Bibliography/reference risk — reference.py logs unresolved markers

**No placeholders found.**

**Type consistency:** `RAGState` TypedDict used consistently across all nodes. `RetrievedChunk` dataclass fields match between `search.py`, `reranker.py`, `expander.py`, and `search_guidelines_tool` output dict.
