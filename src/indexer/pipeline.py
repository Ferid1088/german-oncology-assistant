# src/indexer/pipeline.py
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

from src.indexer.parser import extract_pages, clean_text, normalize_recommendations
from src.indexer.chunker import build_chunks, PageBoundary
from src.indexer.metadata import attach_metadata
from src.indexer.reference import extract_inline_refs, parse_bibliography
from src.indexer.embedder import embed_texts
from src.indexer.store import MilvusStore

load_dotenv()
log = logging.getLogger(__name__)


GUIDELINE_MAP = {
    "mammakarzinom_v4.4.pdf":    ("mamma",  "4.4", "S3-Leitlinie Mammakarzinom"),
    "kolorektales_v3.0.pdf":     ("krk",    "3.0", "S3-Leitlinie Kolorektales Karzinom"),
    "lungenkarzinom_v4.0.pdf":   ("lunge",  "4.0", "S3-Leitlinie Lungenkarzinom"),
    "prostatakarzinom_v8.0.pdf": ("prosta", "8.0", "S3-Leitlinie Prostatakarzinom"),
    "what-is-cancer.pdf":        ("cancer_basics", "1.0", "What Is Cancer"),
}


def _build_page_boundaries(pages: list[dict]) -> tuple[str, list[PageBoundary]]:
    """
    Join per-page cleaned texts and build a list of (line_start, doc_page_number)
    boundaries so the chunker can assign the correct printed page to each chunk.
    Falls back to physical PDF page number when the header number isn't found.
    """
    page_texts: list[str] = []
    boundaries: list[PageBoundary] = []
    current_line = 0
    for p in pages:
        cleaned = normalize_recommendations(clean_text(p["text"]))
        page_num = p.get("doc_page_number") or p["page_number"]
        boundaries.append((current_line, page_num))
        line_count = cleaned.count("\n") + 1
        page_texts.append(cleaned)
        current_line += line_count + 1  # +1 for the joining "\n"
    return "\n".join(page_texts), boundaries


def index_pdf(pdf_path: Path, store: MilvusStore, dry_run: bool = False, enrich: bool = True) -> int:
    """Index a single PDF. Returns number of leaf chunks indexed."""
    filename = pdf_path.name
    guideline_id, version, title = GUIDELINE_MAP[filename]
    log.info("Indexing %s (enrich=%s)", filename, enrich)

    pages = extract_pages(pdf_path)
    full_text, page_boundaries = _build_page_boundaries(pages)

    # Build chunks from full text with per-chunk page assignment
    chunks = build_chunks(guideline_id, version, full_text, page_boundaries=page_boundaries)
    chunks = attach_metadata(chunks, source_filename=filename)

    # Parse bibliography from full text for reference linking
    bib_entries = parse_bibliography(full_text)
    bib_by_id = {e.reference_id: e for e in bib_entries}

    records = []
    leaf_chunks = [c for c in chunks if c.is_leaf]

    if enrich:
        from src.indexer.enricher import (
            generate_contextual_header,
            generate_hypothetical_questions,
            extract_semantic_metadata,
        )

    for chunk in leaf_chunks:
        # Reference linking
        inline_refs = extract_inline_refs(chunk.text)
        resolved = [r for r in inline_refs if r in bib_by_id]
        unresolved = [r for r in inline_refs if r not in bib_by_id]
        if unresolved:
            log.warning("Unresolved refs in chunk %s: %s", chunk.chunk_id, unresolved)

        if enrich:
            header = generate_contextual_header(
                chunk_text=chunk.text,
                section_path=chunk.section_path,
                guideline_title=title,
            )
            hypo_qs = generate_hypothetical_questions(chunk_text=chunk.text)
            semantic = extract_semantic_metadata(chunk_text=chunk.text)
            embed_input = f"{header}\n" + "\n".join(hypo_qs) + f"\n\n{chunk.text}"
        else:
            header = ""
            hypo_qs = []
            semantic = {}
            embed_input = chunk.text

        vector = embed_texts([embed_input])[0]

        record = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text[:16000],
            "dense_vector": vector,
            "guideline_id": chunk.guideline_id,
            "guideline_version": chunk.guideline_version,
            "chunk_type": chunk.chunk_type,
            "section_path": json.dumps(chunk.section_path),
            "section_title": chunk.section_title,
            "recommendation_id": chunk.recommendation_id,
            "recommendation_grade": chunk.recommendation_grade,
            "evidence_level": chunk.evidence_level,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
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
