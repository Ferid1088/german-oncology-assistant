# scripts/run_indexer.py
"""CLI: python scripts/run_indexer.py [--pdf mammakarzinom_v4.4.pdf] [--dry-run]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from src.indexer.pipeline import index_pdf, GUIDELINE_MAP
from src.indexer.store import MilvusStore
from src.retrieval.bm25 import build_bm25_index, reload_bm25_index
from pymilvus import MilvusClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

MILVUS_URI = "./milvus.db"
COLLECTION = "oncology_guidelines"


def _rebuild_bm25(dry_run: bool) -> None:
    """Query all leaf chunks from Milvus and rebuild the on-disk BM25 pickle index.

    Paginates Milvus results at 1000 records per page to avoid memory spikes on large
    corpora.  After building, calls ``reload_bm25_index`` so the in-process singleton
    is updated immediately without restarting the API.

    Args:
        dry_run: When True, skips the BM25 rebuild entirely (used for parse/chunk tests).
    """
    if dry_run:
        return
    print("Building BM25 index from Milvus corpus...")
    c = MilvusClient(uri=MILVUS_URI)
    c.load_collection(COLLECTION)
    all_chunks: list[dict] = []
    offset = 0
    while True:
        batch = c.query(
            collection_name=COLLECTION,
            filter="is_leaf == true",
            output_fields=["chunk_id", "text"],
            limit=1000,
            offset=offset,
        )
        if not batch:
            break
        all_chunks.extend(batch)
        offset += len(batch)
        if len(batch) < 1000:
            break
    build_bm25_index(all_chunks)
    reload_bm25_index()
    print(f"BM25 index built from {len(all_chunks)} chunks.")


def main():
    """CLI entry point for the guideline indexing pipeline.

    Usage examples::

        python scripts/run_indexer.py                          # index all PDFs in GUIDELINE_MAP
        python scripts/run_indexer.py --pdf mammakarzinom.pdf  # index one guideline
        python scripts/run_indexer.py --reset                  # drop collection first, then re-index
        python scripts/run_indexer.py --dry-run                # parse/chunk only, no Milvus writes
        python scripts/run_indexer.py --no-enrich              # skip LLM enrichment (faster)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="Single PDF filename to index (from GUIDELINE_MAP)")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk but do not write to Milvus")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collection before indexing")
    parser.add_argument("--no-enrich", action="store_true", help="Skip LLM enrichment (faster, cheaper re-index)")
    args = parser.parse_args()

    kb_dir = Path("docs/knowledge_base")
    store = MilvusStore()

    if args.reset:
        print("Dropping existing collection...")
        store.drop()
        print("Collection dropped.")

    pdfs = [args.pdf] if args.pdf else list(GUIDELINE_MAP.keys())
    for pdf_name in pdfs:
        pdf_path = kb_dir / pdf_name
        if not pdf_path.exists():
            print(f"WARNING: {pdf_path} not found, skipping")
            continue
        count = index_pdf(pdf_path, store, dry_run=args.dry_run, enrich=not args.no_enrich)
        print(f"Indexed {count} chunks from {pdf_name}")

    _rebuild_bm25(args.dry_run)


if __name__ == "__main__":
    main()
