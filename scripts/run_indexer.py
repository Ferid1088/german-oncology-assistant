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
