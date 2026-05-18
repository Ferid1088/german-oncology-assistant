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
