from src.indexer.chunker import build_chunks
from src.indexer.metadata import attach_metadata

def test_metadata_adds_chunk_index(sample_text):
    chunks = build_chunks("mamma", "4.4", sample_text)
    chunks = attach_metadata(chunks, source_filename="mammakarzinom_v4.4.pdf")
    leaf_chunks = [c for c in chunks if c.is_leaf]
    indices = [c.chunk_index_in_parent for c in leaf_chunks]
    assert all(i is not None for i in indices)
    assert 0 in indices


def test_metadata_chunk_index_resets_per_parent():
    text = "1 Alpha\n\nProse unter Alpha.\n\n2 Beta\n\nProse unter Beta.\n"
    chunks = build_chunks("mamma", "1.0", text)
    chunks = attach_metadata(chunks, source_filename="test.pdf")
    # Find leaves under each section
    alpha_parent = next(c for c in chunks if not c.is_leaf and c.section_path == ["1"])
    beta_parent = next(c for c in chunks if not c.is_leaf and c.section_path == ["2"])
    alpha_leaves = [c for c in chunks if c.is_leaf and c.parent_chunk_id == alpha_parent.chunk_id]
    beta_leaves = [c for c in chunks if c.is_leaf and c.parent_chunk_id == beta_parent.chunk_id]
    # Each group starts from 0
    if alpha_leaves:
        assert min(c.chunk_index_in_parent for c in alpha_leaves) == 0
    if beta_leaves:
        assert min(c.chunk_index_in_parent for c in beta_leaves) == 0


def test_metadata_is_current_false(sample_text):
    chunks = build_chunks("mamma", "4.4", sample_text)
    chunks = attach_metadata(chunks, source_filename="old.pdf", is_current=False)
    assert all(c.is_current is False for c in chunks)

def test_metadata_adds_source_filename(sample_text):
    chunks = build_chunks("mamma", "4.4", sample_text)
    chunks = attach_metadata(chunks, source_filename="mammakarzinom_v4.4.pdf")
    assert all(c.source_filename == "mammakarzinom_v4.4.pdf" for c in chunks)

def test_metadata_marks_is_current(sample_text):
    chunks = build_chunks("mamma", "4.4", sample_text)
    chunks = attach_metadata(chunks, source_filename="mammakarzinom_v4.4.pdf")
    assert all(c.is_current is True for c in chunks)
