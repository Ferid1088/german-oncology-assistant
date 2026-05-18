from src.indexer.chunker import build_chunks, Chunk


def test_empfehlung_becomes_own_leaf_chunk(sample_text):
    chunks = build_chunks("mamma", "1.0", sample_text)
    emp_chunks = [c for c in chunks if c.chunk_type == "recommendation"]
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
    leaf_chunks = [c for c in chunks if c.is_leaf and c.chunk_type == "section"]
    for c in leaf_chunks:
        # Approximate token count: len(text.split()) * 1.3
        approx_tokens = len(c.text.split()) * 1.3
        assert approx_tokens <= 800  # generous upper bound for test text


def test_chunk_index_in_parent_assigned_to_prose(sample_text):
    chunks = build_chunks("mamma", "1.0", sample_text)
    prose_leaves = [c for c in chunks if c.is_leaf and c.chunk_type == "section"]
    # At least some prose chunks should have a chunk_index assigned
    assert any(c.chunk_index_in_parent is not None for c in prose_leaves)


def test_root_chunk_id_tracks_top_level_section():
    text = "1 Top\n\n1.1 Sub\n\nText hier.\n"
    chunks = build_chunks("mamma", "1.0", text)
    # The depth-1 chunk has no root (it IS the root)
    top = next(c for c in chunks if not c.is_leaf and c.section_path == ["1"])
    leaves = [c for c in chunks if c.is_leaf]
    for leaf in leaves:
        assert leaf.root_chunk_id == top.chunk_id
