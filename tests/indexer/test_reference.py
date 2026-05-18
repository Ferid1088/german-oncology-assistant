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
