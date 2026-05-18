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
    assert len(entries) == 2

def test_unresolved_refs_flagged():
    text = "Gemäß [9999] ist es so."
    refs = extract_inline_refs(text)
    assert "9999" in refs  # extracted but no bib entry → flagged by pipeline


def test_parse_bibliography_no_false_positives(sample_text):
    entries = parse_bibliography(sample_text)
    ids = [e.reference_id for e in entries]
    # Section headings like "1 Einleitung" must not appear
    assert "1" not in ids
    assert len(entries) == 2  # only the two [45] and [46] entries


def test_resolve_refs_marks_unresolved():
    from src.indexer.reference import resolve_refs
    bib = [ReferenceEntry(reference_id="45", raw_text="Autor A et al.")]
    resolved = resolve_refs(["45", "9999"], bib)
    found = {r.reference_id: r for r in resolved}
    assert found["45"].unresolved is False
    assert found["9999"].unresolved is True


def test_parse_bibliography_pubmed_url():
    text = "[123] Han S et al. URL: https://pubmed.ncbi.nlm.nih.gov/34567890\n"
    entries = parse_bibliography(text)
    assert len(entries) == 1
    assert entries[0].pubmed_id == "34567890"
    assert "pubmed" in entries[0].pubmed_url
