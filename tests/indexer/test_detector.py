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
