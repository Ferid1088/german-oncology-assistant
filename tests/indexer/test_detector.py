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


def test_detect_structure_empty_input():
    units = detect_structure("")
    assert units == []


def test_heading_not_matched_by_numeric_prose():
    # Clinical numbers must not become phantom headings
    units = detect_structure("45 Jahre alte Patientin.")
    headings = [u for u in units if u.kind == "heading"]
    assert len(headings) == 0


def test_bibliography_requires_brackets():
    # Unbracketed reference numbers must not be classified as bibliography
    units = detect_structure("45 Autor A et al. Titel. Journal 2020.")
    refs = [u for u in units if u.kind == "bibliography_entry"]
    assert len(refs) == 0


def test_grade_normalised_to_uppercase():
    text = "Empfehlung 2.3\nFrauen sollen beraten werden.\nEmpfehlungsgrad: b\n"
    units = detect_structure(text)
    emp = next(u for u in units if u.kind == "empfehlung")
    assert emp.recommendation_grade == "B"


def test_empfehlung_block_terminated_at_eof():
    # No trailing blank line — block must still be captured
    text = "Empfehlung 3.1\nText ohne Leerzeile am Ende.\nEmpfehlungsgrad: A"
    units = detect_structure(text)
    empfehlungen = [u for u in units if u.kind == "empfehlung"]
    assert len(empfehlungen) == 1
    assert empfehlungen[0].recommendation_grade == "A"
