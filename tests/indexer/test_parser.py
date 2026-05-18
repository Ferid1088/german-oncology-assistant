import pytest
from src.indexer.parser import extract_pages, clean_text

def test_extract_pages_returns_list_of_page_dicts(sample_pdf_path):
    pages = extract_pages(sample_pdf_path)
    assert isinstance(pages, list)
    assert len(pages) > 0
    first = pages[0]
    assert "page_number" in first
    assert "text" in first
    assert isinstance(first["text"], str)
    assert len(first["text"]) > 0

def test_extract_pages_page_numbers_are_one_indexed(sample_pdf_path):
    pages = extract_pages(sample_pdf_path)
    assert pages[0]["page_number"] == 1

def test_clean_text_removes_trailing_whitespace():
    raw = "Empfehlung 1.1  \n  Text  \n"
    result = clean_text(raw)
    assert not any(line.endswith("  ") for line in result.splitlines())

def test_clean_text_repairs_german_hyphenation():
    raw = "Die Behand-\nlung erfolgt"
    result = clean_text(raw)
    assert "Behandlung" in result

def test_clean_text_merges_broken_paragraph_lines():
    raw = "Dies ist ein langer\nSatz der weitergeht."
    result = clean_text(raw)
    assert "langer Satz" in result

def test_clean_text_preserves_section_numbers():
    raw = "1.2 Diagnostik\nText"
    result = clean_text(raw)
    assert "1.2 Diagnostik" in result
