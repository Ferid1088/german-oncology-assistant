from unittest.mock import MagicMock
from src.indexer.enricher import generate_contextual_header, generate_hypothetical_questions, extract_semantic_metadata


def _mock_client(response_text: str):
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=response_text))]
    )
    return client


def test_generate_contextual_header_returns_string():
    client = _mock_client("Dieser Abschnitt behandelt die Diagnose des Mammakarzinoms.")
    header = generate_contextual_header(
        client=client,
        chunk_text="Empfehlung 2.1: MRT ist indiziert.",
        section_path=["2", "2.1"],
        guideline_title="Mammakarzinom",
    )
    assert isinstance(header, str)
    assert len(header) > 0


def test_generate_hypothetical_questions_returns_list():
    client = _mock_client("Wann ist MRT indiziert?\nWelche Bildgebung empfohlen?")
    questions = generate_hypothetical_questions(
        client=client,
        chunk_text="MRT ist bei unklarem Befund indiziert.",
    )
    assert isinstance(questions, list)
    assert len(questions) >= 1


def test_extract_semantic_metadata_returns_dict():
    client = _mock_client('{"diseases": ["Mammakarzinom"], "drugs": [], "procedures": ["MRT"], "patient_subgroups": [], "risk_category": []}')
    meta = extract_semantic_metadata(
        client=client,
        chunk_text="MRT ist bei Mammakarzinom-Patientinnen indiziert.",
    )
    assert "diseases" in meta
    assert "Mammakarzinom" in meta["diseases"]


def test_semantic_metadata_returns_empty_on_parse_failure():
    client = _mock_client("invalid json {{")
    meta = extract_semantic_metadata(client=client, chunk_text="text")
    assert meta == {"diseases": [], "drugs": [], "procedures": [], "patient_subgroups": [], "risk_category": []}


from src.indexer.embedder import embed_texts

def test_embed_texts_returns_correct_shape():
    from unittest.mock import MagicMock
    client = MagicMock()
    client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 3072), MagicMock(embedding=[0.2] * 3072)]
    )
    result = embed_texts(["text one", "text two"], client=client)
    assert len(result) == 2
    assert len(result[0]) == 3072
