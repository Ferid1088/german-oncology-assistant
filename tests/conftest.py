import pytest
from pathlib import Path

@pytest.fixture
def sample_pdf_path() -> Path:
    return Path(__file__).parent.parent / "docs/knowledge_base/mammakarzinom_v4.4.pdf"

@pytest.fixture
def sample_text() -> str:
    return """1 Einleitung

1.1 Informationen zu dieser Leitlinie

Die S3-Leitlinie Mammakarzinom wurde erstellt.

Empfehlung 1.1
Frauen mit erhöhtem familiären Risiko sollen eine genetische Beratung erhalten.
Empfehlungsgrad: A
Evidenzlevel: 1a

1.2 Hintergrund

Weiterer Text der Leitlinie.

[45] Autor A et al. Titel. Journal 2020;1:1-10.
[46] Autor B et al. Titel. Journal 2021;2:2-20.
"""

@pytest.fixture
def openrouter_client(mocker):
    """Mock OpenRouter client — prevents real API calls in unit tests."""
    mock = mocker.MagicMock()
    mock.chat.completions.create.return_value = mocker.MagicMock(
        choices=[mocker.MagicMock(message=mocker.MagicMock(content="mocked response"))]
    )
    mock.embeddings.create.return_value = mocker.MagicMock(
        data=[mocker.MagicMock(embedding=[0.1] * 3072)]
    )
    return mock
