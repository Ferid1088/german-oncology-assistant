from src.promt.loader import build_ambiguity_prompt_messages


def test_build_ambiguity_prompt_messages_loads_file_based_examples():
    messages = build_ambiguity_prompt_messages(
        history_block="user: Welche Therapie wird empfohlen?",
        query="Welche Chemotherapie wird empfohlen?",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Beispiel 1" in messages[1]["content"]
    assert "requires_clarification" in messages[1]["content"]
    assert "Welche Chemotherapie wird empfohlen?" in messages[1]["content"]