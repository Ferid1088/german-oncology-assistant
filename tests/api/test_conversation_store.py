from src.api.conversation_store import ConversationStore


def test_conversation_store_persists_turns_and_soft_deletes(tmp_path):
    store = ConversationStore(tmp_path / "conversations.db")

    created = store.create_conversation("conv-1")
    assert created["session_id"] == "conv-1"
    assert created["messages"] == []

    final_state = {
        "answer_professional": "Fachliche Antwort",
        "answer_plain": "Einfache Antwort",
        "citations": [{"id": 1, "title": "Quelle"}],
        "retrieved_chunks": [{"id": "chunk-1"}],
        "tool_calls_log": [{"tool": "search_guidelines"}],
    }

    store.append_turn(
        conversation_id="conv-1",
        user_query="Was ist die Empfehlung?",
        final_state=final_state,
        combined_answer="Fachliche Antwort\n\nEinfache Antwort",
    )

    conversations = store.list_conversations()
    assert len(conversations) == 1
    assert conversations[0]["title"].startswith("Was ist die Empfehlung?")
    assert [message["role"] for message in conversations[0]["messages"]] == ["user", "assistant"]

    memory = store.load_session_memory("conv-1")
    assert len(memory["messages"]) == 2
    assert memory["prior_answer_professional"] == "Fachliche Antwort"
    assert memory["prior_answer_plain"] == "Einfache Antwort"
    assert memory["prior_citations"] == [{"id": 1, "title": "Quelle"}]
    assert memory["prior_retrieved_chunks"] == [{"id": "chunk-1"}]

    assert store.delete_conversation("conv-1") is True
    assert store.list_conversations() == []