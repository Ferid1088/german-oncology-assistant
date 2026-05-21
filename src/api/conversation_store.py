from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _later_timestamp(previous: str) -> str:
    previous_dt = datetime.fromisoformat(previous)
    candidate = datetime.now(timezone.utc)
    if candidate <= previous_dt:
        candidate = previous_dt.replace(microsecond=previous_dt.microsecond + 1)
    return candidate.isoformat()


def _to_json(value) -> str:
    return json.dumps(value or [], ensure_ascii=False)


def _from_json(value: str | None, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _rewritten_query_from_rag_trace(rag_trace: object) -> str:
    steps = rag_trace if isinstance(rag_trace, list) else []
    for step in reversed(steps):
        if not isinstance(step, dict) or step.get("name") != "rewrite":
            continue
        details = step.get("details", {})
        if not isinstance(details, dict):
            continue
        rewritten = details.get("rewritten_query")
        if isinstance(rewritten, str) and rewritten.strip():
            return rewritten.strip()
    return ""


class ConversationStore:
    def __init__(self, db_path: str | Path | None = None):
        configured = db_path or os.getenv("CONVERSATION_DB_PATH", "data/app_state.db")
        self.db_path = Path(configured)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'Neue Konversation',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    deleted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    answer_professional TEXT NOT NULL DEFAULT '',
                    answer_plain TEXT NOT NULL DEFAULT '',
                    citations TEXT NOT NULL DEFAULT '[]',
                    retrieved_chunks TEXT NOT NULL DEFAULT '[]',
                    tool_calls TEXT NOT NULL DEFAULT '[]',
                    rag_trace TEXT NOT NULL DEFAULT '[]',
                    token_usage TEXT NOT NULL DEFAULT '{}',
                    external_search_snippets TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
                ON messages (conversation_id, created_at);
                """
            )
            message_columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
            if "rag_trace" not in message_columns:
                conn.execute("ALTER TABLE messages ADD COLUMN rag_trace TEXT NOT NULL DEFAULT '[]'")
            if "token_usage" not in message_columns:
                conn.execute("ALTER TABLE messages ADD COLUMN token_usage TEXT NOT NULL DEFAULT '{}' ")
            if "external_search_snippets" not in message_columns:
                conn.execute("ALTER TABLE messages ADD COLUMN external_search_snippets TEXT NOT NULL DEFAULT '[]'")

    def _conversation_row(self, row: sqlite3.Row) -> dict:
        return {
            "session_id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _list_message_rows(self, conn: sqlite3.Connection, conversation_id: str) -> list[sqlite3.Row]:
        return conn.execute(
            """
            SELECT id, role, content, answer_professional, answer_plain, citations,
                   retrieved_chunks, tool_calls, rag_trace, token_usage, external_search_snippets, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (conversation_id,),
        ).fetchall()

    def _message_dict(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "answer_professional": row["answer_professional"],
            "answer_plain": row["answer_plain"],
            "citations": _from_json(row["citations"], []),
            "retrieved_chunks": _from_json(row["retrieved_chunks"], []),
            "tool_calls": _from_json(row["tool_calls"], []),
            "rag_trace": _from_json(row["rag_trace"], []),
            "token_usage": _from_json(row["token_usage"], {}),
            "external_search_snippets": _from_json(row["external_search_snippets"], []),
            "created_at": row["created_at"],
        }

    def list_conversations(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                WHERE deleted_at IS NULL
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()

            conversations: list[dict] = []
            for row in rows:
                conversation = self._conversation_row(row)
                conversation["messages"] = [
                    {"role": message["role"], "content": message["content"]}
                    for message in self._list_message_rows(conn, row["id"])
                ]
                conversations.append(conversation)
            return conversations

    def list_conversations_detailed(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                WHERE deleted_at IS NULL
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()

            conversations: list[dict] = []
            for row in rows:
                conversation = self._conversation_row(row)
                conversation["messages"] = [
                    self._message_dict(message)
                    for message in self._list_message_rows(conn, row["id"])
                ]
                conversations.append(conversation)
            return conversations

    def get_conversation(self, conversation_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                WHERE id = ? AND deleted_at IS NULL
                """,
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None

            conversation = self._conversation_row(row)
            conversation["messages"] = [
                {"role": message["role"], "content": message["content"]}
                for message in self._list_message_rows(conn, conversation_id)
            ]
            return conversation

    def export_conversation(self, conversation_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                WHERE id = ? AND deleted_at IS NULL
                """,
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None

            payload = self._conversation_row(row)
            payload["messages"] = [self._message_dict(message) for message in self._list_message_rows(conn, conversation_id)]
            return payload

    def create_conversation(self, conversation_id: str | None = None, title: str = "Neue Konversation") -> dict:
        conversation_id = conversation_id or str(uuid.uuid4())
        now = _utcnow()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM conversations WHERE id = ? AND deleted_at IS NULL",
                (conversation_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO conversations (id, title, created_at, updated_at, deleted_at)
                    VALUES (?, ?, ?, ?, NULL)
                    """,
                    (conversation_id, title, now, now),
                )
            return self.get_conversation(conversation_id) or {
                "session_id": conversation_id,
                "title": title,
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connect() as conn:
            result = conn.execute(
                "UPDATE conversations SET deleted_at = ?, updated_at = ? WHERE id = ? AND deleted_at IS NULL",
                (_utcnow(), _utcnow(), conversation_id),
            )
            return result.rowcount > 0

    def load_session_memory(self, conversation_id: str) -> dict:
        with self._connect() as conn:
            conversation = conn.execute(
                "SELECT id FROM conversations WHERE id = ? AND deleted_at IS NULL",
                (conversation_id,),
            ).fetchone()
            if conversation is None:
                return {
                    "messages": [],
                    "prior_answer_professional": "",
                    "prior_answer_plain": "",
                    "prior_citations": [],
                    "prior_retrieved_chunks": [],
                    "prior_rewritten_query": "",
                    "prior_rag_trace": [],
                    "prior_external_search_snippets": [],
                }

            rows = self._list_message_rows(conn, conversation_id)
            history = []
            prior_answer_professional = ""
            prior_answer_plain = ""
            prior_citations: list[dict] = []
            prior_retrieved_chunks: list[dict] = []
            prior_rewritten_query = ""
            prior_rag_trace: list[dict] = []
            prior_external_search_snippets: list[dict] = []

            for row in rows:
                if row["role"] == "user":
                    history.append(HumanMessage(content=row["content"]))
                elif row["role"] == "assistant":
                    history.append(AIMessage(content=row["content"]))
                    prior_answer_professional = row["answer_professional"] or ""
                    prior_answer_plain = row["answer_plain"] or ""
                    prior_citations = _from_json(row["citations"], [])
                    prior_retrieved_chunks = _from_json(row["retrieved_chunks"], [])
                    prior_rag_trace = _from_json(row["rag_trace"], [])
                    prior_rewritten_query = _rewritten_query_from_rag_trace(prior_rag_trace)
                    prior_external_search_snippets = _from_json(row["external_search_snippets"], [])

            return {
                "messages": history,
                "prior_answer_professional": prior_answer_professional,
                "prior_answer_plain": prior_answer_plain,
                "prior_citations": prior_citations,
                "prior_retrieved_chunks": prior_retrieved_chunks,
                "prior_rewritten_query": prior_rewritten_query,
                "prior_rag_trace": prior_rag_trace,
                "prior_external_search_snippets": prior_external_search_snippets,
            }

    def append_turn(self, conversation_id: str, user_query: str, final_state: dict, combined_answer: str) -> None:
        user_created_at = _utcnow()
        assistant_created_at = _later_timestamp(user_created_at)
        updated_at = assistant_created_at
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, title,
                       (SELECT COUNT(*) FROM messages WHERE conversation_id = conversations.id) AS message_count
                FROM conversations
                WHERE id = ? AND deleted_at IS NULL
                """,
                (conversation_id,),
            ).fetchone()

            if row is None:
                conn.execute(
                    """
                    INSERT INTO conversations (id, title, created_at, updated_at, deleted_at)
                    VALUES (?, ?, ?, ?, NULL)
                    """,
                    (conversation_id, self._title_from_query(user_query), user_created_at, updated_at),
                )
                message_count = 0
                current_title = self._title_from_query(user_query)
            else:
                message_count = row["message_count"]
                current_title = row["title"]

            if message_count == 0 and current_title == "Neue Konversation":
                conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (self._title_from_query(user_query), updated_at, conversation_id),
                )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (updated_at, conversation_id),
                )

            conn.executemany(
                """
                INSERT INTO messages (
                    id, conversation_id, role, content, answer_professional, answer_plain,
                    citations, retrieved_chunks, tool_calls, rag_trace, token_usage, external_search_snippets, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        str(uuid.uuid4()),
                        conversation_id,
                        "user",
                        user_query,
                        "",
                        "",
                        "[]",
                        "[]",
                        "[]",
                        "[]",
                        "{}",
                        "[]",
                        user_created_at,
                    ),
                    (
                        str(uuid.uuid4()),
                        conversation_id,
                        "assistant",
                        combined_answer,
                        final_state.get("answer_professional", ""),
                        final_state.get("answer_plain", ""),
                        _to_json(final_state.get("citations", [])),
                        _to_json(final_state.get("retrieved_chunks", [])),
                        _to_json(final_state.get("tool_calls_log", [])),
                        _to_json(final_state.get("rag_trace", [])),
                        json.dumps(final_state.get("token_usage", {}), ensure_ascii=False),
                        _to_json(final_state.get("external_search_snippets", [])),
                        assistant_created_at,
                    ),
                ],
            )

    @staticmethod
    def _title_from_query(query: str) -> str:
        query = (query or "").strip()
        if not query:
            return "Neue Konversation"
        return query[:40] + ("..." if len(query) > 40 else "")


_STORE: ConversationStore | None = None


def get_conversation_store() -> ConversationStore:
    global _STORE
    if _STORE is None:
        _STORE = ConversationStore()
    return _STORE