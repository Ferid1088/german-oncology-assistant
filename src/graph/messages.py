from __future__ import annotations


def get_message_role(message) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "user"))

    msg_type = getattr(message, "type", None)
    if msg_type == "human":
        return "user"
    if msg_type == "ai":
        return "assistant"
    if msg_type == "tool":
        return "tool"

    role = getattr(message, "role", None)
    if role:
        return str(role)
    return "user"


def get_message_content(message) -> str:
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return " ".join(parts)
    return str(content) if content is not None else ""