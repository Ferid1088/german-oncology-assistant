from __future__ import annotations

import csv
import io
import json
from datetime import datetime


def export_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def export_csv_bytes(payload: dict) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "conversation_id",
            "title",
            "created_at",
            "role",
            "content",
            "answer_professional",
            "answer_plain",
            "citations_count",
            "tool_calls_count",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_usd",
        ],
    )
    writer.writeheader()
    for message in payload.get("messages", []):
        token_usage = message.get("token_usage") or {}
        writer.writerow(
            {
                "conversation_id": payload.get("session_id"),
                "title": payload.get("title"),
                "created_at": message.get("created_at"),
                "role": message.get("role"),
                "content": message.get("content"),
                "answer_professional": message.get("answer_professional", ""),
                "answer_plain": message.get("answer_plain", ""),
                "citations_count": len(message.get("citations", [])),
                "tool_calls_count": len(message.get("tool_calls", [])),
                "input_tokens": token_usage.get("input_tokens", 0),
                "output_tokens": token_usage.get("output_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
                "cost_usd": token_usage.get("cost_usd", 0.0),
            }
        )
    return buffer.getvalue().encode("utf-8")


def _pdf_escape(text: str) -> str:
    safe = (text or "").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    safe = safe.encode("cp1252", errors="replace").decode("cp1252")
    return safe


def _pdf_line_chunks(lines: list[str], per_page: int = 42) -> list[list[str]]:
    return [lines[i:i + per_page] for i in range(0, len(lines), per_page)] or [[""]]


def export_pdf_bytes(payload: dict) -> bytes:
    lines: list[str] = []
    lines.append(f"Conversation: {payload.get('title', 'Conversation')}")
    lines.append(f"Session: {payload.get('session_id', '')}")
    lines.append(f"Exported at: {datetime.utcnow().isoformat()}Z")
    lines.append("")
    for message in payload.get("messages", []):
        role = str(message.get("role", "assistant")).upper()
        lines.append(f"{role}: {message.get('content', '')}")
        if message.get("citations"):
            lines.append(f"Citations: {len(message.get('citations', []))}")
        if message.get("tool_calls"):
            lines.append(f"Tool calls: {len(message.get('tool_calls', []))}")
        usage = message.get("token_usage") or {}
        if usage.get("total_tokens"):
            lines.append(
                f"Tokens: {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out / {usage.get('total_tokens', 0)} total"
            )
            lines.append(f"Cost (USD): {usage.get('cost_usd', 0.0)}")
        lines.append("")

    pages = _pdf_line_chunks(lines)
    objects: list[bytes] = []

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{4 + i * 2} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("latin-1"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index, page_lines in enumerate(pages):
        page_object_number = 4 + index * 2
        content_object_number = page_object_number + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_object_number} 0 R >>".encode(
                "latin-1"
            )
        )
        stream_lines = ["BT", "/F1 11 Tf", "50 790 Td", "14 TL"]
        for idx, line in enumerate(page_lines):
            escaped = _pdf_escape(line)
            if idx == 0:
                stream_lines.append(f"({escaped}) Tj")
            else:
                stream_lines.append(f"T* ({escaped}) Tj")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("cp1252", errors="replace")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream")

    buffer = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(buffer))
        buffer.extend(f"{idx} 0 obj\n".encode("latin-1"))
        buffer.extend(obj)
        buffer.extend(b"\nendobj\n")

    xref_offset = len(buffer)
    buffer.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    buffer.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        buffer.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    buffer.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode("latin-1")
    )
    return bytes(buffer)