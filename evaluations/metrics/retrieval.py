"""Retrieval-grounding metrics against gold chunks, sections, and pages."""

from __future__ import annotations


def _chunk_ids(records: list[dict]) -> list[str]:
    return [str(record.get("chunk_id", "")).strip() for record in records if str(record.get("chunk_id", "")).strip()]


def _section_labels(records: list[dict]) -> set[str]:
    labels: set[str] = set()
    for record in records:
        for section in record.get("section_path", []) or []:
            if str(section).strip():
                labels.add(str(section).strip())
        section_title = str(record.get("section_title") or "").strip()
        if section_title:
            labels.add(section_title)
    return labels


def _page_values(records: list[dict]) -> set[int]:
    pages: set[int] = set()
    for record in records:
        page_numbers = record.get("page_numbers") or []
        if isinstance(page_numbers, list):
            for page in page_numbers:
                try:
                    pages.add(int(page))
                except (TypeError, ValueError):
                    continue

        for key in ("page_start", "page_end"):
            value = record.get(key)
            try:
                if value is not None:
                    pages.add(int(value))
            except (TypeError, ValueError):
                continue
    return pages


def _safe_ratio(hit_count: int, total_count: int) -> float | None:
    if total_count <= 0:
        return None
    return round(hit_count / total_count, 4)


def _retrieved_records(resp: dict) -> list[dict]:
    records = resp.get("retrieved_chunks") or []
    if isinstance(records, list) and records:
        return [record for record in records if isinstance(record, dict)]
    citations = resp.get("citations") or []
    return [record for record in citations if isinstance(record, dict)]


def chunk_recall(item: dict, resp: dict) -> float | None:
    gold = {str(value) for value in item.get("gold_chunk_ids", []) if str(value) and not str(value).startswith("seed:")}
    if not gold:
        return None
    retrieved = set(_chunk_ids(_retrieved_records(resp)))
    return _safe_ratio(len(gold & retrieved), len(gold))


def chunk_precision(item: dict, resp: dict) -> float | None:
    gold = {str(value) for value in item.get("gold_chunk_ids", []) if str(value) and not str(value).startswith("seed:")}
    if not gold:
        return None
    retrieved = set(_chunk_ids(_retrieved_records(resp)))
    if not retrieved:
        return 0.0
    return _safe_ratio(len(gold & retrieved), len(retrieved))


def top_gold_chunk_hit(item: dict, resp: dict) -> bool | None:
    gold = {str(value) for value in item.get("gold_chunk_ids", []) if str(value) and not str(value).startswith("seed:")}
    if not gold:
        return None
    top_ids = _chunk_ids(_retrieved_records(resp)[:5])
    return any(chunk_id in gold for chunk_id in top_ids)


def section_recall(item: dict, resp: dict) -> float | None:
    gold = {str(value).strip() for value in item.get("gold_sections", []) if str(value).strip()}
    if not gold:
        return None
    retrieved = _section_labels(_retrieved_records(resp))
    return _safe_ratio(len(gold & retrieved), len(gold))


def page_recall(item: dict, resp: dict) -> float | None:
    gold = set()
    for value in item.get("gold_pages", []) or []:
        try:
            gold.add(int(value))
        except (TypeError, ValueError):
            continue
    if not gold:
        return None
    retrieved = _page_values(_retrieved_records(resp))
    return _safe_ratio(len(gold & retrieved), len(gold))


def citation_chunk_recall(item: dict, resp: dict) -> float | None:
    must_chunks = {
        str(citation.get("chunk_id"))
        for citation in item.get("required_citations", [])
        if citation.get("citation_importance") == "must"
        and citation.get("chunk_id")
        and not str(citation.get("chunk_id")).startswith("seed:")
    }
    if not must_chunks:
        return None
    cited = {str(citation.get("chunk_id")) for citation in resp.get("citations", []) if citation.get("chunk_id")}
    return _safe_ratio(len(must_chunks & cited), len(must_chunks))


def compute_all(item: dict, resp: dict) -> dict:
    return {
        "retrieved_chunk_recall": chunk_recall(item, resp),
        "retrieved_chunk_precision": chunk_precision(item, resp),
        "top_gold_chunk_hit": top_gold_chunk_hit(item, resp),
        "retrieved_section_recall": section_recall(item, resp),
        "retrieved_page_recall": page_recall(item, resp),
        "citation_chunk_recall": citation_chunk_recall(item, resp),
    }
