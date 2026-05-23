"""Page number normalisation and German-language citation formatting utilities.

Used by the retrieval pipeline and answer generation node to produce consistent
``Seite``/``Seiten`` references in the style of German academic citations.
"""

from __future__ import annotations


def normalize_page_numbers(
    page_numbers: object = None,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[int]:
    """Return a sorted, deduplicated list of valid page numbers.

    Accepts either an explicit ``page_numbers`` list or a ``page_start``/
    ``page_end`` range (inclusive).  Filters out non-integer and sub-1 values.

    Args:
        page_numbers: Explicit list of page numbers (preferred).
        page_start: First page of the range (used as fallback).
        page_end: Last page of the range (inclusive); defaults to ``page_start``.

    Returns:
        Sorted list of unique page integers ≥ 1, or empty list if none valid.
    """
    normalized: list[int] = []

    if isinstance(page_numbers, list):
        for value in page_numbers:
            if isinstance(value, int) and value >= 1:
                normalized.append(value)

    if not normalized and isinstance(page_start, int) and page_start >= 1:
        if isinstance(page_end, int) and page_end >= page_start:
            normalized.extend(range(page_start, page_end + 1))
        else:
            normalized.append(page_start)

    deduped = sorted({value for value in normalized if value >= 1})
    return deduped


def merge_page_numbers(*page_groups: object) -> list[int]:
    """Merge multiple page number lists into a single sorted, deduplicated list.

    Used by the parent-chunk expander to combine the page ranges of a leaf chunk
    and its parent into a single citation range.

    Args:
        *page_groups: Any number of page-number lists (non-list values ignored).

    Returns:
        Sorted list of unique page integers ≥ 1.
    """
    merged: list[int] = []
    for group in page_groups:
        if isinstance(group, list):
            merged.extend(value for value in group if isinstance(value, int) and value >= 1)
    return sorted({value for value in merged if value >= 1})


def format_page_reference(
    page_numbers: object = None,
    page_start: int | None = None,
    page_end: int | None = None,
    *,
    short: bool = False,
) -> str | None:
    """Format a page reference as a German-language string.

    Examples:
        - Single page, long form: ``"Seite 42"``
        - Contiguous range, short form: ``"S. 42–45"``
        - Non-contiguous, long form: ``"Seiten 42, 44, 47"``

    Args:
        page_numbers: Explicit list of page numbers (preferred).
        page_start: First page (used as fallback).
        page_end: Last page (used as fallback).
        short: If True, use abbreviated prefix ``"S."`` instead of ``"Seite/Seiten"``.

    Returns:
        Formatted string, or ``None`` if no valid page numbers are available.
    """
    numbers = normalize_page_numbers(page_numbers, page_start, page_end)
    if not numbers:
        return None

    prefix_single = "S." if short else "Seite"
    prefix_multi = "S." if short else "Seiten"

    if len(numbers) == 1:
        return f"{prefix_single} {numbers[0]}"

    # Check whether pages form an unbroken sequence to decide between range and list format.
    is_contiguous = numbers == list(range(numbers[0], numbers[-1] + 1))
    if is_contiguous:
        return f"{prefix_multi} {numbers[0]}–{numbers[-1]}"

    return f"{prefix_multi} {', '.join(str(value) for value in numbers)}"