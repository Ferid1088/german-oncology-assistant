"""Bibliography reference extraction and resolution for guideline PDFs.

Provides three utilities:
- ``extract_inline_refs`` — finds all ``[N]`` citation markers in a text block.
- ``parse_bibliography`` — parses the reference list section into ``ReferenceEntry`` objects.
- ``resolve_refs`` — cross-links inline citation IDs to their full bibliography entries.
"""

import re
from dataclasses import dataclass

INLINE_REF_RE = re.compile(r"(?<!\w)\[(\d+(?:,\s*\d+)*)\]")
BIB_ENTRY_RE = re.compile(r"^\[(\d+)\]\s+(.+)$")
PUBMED_URL_RE = re.compile(r"https?://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")


@dataclass
class ReferenceEntry:
    """A single bibliography entry parsed from a guideline PDF.

    Attributes:
        reference_id: The numeric string from the ``[N]`` bracket (e.g. "42").
        raw_text: Full raw text of the bibliography line after the ``[N]`` prefix.
        pubmed_id: PubMed article ID extracted from a pubmed.ncbi.nlm.nih.gov URL, or "".
        pubmed_url: Full PubMed URL found in ``raw_text``, or "".
        unresolved: True when an inline citation ID has no matching bibliography entry.
    """

    reference_id: str
    raw_text: str
    pubmed_id: str = ""
    pubmed_url: str = ""
    unresolved: bool = False


def extract_inline_refs(text: str) -> list[str]:
    """Return list of all cited reference IDs found in text. Duplicates are preserved."""
    ids: list[str] = []
    for match in INLINE_REF_RE.finditer(text):
        for ref_id in match.group(1).split(","):
            ids.append(ref_id.strip())
    return ids


def parse_bibliography(text: str) -> list[ReferenceEntry]:
    """Extract structured bibliography entries from document text."""
    entries: list[ReferenceEntry] = []
    for line in text.splitlines():
        line = line.strip()
        m = BIB_ENTRY_RE.match(line)
        if m:
            ref_id = m.group(1)
            raw = m.group(2)
            pubmed_url = ""
            pubmed_id = ""
            url_m = PUBMED_URL_RE.search(raw)
            if url_m:
                pubmed_url = url_m.group(0)
                pubmed_id = url_m.group(1)
            entries.append(ReferenceEntry(
                reference_id=ref_id,
                raw_text=raw,
                pubmed_url=pubmed_url,
                pubmed_id=pubmed_id,
            ))
    return entries


def resolve_refs(
    inline_ids: list[str], entries: list[ReferenceEntry]
) -> list[ReferenceEntry]:
    """Cross-link inline citation IDs against bibliography entries.

    Returns a list of ReferenceEntry objects for each inline_id.
    IDs with no matching entry get a stub ReferenceEntry with unresolved=True.
    """
    index = {e.reference_id: e for e in entries}
    result: list[ReferenceEntry] = []
    for ref_id in inline_ids:
        if ref_id in index:
            result.append(index[ref_id])
        else:
            result.append(ReferenceEntry(
                reference_id=ref_id,
                raw_text="",
                unresolved=True,
            ))
    return result
