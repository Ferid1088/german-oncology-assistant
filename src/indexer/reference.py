import re
from dataclasses import dataclass

INLINE_REF_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")
BIB_ENTRY_RE = re.compile(r"^\[?(\d+)\]?\s+(.+)$")
PUBMED_URL_RE = re.compile(r"https?://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")


@dataclass
class ReferenceEntry:
    reference_id: str
    raw_text: str
    pubmed_id: str = ""
    pubmed_url: str = ""
    unresolved: bool = False


def extract_inline_refs(text: str) -> list[str]:
    """Return list of all cited reference IDs found in text."""
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
        if m and len(line) > 10:  # avoid false positives on short lines
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
