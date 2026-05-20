import re
import html as _html


def _tooltip_content(citation: dict) -> str:
    parts = []
    if citation.get("label"):
        parts.append(f"<b>{_html.escape(citation['label'])}</b>")
    if citation.get("source_filename"):
        parts.append(f"<b>📄 {_html.escape(citation['source_filename'])}</b>")
    if citation.get("guideline_id"):
        parts.append(f"🏷 {_html.escape(citation['guideline_id'].upper())}")
    if citation.get("section_title"):
        parts.append(f"📑 {_html.escape(citation['section_title'])}")
    if citation.get("section_path"):
        path = citation["section_path"]
        if isinstance(path, list) and path:
            joined = " › ".join(_html.escape(str(p)) for p in path if p)
            parts.append(f"📍 {joined}")
    if citation.get("recommendation_id"):
        grade = citation.get("recommendation_grade", "")
        evidence = citation.get("evidence_level", "")
        rec_line = f"📋 Empfehlung {_html.escape(citation['recommendation_id'])}"
        if grade:
            rec_line += f" · Grad {_html.escape(grade)}"
        if evidence:
            rec_line += f" · LoE {_html.escape(evidence)}"
        parts.append(rec_line)
    if citation.get("page_start"):
        page = f"Seite {citation['page_start']}"
        if citation.get("page_end") and citation["page_end"] != citation["page_start"]:
            page += f"–{citation['page_end']}"
        parts.append(f"🔖 {_html.escape(page)}")
    if citation.get("reference_ids"):
        refs = citation["reference_ids"]
        if isinstance(refs, list) and refs:
            parts.append(f"📚 Referenzen: {_html.escape(', '.join(str(r) for r in refs))}")
    if citation.get("contextual_header"):
        parts.append(f"🧭 {_html.escape(citation['contextual_header'])}")
    return "<br>".join(parts) if parts else _html.escape(citation.get("citation", ""))


def annotate_citations(text: str, citations: list[dict]) -> str:
    """Replace [N] in markdown text with hoverable HTML citation spans."""
    index: dict[int, dict] = {}
    for c in citations:
        m = re.match(r'\[(\d+)\]', c.get("label", ""))
        if m:
            index[int(m.group(1))] = c

    def _replace(match: re.Match) -> str:
        n = int(match.group(1))
        if n not in index:
            return match.group(0)
        tip = _tooltip_content(index[n])
        return (
            f'<span class="cit-wrap">'
            f'<span class="cit-num">[{n}]</span>'
            f'<span class="cit-tip">{tip}</span>'
            f'</span>'
        )

    return re.sub(r'\[(\d+)\]', _replace, text)
