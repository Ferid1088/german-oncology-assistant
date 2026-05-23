"""PubMed search tool: queries external literature via NCBI E-utilities.

Uses a two-call pattern required by the E-utilities API:
1. ``esearch`` — returns a list of PubMed IDs (PMIDs) matching the query.
2. ``esummary`` — fetches title, authors, date, and source for each PMID.

Results carry a ``PUBMED_DISCLOSURE`` label to make clear they are external
sources outside the S3 guideline corpus.
"""

import os

import httpx


PUBMED_DISCLOSURE = (
    "Quelle: U.S. National Library of Medicine (NLM) – PubMed. "
    "Diese Ergebnisse stammen aus externen Datenquellen außerhalb der deutschen S3-Leitlinien."
)


def pubmed_search_tool(query: str, max_results: int = 5) -> dict:
    """Very small PubMed E-utilities wrapper for external literature search."""
    max_results = max(1, min(max_results, 10))
    timeout = float(os.getenv("PUBMED_TIMEOUT", "20"))

    with httpx.Client(timeout=timeout) as client:
        search_resp = client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max_results,
            },
        )
        search_resp.raise_for_status()
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])

        if not ids:
            return {
                "query": query,
                "results": [],
                "disclosure": PUBMED_DISCLOSURE,
            }

        summary_resp = client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "json",
            },
        )
        summary_resp.raise_for_status()
        payload = summary_resp.json().get("result", {})

    results = []
    for pmid in ids:
        item = payload.get(pmid, {})
        results.append(
            {
                "pmid": pmid,
                "title": item.get("title", ""),
                "pubdate": item.get("pubdate", ""),
                "source": item.get("source", ""),
                "authors": [a.get("name", "") for a in item.get("authors", [])],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "disclosure": PUBMED_DISCLOSURE,
            }
        )

    return {
        "query": query,
        "results": results,
        "disclosure": PUBMED_DISCLOSURE,
    }