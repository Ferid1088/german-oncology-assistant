"""Web search tool: fetches supplemental web snippets for a query.

Tries Google Custom Search first (requires ``GOOGLE_SEARCH_API_KEY`` and
``GOOGLE_SEARCH_ENGINE_ID`` environment variables).  Falls back to a
lightweight DuckDuckGo HTML scraper when Google credentials are absent or the
API call fails.  Results are labelled with a disclosure string reminding the
user that web snippets are supplemental and not part of the S3 guideline corpus.

This tool is called by the ``external_search`` node, not by the agent loop.
"""

from __future__ import annotations

import os
import re
from html import unescape
from urllib.parse import quote_plus

import httpx


WEB_SEARCH_DISCLOSURE = (
    "External web snippets are supplemental only and are not part of the German S3 guideline corpus."
)


def _strip_html(value: str) -> str:
    """Remove HTML tags from *value* and normalise whitespace.

    Args:
        value: Raw HTML string, or None/empty.

    Returns:
        Plain text with HTML entities decoded and runs of whitespace collapsed.
    """
    # Strip all tags, then decode entities, then collapse whitespace.
    text = re.sub(r"<[^>]+>", " ", value or "")
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _google_custom_search(query: str, max_results: int) -> dict | None:
    """Call the Google Custom Search JSON API and return normalised results.

    Returns ``None`` (instead of raising) when API credentials are not
    configured, so the caller can fall through to the DuckDuckGo fallback.

    Args:
        query: Search query string.
        max_results: Maximum results to return (capped at 5 by the API).

    Returns:
        Result dict with ``query``, ``provider``, ``results``, and ``disclosure``
        keys, or ``None`` if credentials are missing.

    Raises:
        httpx.HTTPStatusError: On non-2xx API responses.
    """
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY", "").strip()
    engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID", "").strip()
    if not api_key or not engine_id:
        return None

    timeout = float(os.getenv("WEB_SEARCH_TIMEOUT", "15"))
    with httpx.Client(timeout=timeout) as client:
        response = client.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": api_key,
                "cx": engine_id,
                "q": query,
                "num": max(1, min(max_results, 5)),
            },
        )
        response.raise_for_status()
        payload = response.json()

    items = payload.get("items", [])
    results = [
        {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": item.get("link", ""),
            "source": item.get("displayLink", ""),
        }
        for item in items[:max_results]
    ]
    return {
        "query": query,
        "provider": "google",
        "results": results,
        "disclosure": WEB_SEARCH_DISCLOSURE,
    }


def _duckduckgo_fallback(query: str, max_results: int) -> dict:
    """Scrape DuckDuckGo HTML search results as a fallback when Google is unavailable.

    Parses the ``result__a`` anchor and ``result__snippet`` elements from
    DuckDuckGo's HTML interface using a regex — fragile by nature but avoids
    requiring an API key for basic functionality.

    Args:
        query: Search query string.
        max_results: Maximum results to return (3–5 recommended).

    Returns:
        Result dict with ``query``, ``provider`` ``"duckduckgo"``, ``results``,
        and ``disclosure`` keys.

    Raises:
        httpx.HTTPError: On network or HTTP failures.
    """
    timeout = float(os.getenv("WEB_SEARCH_TIMEOUT", "15"))
    encoded_query = quote_plus(query)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(
            f"https://duckduckgo.com/html/?q={encoded_query}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        html = response.text

    pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
        r'<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
        re.DOTALL,
    )
    results = []
    for match in pattern.finditer(html):
        title = _strip_html(match.group("title"))
        snippet = _strip_html(match.group("snippet"))
        url = unescape(match.group("url"))
        source = re.sub(r"^https?://", "", url).split("/")[0]
        if not title and not snippet:
            continue
        results.append(
            {
                "title": title,
                "snippet": snippet,
                "url": url,
                "source": source,
            }
        )
        if len(results) >= max_results:
            break

    return {
        "query": query,
        "provider": "duckduckgo",
        "results": results,
        "disclosure": WEB_SEARCH_DISCLOSURE,
    }


def web_search_snippets_tool(query: str, max_results: int = 5) -> dict:
    """Fetch supplemental web snippets for *query* using the best available provider.

    Provider priority:
    1. Google Custom Search (if credentials configured).
    2. DuckDuckGo HTML scraper (automatic fallback).
    3. Empty result dict with ``provider: "unavailable"`` on all failures.

    Args:
        query: Search query string.
        max_results: Desired result count; clamped to [3, 5].

    Returns:
        Dict with ``query``, ``provider``, ``results`` (list of title/snippet/url),
        ``disclosure``, and optionally ``error`` on failure.
    """
    max_results = max(3, min(max_results, 5))
    try:
        google_result = _google_custom_search(query, max_results=max_results)
        if google_result is not None:
            return google_result
        return _duckduckgo_fallback(query, max_results=max_results)
    except Exception as exc:
        return {
            "query": query,
            "provider": "unavailable",
            "results": [],
            "disclosure": WEB_SEARCH_DISCLOSURE,
            "error": str(exc),
        }