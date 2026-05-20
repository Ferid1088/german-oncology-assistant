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
    text = re.sub(r"<[^>]+>", " ", value or "")
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _google_custom_search(query: str, max_results: int) -> dict | None:
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