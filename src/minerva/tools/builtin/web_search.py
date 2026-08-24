"""A real web search tool, backed by DuckDuckGo's HTML endpoint.

Swift's training corpus deliberately excludes encyclopedic and news content
(see ``training/data.py``): a 9.9M-parameter model has nowhere to reliably
store facts about the world, and a model that tries just confabulates them
fluently. This tool is the honest alternative - instead of teaching Swift
fixed-in-time facts it cannot keep straight, it gives Swift a way to go look
them up. No API key is required: the HTML endpoint is DuckDuckGo's own
no-JS search results page, parsed for real.
"""

from __future__ import annotations

import html
import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from ..base import tool

__all__ = ["web_search"]

_ENDPOINT = "https://html.duckduckgo.com/html/"
_USER_AGENT = "Mozilla/5.0 (compatible; Minerva/0.1; +https://github.com/giamat13/minerva)"

# One search result: an <h2 class="result__title"> anchor followed, later in
# the same result block, by an <a class="result__snippet">. Matched loosely
# and non-greedily across the two anchors rather than parsing the full DOM -
# DuckDuckGo's HTML result markup has stayed stable for years and a full HTML
# parser is more machinery than this page needs.
_RESULT_RE = re.compile(
    r'result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(fragment: str) -> str:
    return html.unescape(_TAG_RE.sub("", fragment)).strip()


def _resolve_url(redirect_href: str) -> str:
    """DuckDuckGo links to //duckduckgo.com/l/?uddg=<real-url>&...; unwrap it."""
    parsed = urlparse(redirect_href if "//" in redirect_href else f"//{redirect_href}")
    target = parse_qs(parsed.query).get("uddg")
    return unquote(target[0]) if target else redirect_href


@tool(name="web_search")
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information and real-world facts.

    Use this for anything Swift cannot know on its own: current events, dates,
    prices, people, places, or any factual claim that could be wrong or
    outdated. Never guess a fact that this tool could look up instead.

    Args:
        query: What to search for, as plain search-engine keywords.
        max_results: How many results to return (1-10). Defaults to 5.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    count = max(1, min(int(max_results), 10))

    try:
        response = httpx.get(
            _ENDPOINT,
            params={"q": query},
            headers={"User-Agent": _USER_AGENT},
            timeout=10.0,
            follow_redirects=True,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"web search failed: {exc}") from exc

    results = []
    for href, title, snippet in _RESULT_RE.findall(response.text):
        title = _clean(title)
        if not title:
            continue
        results.append(f"{title}\n{_resolve_url(href)}\n{_clean(snippet)}")
        if len(results) >= count:
            break

    if not results:
        return f"No results found for {query!r}."
    return "\n\n".join(results)
