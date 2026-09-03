"""Web search tool with on-disk caching, so repeated PPO iterations over the
same topics don't re-hit the search API (see PRD risk: search cost/rate limits).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Dict

import requests

from crucible import config

Path(config.SEARCH_CACHE_DIR).mkdir(parents=True, exist_ok=True)


def _cache_path(query: str) -> Path:
    key = hashlib.sha256(query.strip().lower().encode()).hexdigest()
    return Path(config.SEARCH_CACHE_DIR) / f"{key}.json"


def search(query: str, max_results: int = 5, use_cache: bool = True) -> List[Dict]:
    """Search the web via Tavily and return a list of {title, url, content}.

    Raises a clear error if TAVILY_API_KEY isn't set, rather than silently
    returning an empty list that would look like "no sources found" downstream.
    """
    if not isinstance(query, str):
        raise TypeError(
            f"search() expected a string query, got {type(query).__name__}: {query!r}. "
            f"This usually means an upstream agent (e.g. the Planner) returned a "
            f"malformed sub-question that wasn't caught before reaching search()."
        )

    cache_file = _cache_path(query)
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())["results"]

    if not config.TAVILY_API_KEY:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Get a free key at https://tavily.com "
            "and set it as an environment variable."
        )

    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": config.TAVILY_API_KEY,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    results = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in data.get("results", [])
    ]

    if use_cache:
        cache_file.write_text(json.dumps({"query": query, "results": results}, indent=2))

    return results