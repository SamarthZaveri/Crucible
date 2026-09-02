"""Thin wrapper around the Ollama HTTP API for chat completions."""
from __future__ import annotations

import json
import time
from typing import List, Dict, Optional

import requests

from crucible import config


class OllamaError(RuntimeError):
    pass


def chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_retries: int = 3,
    timeout: int = 120,
) -> str:
    """Call Ollama's /api/chat endpoint and return the assistant's text reply.

    Retries a few times on transient connection errors, since local models
    can be slow to load into memory on first call.
    """
    url = f"{config.OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]
        except (requests.RequestException, KeyError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise OllamaError(
        f"Failed to reach Ollama at {config.OLLAMA_HOST} for model '{model}' "
        f"after {max_retries} attempts. Is `ollama serve` running and is the "
        f"model pulled (`ollama pull {model}`)? Last error: {last_err}"
    )


def chat_json(model: str, messages: List[Dict[str, str]], temperature: float = 0.3, **kwargs) -> dict:
    """Call chat() and parse the reply as JSON, stripping markdown fences if the
    model wrapped its output in them despite instructions not to.
    """
    raw = chat(model, messages, temperature=temperature, **kwargs)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise OllamaError(f"Model '{model}' did not return valid JSON. Raw output:\n{raw}") from e
