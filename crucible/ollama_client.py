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
    json_mode: bool = False,
    num_predict: int = 1500,
    repeat_penalty: float = 1.15,
) -> str:
    """Call Ollama's /api/chat endpoint and return the assistant's text reply.

    Retries a few times on transient connection errors, since local models
    can be slow to load into memory on first call.

    json_mode=True sets Ollama's `format: "json"` option, which uses
    grammar-constrained decoding to force valid JSON output. This matters
    because smaller instruct models (e.g. 7B-class) are unreliable about
    following a plain "respond only with JSON" instruction in the prompt --
    they'll happily write normal prose instead. format=json makes that
    structurally impossible rather than just requested.

    repeat_penalty and num_predict guard against degenerate generation.
    Tuning this is a genuine balancing act, confirmed empirically here, not
    just in theory: repeat_penalty=1.3 with repeat_last_n=512 (an earlier
    version of these defaults) fixed paragraph-level repetition but
    overcorrected into a worse failure -- the model, punished too hard for
    reusing any word seen in the last 512 tokens, degenerated into
    incoherent synonym-salad on topics that inherently need to repeat their
    own subject terms (e.g. "remote work" said many times in a report about
    remote work). 1.15 with a 256-token window is a milder middle ground:
    enough to discourage short verbatim loops without punishing normal
    topical repetition into incoherence. num_predict remains a hard backstop
    on generation length either way.
    """
    url = f"{config.OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": 256,
            "num_predict": num_predict,
        },
    }
    if json_mode:
        payload["format"] = "json"

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


def chat_json(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_json_retries: int = 2,
    required_keys: Optional[List[str]] = None,
    **kwargs,
) -> dict:
    """Call chat() with json_mode=True and parse the reply as JSON.

    Even with format=json, retries with a sharper reminder if parsing still
    fails -- format=json guarantees syntactically valid JSON but a model can
    still return the wrong shape entirely under pressure (seen in practice:
    a critic call once returned {"*": {"title": ..., "summary": ...}} instead
    of the requested {"approved", "score", "issues", "feedback"} -- valid
    JSON, wrong schema). Pass required_keys to catch that: a response missing
    any of them is treated the same as invalid JSON and retried.
    """
    last_err: Optional[Exception] = None
    working_messages = list(messages)

    for attempt in range(max_json_retries + 1):
        raw = chat(model, working_messages, temperature=temperature, json_mode=True, **kwargs)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        retry_reason = None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_err = e
            retry_reason = "Your last response was not valid JSON."
        else:
            if required_keys:
                missing = [k for k in required_keys if k not in parsed]
                if missing:
                    last_err = OllamaError(f"Missing required keys {missing} in: {parsed}")
                    retry_reason = (
                        f"Your last response was valid JSON but was missing required "
                        f"field(s): {', '.join(missing)}."
                    )
            if retry_reason is None:
                return parsed

        working_messages = messages + [
            {
                "role": "user",
                "content": (
                    f"{retry_reason} Respond with ONLY a single JSON object "
                    f"matching the exact requested shape -- no prose, no "
                    f"markdown, no explanation, no extra or renamed fields."
                ),
            }
        ]

    raise OllamaError(
        f"Model '{model}' did not return valid, correctly-shaped JSON after "
        f"{max_json_retries + 1} attempts. This can happen with smaller models "
        f"under load -- if it's persistent, consider a larger model for this "
        f"role. Last error: {last_err}"
    )