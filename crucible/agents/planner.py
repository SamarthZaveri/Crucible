"""Planner agent: decomposes a research topic into a small set of sub-questions."""
from __future__ import annotations

from typing import List

from crucible import config
from crucible.ollama_client import chat_json

SYSTEM_PROMPT = """You are a research planner. Given a topic, break it down into \
3-5 focused sub-questions that, if answered, would let someone write a \
well-grounded, complete report on the topic. Sub-questions should be specific \
and searchable (good for a web search engine), not vague.

Respond ONLY with JSON in this exact shape, no other text:
{"sub_questions": ["...", "...", "..."]}
"""


def plan(topic: str, model: str = None) -> List[str]:
    model = model or config.PLANNER_MODEL
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Topic: {topic}"},
    ]
    data = chat_json(model, messages)
    sub_questions = data.get("sub_questions", [])
    if not sub_questions:
        raise ValueError(f"Planner returned no sub-questions for topic: {topic}")
    return sub_questions
