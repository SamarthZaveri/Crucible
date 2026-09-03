"""Planner agent: decomposes a research topic into a small set of sub-questions."""
from __future__ import annotations

from typing import List, Optional

from crucible import config
from crucible.ollama_client import chat_json

SYSTEM_PROMPT = """You are a research planner. Given a topic, break it down into \
3-5 focused sub-questions that, if answered, would let someone write a \
well-grounded, complete report on the topic. Sub-questions should be specific \
and searchable (good for a web search engine), not vague.

Respond ONLY with JSON in this exact shape, no other text:
{"sub_questions": ["...", "...", "..."]}
"""


def _coerce_to_string(item, index: int) -> Optional[str]:
    """The Planner is supposed to return a flat list of strings, but small
    models sometimes wrap each entry in an object instead (e.g.
    {"question": "..."} instead of "..."), and the exact key they use for
    the text isn't predictable -- seen in practice: "question", and also
    "text_only_question" (an unrecognized key, from a model apparently
    echoing an unrelated schema). Rather than guess every possible key name,
    or fall back to stringifying the whole dict (which previously produced
    literal garbage like "{'question_type': '', 'text_only_question': ''}"
    that got used as a real search query), return None for anything we can't
    confidently extract real question text from. The caller filters None out
    and retries if too few genuine sub-questions remain -- silently keeping
    unusable placeholder text is worse than asking again.
    """
    if isinstance(item, str):
        text = item.strip()
        return text if len(text) >= 10 else None
    if isinstance(item, dict):
        for key in ("question", "sub_question", "text", "value"):
            val = item.get(key)
            if isinstance(val, str) and len(val.strip()) >= 10:
                return val.strip()
        return None
    return None


MIN_SUB_QUESTIONS = 3


def plan(topic: str, model: str = None, max_retries: int = 2) -> List[str]:
    model = model or config.PLANNER_MODEL
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Topic: {topic}"},
    ]

    sub_questions: List[str] = []
    for attempt in range(max_retries + 1):
        data = chat_json(model, messages, required_keys=["sub_questions"])
        raw_sub_questions = data.get("sub_questions", [])
        coerced = [_coerce_to_string(q, i) for i, q in enumerate(raw_sub_questions)]
        sub_questions = [q for q in coerced if q is not None]

        if len(sub_questions) >= MIN_SUB_QUESTIONS:
            return sub_questions

        # Too few genuine sub-questions narrows everything downstream (search
        # coverage, report completeness) -- or, worse, garbage sub-questions
        # get used verbatim as real search queries. Ask again explicitly
        # rather than silently accepting a thin or malformed plan.
        messages = messages + [
            {
                "role": "user",
                "content": (
                    f"You returned only {len(sub_questions)} usable sub-question(s) "
                    f"(some entries were empty, too short, or not plain question "
                    f"text). Return at least {MIN_SUB_QUESTIONS} (aim for 3-5) "
                    f"distinct, specific sub-questions as plain strings -- not "
                    f"objects -- in the same JSON shape."
                ),
            }
        ]

    if not sub_questions:
        raise ValueError(
            f"Planner failed to produce usable sub-questions for topic: {topic} "
            f"after {max_retries + 1} attempts."
        )
    # Exhausted retries but got at least something usable -- proceed with what
    # we have rather than failing the whole run over a thin-but-usable plan.
    return sub_questions