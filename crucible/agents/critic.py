"""Critic agent: reviews a draft report against its sources and either approves
it or sends it back with specific, actionable feedback.
"""
from __future__ import annotations

from typing import Dict, List

from crucible import config
from crucible.ollama_client import chat_json

SYSTEM_PROMPT = """You are a strict fact-checking critic for research reports. \
Given a report and its numbered sources, check every factual claim:

1. Is it actually supported by the cited source (not just plausible-sounding)?
2. Are there unsupported claims with no citation, or citations that don't \
   match the claim?
3. Is the report complete relative to the sub-questions it was meant to answer?
4. Is it internally coherent (no contradictions, logical flow)?

Respond ONLY with JSON in this exact shape, no other text:
{
  "approved": true or false,
  "score": 0-10,
  "issues": ["specific issue 1", "specific issue 2"],
  "feedback": "concise, actionable feedback for the writer to fix the issues"
}

Set "approved" to true only if there are no unsupported claims and the report \
is reasonably complete. Be strict -- approving a flawed report defeats the point.
"""


def review(topic: str, sub_questions: List[str], report: str, sources: List[Dict], model: str = None) -> Dict:
    model = model or config.CRITIC_MODEL
    sources_block = "\n\n".join(
        f"[{i}] {s['title']} -- {s['url']}\n{s['content'][:600]}"
        for i, s in enumerate(sources, start=1)
    ) or "(no sources were available)"

    user_msg = (
        f"Topic: {topic}\n\n"
        f"Sub-questions this report should answer:\n" + "\n".join(f"- {q}" for q in sub_questions) + "\n\n"
        f"Report:\n{report}\n\n"
        f"Numbered sources:\n{sources_block}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return chat_json(
        model, messages, temperature=0.2,
        required_keys=["approved", "score", "issues", "feedback"],
    )