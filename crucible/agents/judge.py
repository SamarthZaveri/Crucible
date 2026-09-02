"""Reward model / Judge: scores a finished report 0-10 on groundedness,
completeness, and coherence. This is the reward signal PPO optimizes against --
see crucible/calibration.py for why it must be validated before trusting it.
"""
from __future__ import annotations

from typing import Dict, List

from crucible import config
from crucible.ollama_client import chat_json

SYSTEM_PROMPT = """You are an expert grader of research reports. Score the \
given report on three dimensions, each 0-10:

- groundedness: are claims actually backed by the cited sources? Penalize \
  unsupported or misattributed claims heavily.
- completeness: does it address all the sub-questions with adequate depth?
- coherence: is it well-organized, logically consistent, readable?

Respond ONLY with JSON in this exact shape, no other text:
{
  "groundedness": 0-10,
  "completeness": 0-10,
  "coherence": 0-10,
  "overall": 0-10,
  "rationale": "1-2 sentence justification"
}

"overall" should reflect groundedness most heavily (it's the hardest thing to \
fake and the thing most worth rewarding), then completeness, then coherence.
Do not reward confident-sounding writing that isn't actually grounded.
"""


def score(topic: str, sub_questions: List[str], report: str, sources: List[Dict], model: str = None) -> Dict:
    model = model or config.JUDGE_MODEL
    sources_block = "\n\n".join(
        f"[{i}] {s['title']} -- {s['url']}\n{s['content'][:600]}"
        for i, s in enumerate(sources, start=1)
    ) or "(no sources were available)"

    user_msg = (
        f"Topic: {topic}\n\n"
        f"Sub-questions:\n" + "\n".join(f"- {q}" for q in sub_questions) + "\n\n"
        f"Report:\n{report}\n\n"
        f"Numbered sources:\n{sources_block}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return chat_json(model, messages, temperature=0.1)
