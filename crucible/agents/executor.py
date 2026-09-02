"""Executor agent: runs searches for each sub-question and drafts a cited report."""
from __future__ import annotations

from typing import List, Dict

from crucible import config
from crucible.ollama_client import chat
from crucible.tools.search import search

DRAFT_SYSTEM_PROMPT = """You are a research report writer. You will be given a \
topic, a set of sub-questions, and search results for each sub-question. Write \
a structured report that answers the topic using ONLY the information in the \
search results.

Rules:
- Every factual claim must be traceable to a source. Cite sources inline as \
[1], [2], etc., matching the numbered source list you're given.
- Do not invent facts not present in the sources.
- Structure the report with a short intro, one section per sub-question, and \
a brief conclusion.
- End with a "Sources" list mapping [n] to the source URL.
"""

REVISE_SYSTEM_PROMPT = """You are revising a research report based on critic \
feedback. Keep everything that was already well-grounded; fix only what the \
critic flagged. Preserve the citation format ([1], [2], ...) and the Sources list.
"""


def _gather_sources(sub_questions: List[str], max_results_per_q: int = 4) -> List[Dict]:
    sources = []
    for q in sub_questions:
        try:
            results = search(q, max_results=max_results_per_q)
        except RuntimeError:
            # No search API key configured -- degrade gracefully so the loop
            # is still runnable end-to-end for local plumbing tests.
            results = []
        for r in results:
            sources.append({"sub_question": q, **r})
    return sources


def _format_sources_block(sources: List[Dict]) -> str:
    lines = []
    for i, s in enumerate(sources, start=1):
        lines.append(f"[{i}] ({s['sub_question']}) {s['title']} -- {s['url']}\n{s['content'][:800]}")
    return "\n\n".join(lines) if lines else "(no search results available)"


def draft(topic: str, sub_questions: List[str], model: str = None) -> Dict:
    model = model or config.EXECUTOR_MODEL
    sources = _gather_sources(sub_questions)
    sources_block = _format_sources_block(sources)

    user_msg = (
        f"Topic: {topic}\n\n"
        f"Sub-questions:\n" + "\n".join(f"- {q}" for q in sub_questions) + "\n\n"
        f"Numbered sources:\n{sources_block}"
    )
    messages = [
        {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    report_text = chat(model, messages, temperature=0.5)
    return {"report": report_text, "sources": sources}


def revise(topic: str, previous_report: str, critique: str, sources: List[Dict], model: str = None) -> str:
    model = model or config.EXECUTOR_MODEL
    sources_block = _format_sources_block(sources)
    user_msg = (
        f"Topic: {topic}\n\n"
        f"Previous report:\n{previous_report}\n\n"
        f"Critic feedback to address:\n{critique}\n\n"
        f"Numbered sources (unchanged):\n{sources_block}"
    )
    messages = [
        {"role": "system", "content": REVISE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return chat(model, messages, temperature=0.5)
