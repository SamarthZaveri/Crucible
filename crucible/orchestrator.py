"""Ties Planner -> Executor -> Critic (with bounded revision loop) -> Judge
into a single research run, and persists the full trace to disk for the
dashboard to read.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Dict

from crucible import config
from crucible.agents import planner, executor, critic, judge

Path(config.RUNS_DIR).mkdir(parents=True, exist_ok=True)


def run_topic(topic: str) -> Dict:
    run_id = str(uuid.uuid4())[:8]
    started = time.time()
    trace = {"run_id": run_id, "topic": topic, "revisions": []}

    sub_questions = planner.plan(topic)
    trace["sub_questions"] = sub_questions

    draft_result = executor.draft(topic, sub_questions)
    report = draft_result["report"]
    sources = draft_result["sources"]

    revision_count = 0
    approved = False
    best_report = report
    best_score = -1.0

    while revision_count < config.MAX_CRITIC_REVISIONS:
        review = critic.review(topic, sub_questions, report, sources)
        trace["revisions"].append({"revision": revision_count, "report": report, "critique": review})

        score = review.get("score", 0) or 0
        if score > best_score:
            best_score = score
            best_report = report

        if review.get("approved") or score >= config.CRITIC_APPROVE_THRESHOLD:
            approved = True
            break

        # If this was the last allowed revision, stop here rather than
        # calling revise() again -- a report the Critic never gets to see
        # should never become the final output. Fall back to the
        # best-reviewed version below instead.
        if revision_count == config.MAX_CRITIC_REVISIONS - 1:
            break

        report = executor.revise(topic, sub_questions, report, review.get("feedback", ""), sources)
        revision_count += 1

    if not approved:
        # Loop exhausted without approval -- use the highest-scoring reviewed
        # version rather than whatever the last revision happened to produce.
        # (Revisions aren't guaranteed to be monotonic improvements -- see
        # README note on this -- so "best seen" beats "most recent".)
        report = best_report
        trace["max_revisions_hit"] = True

    final_score = judge.score(topic, sub_questions, report, sources)
    trace["final_report"] = report
    trace["sources"] = sources
    trace["judge_score"] = final_score
    trace["duration_seconds"] = round(time.time() - started, 1)

    out_path = Path(config.RUNS_DIR) / f"{run_id}.json"
    out_path.write_text(json.dumps(trace, indent=2))

    return trace


if __name__ == "__main__":
    import sys

    topic = " ".join(sys.argv[1:]) or "The environmental impact of lithium mining for EV batteries"
    result = run_topic(topic)
    print(f"Run {result['run_id']} -- judge overall score: {result['judge_score'].get('overall')}")
    print(f"Saved trace to {config.RUNS_DIR}/{result['run_id']}.json")