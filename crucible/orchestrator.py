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
    while revision_count < config.MAX_CRITIC_REVISIONS:
        review = critic.review(topic, sub_questions, report, sources)
        trace["revisions"].append({"revision": revision_count, "report": report, "critique": review})

        if review.get("approved") or review.get("score", 0) >= config.CRITIC_APPROVE_THRESHOLD:
            approved = True
            break

        report = executor.revise(topic, report, review.get("feedback", ""), sources)
        revision_count += 1

    if not approved:
        # Loop exhausted without approval -- still proceed to judging, but flag it
        # rather than silently presenting an unapproved report as if it passed.
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
