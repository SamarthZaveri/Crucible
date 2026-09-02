"""FastAPI backend: kicks off research runs in the background and serves run
history + PPO training history for the dashboard.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from crucible import config
from crucible.orchestrator import run_topic

app = FastAPI(title="Crucible API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    topic: str


@app.post("/runs")
def start_run(req: RunRequest, background_tasks: BackgroundTasks):
    def _execute():
        run_topic(req.topic)

    background_tasks.add_task(_execute)
    return {"status": "started", "topic": req.topic}


@app.get("/runs")
def list_runs() -> List[dict]:
    runs_dir = Path(config.RUNS_DIR)
    runs = []
    for f in sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        run = json.loads(f.read_text())
        runs.append({
            "run_id": run["run_id"],
            "topic": run["topic"],
            "judge_overall": run.get("judge_score", {}).get("overall"),
            "revisions": len(run.get("revisions", [])),
            "duration_seconds": run.get("duration_seconds"),
        })
    return runs


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    path = Path(config.RUNS_DIR) / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return json.loads(path.read_text())


@app.get("/training/history")
def training_history() -> List[dict]:
    path = Path(config.TRAINING_HISTORY_PATH)
    if not path.exists():
        return []
    return json.loads(path.read_text())


@app.get("/health")
def health():
    return {"status": "ok"}
