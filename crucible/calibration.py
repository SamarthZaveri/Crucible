"""Judge calibration: score a handful of existing runs by hand and compare
against the Judge's scores. Run this before trusting the Judge to drive PPO
(see PRD risk: "Judge model too weak/noisy to give a usable signal").

Usage:
    python -m crucible.calibration            # interactively score uncalibrated runs
    python -m crucible.calibration --report    # print correlation report only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from crucible import config

CALIBRATION_PATH = Path(config.RUNS_DIR).parent / "calibration.json"


def _load_calibration() -> dict:
    if CALIBRATION_PATH.exists():
        return json.loads(CALIBRATION_PATH.read_text())
    return {}


def _save_calibration(data: dict) -> None:
    CALIBRATION_PATH.write_text(json.dumps(data, indent=2))


def _pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    varx = sum((x - mx) ** 2 for x in xs)
    vary = sum((y - my) ** 2 for y in ys)
    if varx == 0 or vary == 0:
        return float("nan")
    return cov / (varx ** 0.5 * vary ** 0.5)


def score_interactively(target_n: int = 15) -> None:
    calibration = _load_calibration()
    run_files = sorted(Path(config.RUNS_DIR).glob("*.json"))

    scored = 0
    for run_file in run_files:
        if scored >= target_n:
            break
        run_id = run_file.stem
        if run_id in calibration:
            continue

        run = json.loads(run_file.read_text())
        print("\n" + "=" * 70)
        print(f"Topic: {run['topic']}")
        print(f"\n--- Final report ---\n{run['final_report'][:2000]}")
        print(f"\nJudge scored this: {run['judge_score']}")
        raw = input("\nYour overall score for this report (0-10, or 's' to skip): ").strip()
        if raw.lower() == "s":
            continue
        try:
            human_score = float(raw)
        except ValueError:
            print("Not a number, skipping.")
            continue

        calibration[run_id] = {
            "human_overall": human_score,
            "judge_overall": run["judge_score"].get("overall"),
        }
        _save_calibration(calibration)
        scored += 1

    print(f"\nScored {scored} new runs. Total calibrated: {len(calibration)}")
    _print_report(calibration)


def _print_report(calibration: dict) -> None:
    if len(calibration) < 2:
        print("Need at least 2 calibrated runs to compute correlation.")
        return
    human = [v["human_overall"] for v in calibration.values() if v["judge_overall"] is not None]
    judge = [v["judge_overall"] for v in calibration.values() if v["judge_overall"] is not None]
    r = _pearson(human, judge)
    print(f"\nCalibration set size: {len(human)}")
    print(f"Pearson correlation (human vs. judge overall score): {r:.3f}")
    if r < 0.5:
        print("Low correlation -- consider a larger judge model, or refining the rubric "
              "prompt in agents/judge.py before using this Judge to drive PPO.")
    else:
        print("Correlation looks reasonable to proceed with PPO training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="Print correlation report only")
    args = parser.parse_args()

    if args.report:
        _print_report(_load_calibration())
    else:
        score_interactively()
