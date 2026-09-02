# Crucible

Multi-agent research-report pipeline (Planner → Executor → Critic → Judge) with
PPO-driven policy improvement — fully local via Ollama + LoRA, no cloud API
dependency for the core loop.

## Setup

1. Install Ollama: https://ollama.com
2. Pull the models:
   ```
   ollama pull qwen2.5:1.5b-instruct   # Planner / Executor
   ollama pull qwen2.5:7b-instruct     # Critic
   ollama pull qwen2.5:14b-instruct    # Judge (use a smaller one if VRAM-limited)
   ```
3. `pip install -r requirements.txt`
4. Get a free Tavily API key (https://tavily.com) and `export TAVILY_API_KEY=...`
   (the pipeline still runs without it, with empty search results, so you can
   test the plumbing before wiring up search)

## Run a single research topic (no backend needed)

```
cd /path/to/crucible-project
python -m crucible.orchestrator "The economics of remote work for large companies"
```

Saves the full trace to `data/runs/<run_id>.json`.

## Run the backend + dashboard

```
uvicorn crucible.backend.main:app --reload --port 8000
```

Then open `frontend/index.html` directly in a browser (it talks to
`http://localhost:8000`). Submit topics from the sidebar; the run list polls
every 5s and the "Training Progress" tab reads `data/training_history.json`.

## Calibrate the Judge (do this before trusting it for PPO)

Run a handful of topics through the orchestrator first (5-10, per the PRD's
Week 1 milestone), then:

```
python -m crucible.calibration
```

This asks you to hand-score ~15 reports and reports the Pearson correlation
with the Judge's scores. If correlation is low (<0.5), swap in a larger Judge
model or refine the rubric prompt in `crucible/agents/judge.py` before
proceeding to PPO — training against a noisy reward signal just teaches the
policy to game the noise.

## PPO training (Phase 2, requires a CUDA GPU with ~8GB+ VRAM)

```
python -m crucible.ppo.train --topics topics.txt --iterations 20 --batch-size 4
```

**What it trains on:** there's no labeled dataset -- PPO is on-policy. `topics.txt`
(92 diverse topics included) is just a list of prompts; the model generates its
own report for a sampled batch each iteration, the Judge scores it, and that
score becomes the reward. Always pass `--topics topics.txt` (or your own,
larger list) -- without it, the script falls back to 5 hardcoded default
topics, and training against a handful of topics repeated across many
iterations risks the policy overfitting to whatever the Judge rewards on
*those specific topics* rather than learning something that generalizes
(reward hacking). More topics, or topics you add over time, meaningfully
improve training quality here.

**Grounding:** each PPO step does one search per topic (cached to disk, same
mechanism as `tools/search.py`) and gives those sources to both the generation
prompt and the Judge's scoring call. This matters because the Judge's rubric
weights groundedness most heavily -- without real sources to check claims
against, that dimension can't be assessed at all, and training would optimize
against a signal that's silently missing its main component. Requires
`TAVILY_API_KEY` to be set for this to actually work; without it, sources are
empty and groundedness reward reverts to being unmeasured.

Trains LoRA adapters on top of the base Planner/Executor model
(`Qwen/Qwen2.5-1.5B-Instruct` by default, loaded via `transformers`+`peft`+
4-bit quantization to fit 8GB), using the Judge as the reward signal. Writes
reward-over-iterations to `data/training_history.json`, which the dashboard's
"Training Progress" tab reads directly. LoRA adapters + tokenizer are saved to
`data/ppo_checkpoints/` when training finishes.

Each PPO step now calls the Judge *and* the search API once per topic in the
batch, so expect training to be bottlenecked by Ollama/search latency, not
just the LoRA update itself.

**Note on this script:** it's a working scaffold built against the `trl`
`PPOTrainer`/`PPOConfig` API — `trl`'s API has shifted across versions
(especially the config and generation call signatures), so if you hit an
error on a specific `trl` version, check that library's PPO example for the
current signature and adjust `ppo/train.py` accordingly rather than assuming
the logic itself is wrong.

If PPO doesn't converge cleanly on your setup, that's an expected possible
outcome per the PRD risk table — the honest fallback is prompt-level policy
search (evolving the Planner's system prompt via reward-guided search using
the Judge score), which stays fully within Ollama and needs no GPU. That
fallback isn't implemented here yet; it's a reasonable next script to add
under `crucible/ppo/prompt_search.py` following the same reward function in
`compute_reward()`.

## Project layout

```
crucible/
  config.py               # model names, thresholds, paths (env-var overridable)
  ollama_client.py         # thin wrapper around Ollama's /api/chat
  agents/
    planner.py              # decomposes topic -> sub-questions
    executor.py              # searches + drafts/revises the report
    critic.py                # fact-checks draft, approves or sends back
    judge.py                 # scores final report (the PPO reward signal)
  tools/
    search.py                # Tavily search with on-disk caching
  orchestrator.py            # runs one full Topic -> Report loop, saves trace
  calibration.py             # hand-score runs, correlate against Judge
  ppo/
    train.py                  # trl PPOTrainer + peft LoRA training loop
  backend/
    main.py                   # FastAPI: start runs, serve run/training history
frontend/
  index.html                  # single-file dashboard (run traces + reward chart)
data/                          # runs, search cache, training history (gitignored)
```

## What's implemented vs. what's a scaffold

- **Fully working, testable today** (once Ollama models are pulled):
  Planner, Executor, Critic, Judge, the revision loop, the orchestrator, the
  calibration script, the FastAPI backend, and the dashboard.
- **Scaffold, needs a GPU to actually run**: `ppo/train.py`. The logic is
  complete and follows the PRD's hybrid Ollama+LoRA design, but `trl`'s exact
  API surface changes between releases — treat this as a strong starting
  point to debug against your installed `trl` version, not a guaranteed
  drop-in run.
- **Not yet built**: the prompt-level policy search fallback, and the trained
  (non-LLM-judge) reward model from the PRD's Phase 2 stretch goals.