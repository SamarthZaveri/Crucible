# Crucible

Multi-agent research-report pipeline (Planner → Executor → Critic → Judge) with
PPO-driven policy improvement — fully local via Ollama + LoRA, no cloud API
dependency for the core loop.

## Setup

1. Install Ollama: https://ollama.com
2. Pull the models. Use whatever you have; adjust `JUDGE_MODEL` in `.env`
   accordingly (see step 4) — the Judge should ideally be your *largest*
   pulled model, since a weak Judge directly undermines PPO later:
   ```
   ollama pull qwen2.5:1.5b-instruct   # Planner / Executor
   ollama pull qwen2.5:7b-instruct     # Critic (and Judge, if that's your largest model)
   ```
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and fill in your real values:
   ```
   cp .env.example .env
   ```
   At minimum, set `TAVILY_API_KEY` (free key at https://tavily.com). Without
   it, search silently returns empty results and the pipeline still runs, but
   the Executor has nothing real to cite — in practice this produces
   fabricated-looking citations, not an obvious error, so don't skip this.
   If your Judge model isn't `qwen2.5:14b-instruct`, also set `JUDGE_MODEL` in
   `.env` (e.g. `JUDGE_MODEL=qwen2.5:7b-instruct`) to match what you pulled.

## Run a single research topic (no backend needed)

```
cd /path/to/crucible-project
python -m crucible.orchestrator "The economics of remote work for large companies"
```

Saves the full trace to `data/runs/<run_id>.json`.

**Expect some rough runs while stabilizing.** Small local models (1.5B-7B)
fail in specific, recurring ways under this kind of orchestration — degenerate
repetition, malformed JSON, sub-questions that come back as objects instead of
plain strings. The pipeline now defends against all of these (see "Known
failure modes" below), but if you see something new, that's normal for this
setup, not a sign the approach is broken — check that section first, then the
trace itself for what actually went wrong.

## Run the backend + dashboard

```
uvicorn crucible.backend.main:app --reload --port 8000
```

Then open `frontend/index.html` directly in a browser (it talks to
`http://localhost:8000`). Submit topics from the sidebar; the run list polls
every 5s. The "Training Progress" tab reads `data/training_history.json` when
you switch to it — it does not auto-refresh while you're on that tab, so
re-click it to see new PPO iterations as they land.

## Calibrate the Judge (do this before trusting it for PPO)

Run a handful of topics through the orchestrator first (5-10, per the PRD's
Week 1 milestone) — ideally after a couple of clean runs in a row, since a
calibration set built on broken runs won't tell you much. Then:

```
python -m crucible.calibration
```

This asks you to hand-score ~15 reports and reports the Pearson correlation
with the Judge's scores. If correlation is low (<0.5), swap in a larger Judge
model or refine the rubric prompt in `crucible/agents/judge.py` before
proceeding to PPO — training against a noisy reward signal just teaches the
policy to game the noise. This matters more than usual here since Critic and
Judge may currently be running on the *same* model (see Setup) — that removes
some of the independence the calibration step is meant to catch problems with.

## PPO training (Phase 2, requires a CUDA GPU with ~8GB+ VRAM)

```
python -m crucible.ppo.train --topics topics.txt --iterations 20 --batch-size 4
```

**Status: not yet run.** This is a complete, compiling scaffold, not a
verified pipeline — everything below the pipeline itself (Planner through
Judge) has been debugged against real local runs; PPO training has not.
Expect to hit new issues here the same way real runs surfaced new issues in
the base pipeline (`trl` API drift, VRAM limits, `bitsandbytes` on Windows —
see Known failure modes).

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
"Training Progress" tab reads. LoRA adapters + tokenizer are saved to
`data/ppo_checkpoints/` when training finishes.

Each PPO step calls the Judge *and* the search API once per topic in the
batch, so expect training to be bottlenecked by Ollama/search latency, not
just the LoRA update itself.

**Note on this script:** it's built against the `trl` `PPOTrainer`/`PPOConfig`
API — `trl`'s API has shifted across versions (especially the config and
generation call signatures), so if you hit an error on a specific `trl`
version, check that library's PPO example for the current signature and
adjust `ppo/train.py` accordingly rather than assuming the logic itself is
wrong. `bitsandbytes` (used for 4-bit quantization) has historically had
rockier Windows support than Linux — if it fails to install or errors at
runtime, that's a known category of problem, not something wrong with your
setup; the fallback is `load_in_8bit` instead of 4-bit, at the cost of more
VRAM headroom.

**Not yet built:** the pipeline doesn't currently do anything with a
finished PPO checkpoint — training produces LoRA weights in
`data/ppo_checkpoints/`, but nothing loads them back into the report-generation
loop. Closing that loop (merge the adapter into the base model, either
re-imported into Ollama or served directly via `transformers`) is the next
piece of real work once a training run completes successfully.

If PPO doesn't converge cleanly on your setup, that's an expected possible
outcome per the PRD risk table — the honest fallback is prompt-level policy
search (evolving the Planner's system prompt via reward-guided search using
the Judge score), which stays fully within Ollama and needs no GPU. That
fallback isn't implemented here yet; it's a reasonable next script to add
under `crucible/ppo/prompt_search.py` following the same reward function in
`compute_reward()`.

## Known failure modes (already handled)

These were all found via real runs, not anticipated in the abstract — each
one caused a visibly broken report or a crash before the corresponding fix
went in. Listed here so a new failure reads as "one more item for this list"
rather than "something is fundamentally wrong":

- **Small models ignore "respond only with JSON."** Fixed via Ollama's
  `format: "json"` grammar-constrained decoding (`ollama_client.chat_json`),
  plus a `required_keys` check that retries when the JSON is syntactically
  valid but missing expected fields (e.g. Critic once returned
  `{"*": {"title": ..., "summary": ...}}` instead of the requested schema —
  valid JSON, wrong shape entirely).
- **Degenerate generation.** Two distinct forms showed up: exact token-level
  loops (the same sentence repeated dozens of times) and, after an
  over-aggressive first fix, word-salad (incoherent synonym strings from a
  repetition penalty tuned too high). Current settings —
  `repeat_penalty=1.15`, `repeat_last_n=256`, `num_predict=1500` cap — are a
  middle ground found by iterating against both failure directions, not a
  guaranteed-correct value. If you see either symptom again, that's the
  first place to adjust.
- **The Planner returning malformed sub-questions.** Sometimes as objects
  instead of plain strings, with unpredictable key names (`question`,
  `text_only_question`, etc.), or too few of them (fewer than the requested
  3-5). `planner.py` now coerces recognized shapes, rejects unrecognized
  ones outright (never silently stringifies garbage), and retries if fewer
  than 3 genuine sub-questions survive that filter. Garbage sub-questions
  are dangerous specifically because they get used verbatim as real search
  queries — one bad Planner response can silently corrupt an entire run's
  sources.
- **The final report skipping Critic review entirely.** The original
  revision loop could produce one extra, uncritiqued revision after the last
  critique and use *that* as the final output. Fixed: the orchestrator now
  tracks the best-scoring *reviewed* revision throughout the loop and falls
  back to that if nothing gets approved, rather than trusting whatever the
  last revision happened to be.

## Project layout

```
crucible/
  config.py               # model names, thresholds, paths (env-var / .env overridable)
  ollama_client.py         # Ollama /api/chat wrapper: JSON-mode, schema validation, sampling params
  agents/
    planner.py              # decomposes topic -> sub-questions (with retry/validation)
    executor.py              # searches + drafts/revises the report
    critic.py                # fact-checks draft, approves or sends back
    judge.py                 # scores final report (the PPO reward signal)
  tools/
    search.py                # Tavily search with on-disk caching
  orchestrator.py            # runs one full Topic -> Report loop, saves trace
  calibration.py             # hand-score runs, correlate against Judge
  ppo/
    train.py                  # trl PPOTrainer + peft LoRA training loop (not yet run)
  backend/
    main.py                   # FastAPI: start runs, serve run/training history
frontend/
  index.html                  # single-file dashboard (run traces + reward chart)
data/                          # runs, search cache, training history (gitignored)
.env.example                   # copy to .env and fill in TAVILY_API_KEY etc.
topics.txt                     # 92 diverse topics for PPO training
```

## What's implemented vs. what's a scaffold

- **Working and actively debugged against real runs**: Planner, Executor,
  Critic, Judge, the revision loop, the orchestrator, the calibration script,
  the FastAPI backend, and the dashboard. "Working" here means "has survived
  several rounds of real-failure-driven fixes," not "guaranteed bug-free" —
  small local models keep surfacing new edge cases; see Known failure modes.
- **Complete scaffold, not yet run**: `ppo/train.py`. Follows the PRD's
  hybrid Ollama+LoRA design and compiles cleanly, but hasn't been executed
  against real hardware yet, and nothing currently reconnects a finished
  checkpoint to the pipeline (see PPO training section above).
- **Not yet built**: the prompt-level policy search fallback, the merge-back
  bridge for a trained checkpoint, and the trained (non-LLM-judge) reward
  model from the PRD's Phase 2 stretch goals.