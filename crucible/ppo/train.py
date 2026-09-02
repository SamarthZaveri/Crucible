"""PPO training loop for the Planner/Executor policy, using LoRA adapters to
fit within ~8GB VRAM (see PRD: hybrid Ollama + transformers/trl approach).

The frozen Critic and Judge continue to run via Ollama (see crucible.agents.*).
Only the Planner/Executor model is loaded here via `transformers`, since Ollama
is inference-only and can't backprop.

This treats one full research-report generation (a combined plan+draft text)
as a single PPO "response" to a topic "query" -- a simplification that keeps
the PPO trajectory tractable, per the PRD (Planner/Executor can share one
trainable model via role-conditioned prompting).

Sources are pre-fetched once per topic (one search call, not per sub-question)
and used two ways: (1) injected into the generation prompt as context, and
(2) passed to the Judge so "groundedness" is actually measurable against real
sources rather than scored against an empty list. Without this, groundedness
-- the dimension the Judge rubric weights most heavily -- can't be assessed at
all during training, since there's nothing to check claims against.

Requires: transformers, trl, peft, bitsandbytes, accelerate, torch (CUDA).
Requires TAVILY_API_KEY to be set (see crucible/tools/search.py) -- without
it, sources will be empty and groundedness reward reverts to being unmeasured.

Usage:
    python -m crucible.ppo.train --topics topics.txt --iterations 20 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from crucible import config
from crucible.agents import judge
from crucible.tools.search import search

DEFAULT_TOPICS = [
    "The environmental impact of lithium mining for EV batteries",
    "How mRNA vaccines work and their development timeline",
    "The economics of remote work for large companies",
    "Recent progress in fusion energy research",
    "The impact of AI on entry-level software engineering jobs",
]


def load_topics(path: str = None) -> List[str]:
    if path and Path(path).exists():
        return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    return DEFAULT_TOPICS


def fetch_topic_sources(topic: str, max_results: int = 5) -> List[Dict]:
    """One search call per topic (not per sub-question, to keep PPO steps
    cheap) so both the prompt and the Judge have real material to ground
    against. Falls back to an empty list if no search API key is configured --
    training still runs, but groundedness reward is then unmeasured again.
    """
    try:
        return search(topic, max_results=max_results)
    except RuntimeError:
        return []


def _format_sources_block(sources: List[Dict]) -> str:
    if not sources:
        return "(no sources available)"
    return "\n\n".join(
        f"[{i}] {s['title']} -- {s['url']}\n{s['content'][:500]}"
        for i, s in enumerate(sources, start=1)
    )


def build_prompt(topic: str, sources: List[Dict]) -> str:
    sources_block = _format_sources_block(sources)
    return (
        "You are a research planner and writer combined. Given a topic and "
        "some search results, first list 3-5 sub-questions, then write a "
        "short structured report answering them using ONLY the information "
        "in the sources. Cite sources inline as [1], [2], etc.\n\n"
        f"Topic: {topic}\n\nSources:\n{sources_block}\n\nPlan and report:"
    )


def compute_reward(topic: str, generated_text: str, sources: List[Dict]) -> float:
    """Reward = the (calibrated) Judge's overall score (0-10) on the generated
    text, normalized to [0, 1] for PPO stability. Scored against the same
    sources that were given at generation time, so groundedness is actually
    measurable rather than scored against an empty list.
    """
    sub_questions = [topic]
    result = judge.score(topic, sub_questions, generated_text, sources=sources)
    overall = result.get("overall", 0)
    try:
        return float(overall) / 10.0
    except (TypeError, ValueError):
        return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", type=str, default=None, help="Path to a newline-delimited topics file")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-model", type=str, default=config.PPO_BASE_MODEL)
    parser.add_argument("--output-dir", type=str, default=config.PPO_OUTPUT_DIR)
    args = parser.parse_args()

    topics = load_topics(args.topics)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 4-bit quantized base model + LoRA adapters -- fits an 8GB card.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        peft_config=lora_config,
        device_map="auto",
    )

    ppo_config = PPOConfig(
        model_name=args.base_model,
        learning_rate=1.4e-5,
        batch_size=args.batch_size,
        mini_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
    )
    ppo_trainer = PPOTrainer(ppo_config, model, ref_model=None, tokenizer=tokenizer)

    generation_kwargs = {
        "max_new_tokens": 400,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.7,
        "pad_token_id": tokenizer.eos_token_id,
    }

    history = []
    history_path = Path(config.TRAINING_HISTORY_PATH)
    if history_path.exists():
        history = json.loads(history_path.read_text())

    for iteration in range(args.iterations):
        batch_topics = random.sample(topics, k=min(args.batch_size, len(topics)))
        # One search per topic, cached to disk by crucible.tools.search -- so
        # revisiting the same topic in a later iteration doesn't re-hit the API.
        batch_sources = [fetch_topic_sources(t) for t in batch_topics]

        prompts = [build_prompt(t, s) for t, s in zip(batch_topics, batch_sources)]
        query_tensors = [tokenizer(p, return_tensors="pt").input_ids[0] for p in prompts]

        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

        rewards = [
            torch.tensor(compute_reward(t, r, s))
            for t, r, s in zip(batch_topics, responses, batch_sources)
        ]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        mean_reward = float(sum(r.item() for r in rewards) / len(rewards))
        record = {
            "iteration": iteration,
            "mean_reward": mean_reward,
            "topics": batch_topics,
            "policy_loss": stats.get("ppo/loss/policy"),
            "value_loss": stats.get("ppo/loss/value"),
        }
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2))
        print(f"[iter {iteration}] mean_reward={mean_reward:.3f}")

    ppo_trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()