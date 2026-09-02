"""Central config for Crucible — model names, thresholds, paths. All
overridable via environment variables so you can swap models without
touching code (e.g. a smaller Judge model if VRAM/CPU-bound).
"""
import os

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "qwen2.5:1.5b-instruct")
EXECUTOR_MODEL = os.environ.get("EXECUTOR_MODEL", "qwen2.5:1.5b-instruct")
CRITIC_MODEL = os.environ.get("CRITIC_MODEL", "qwen2.5:7b-instruct")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen2.5:7b-instruct")

# Search
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "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")
SEARCH_CACHE_DIR = os.environ.get("SEARCH_CACHE_DIR", "./data/search_cache")

# Orchestration
MAX_CRITIC_REVISIONS = int(os.environ.get("MAX_CRITIC_REVISIONS", "3"))
CRITIC_APPROVE_THRESHOLD = float(os.environ.get("CRITIC_APPROVE_THRESHOLD", "7.0"))

# Storage
RUNS_DIR = os.environ.get("RUNS_DIR", "./data/runs")
TRAINING_HISTORY_PATH = os.environ.get("TRAINING_HISTORY_PATH", "./data/training_history.json")

# PPO / LoRA
PPO_BASE_MODEL = os.environ.get("PPO_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
PPO_OUTPUT_DIR = os.environ.get("PPO_OUTPUT_DIR", "./data/ppo_checkpoints")
