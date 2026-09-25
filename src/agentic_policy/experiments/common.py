from __future__ import annotations

import argparse
from pathlib import Path

from agentic_policy.loop import AgentLoop
from agentic_policy.model_adapters import TransformersCausalLM
from agentic_policy.policy import HeuristicPolicy, QwenConstrainedPolicy
from agentic_policy.retrieval import RetrievalTools
from agentic_policy.state_interpreter import QwenTaskStateInterpreter


ROOT = Path(__file__).resolve().parents[3]


def add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", choices=["heuristic", "qwen"], default="heuristic")
    parser.add_argument("--state-interpreter", choices=["passthrough", "qwen"], default="passthrough")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampling", choices=["greedy", "categorical"], default="greedy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "demo_corpus.jsonl")


def make_loop(args: argparse.Namespace) -> AgentLoop:
    runtime = None
    if args.policy == "qwen" or args.state_interpreter == "qwen":
        runtime = TransformersCausalLM(args.model, args.device)
    policy = (
        HeuristicPolicy(args.sampling, args.seed)
        if args.policy == "heuristic"
        else QwenConstrainedPolicy(runtime=runtime, sampling=args.sampling, seed=args.seed)
    )
    interpreter = QwenTaskStateInterpreter(runtime) if args.state_interpreter == "qwen" else None
    return AgentLoop(policy, RetrievalTools.from_jsonl(args.corpus), state_interpreter=interpreter)
