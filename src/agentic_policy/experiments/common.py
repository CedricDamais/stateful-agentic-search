from __future__ import annotations

import argparse
from pathlib import Path

from agentic_policy.loop import AgentLoop
from agentic_policy.policy import HeuristicPolicy, QwenConstrainedPolicy
from agentic_policy.retrieval import RetrievalTools


ROOT = Path(__file__).resolve().parents[3]


def add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", choices=["heuristic", "qwen"], default="heuristic")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampling", choices=["greedy", "categorical"], default="greedy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "demo_corpus.jsonl")


def make_loop(args: argparse.Namespace) -> AgentLoop:
    policy = (HeuristicPolicy(args.sampling, args.seed) if args.policy == "heuristic"
              else QwenConstrainedPolicy(args.model, args.device, args.sampling, args.seed))
    return AgentLoop(policy, RetrievalTools.from_jsonl(args.corpus))
