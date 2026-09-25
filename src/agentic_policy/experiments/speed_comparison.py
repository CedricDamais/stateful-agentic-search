from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from agentic_policy.evaluation import evaluate
from agentic_policy.loop import AgentLoop
from agentic_policy.model_adapters import TransformersCausalLM
from agentic_policy.policy import QwenConstrainedPolicy, QwenReActPolicy
from agentic_policy.retrieval import RetrievalTools
from agentic_policy.state_interpreter import QwenTaskStateInterpreter
from .common import ROOT


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def summarize(controller: str, runs: list[tuple]) -> dict[str, float | int | str]:
    totals = [result.total_latency_ms for result, _ in runs]
    model_times = [sum(step.model_latency_ms for step in result.trajectory) + (result.state_interpretation or {}).get("latency_ms", 0.0) for result, _ in runs]
    tool_times = [sum(max(0.0, step.latency_ms - step.model_latency_ms) for step in result.trajectory) for result, _ in runs]
    metrics = evaluate(runs)
    return {
        "controller": controller,
        "runs": len(runs),
        "latency_mean_ms": round(statistics.mean(totals), 3),
        "latency_p50_ms": round(statistics.median(totals), 3),
        "latency_p95_ms": round(percentile(totals, 0.95), 3),
        "model_time_mean_ms": round(statistics.mean(model_times), 3),
        "tool_time_mean_ms": round(statistics.mean(tool_times), 3),
        "steps_mean": round(statistics.mean(len(result.trajectory) for result, _ in runs), 3),
        "prompt_tokens_mean": metrics.get("prompt_tokens", 0.0),
        "completion_tokens_mean": round(metrics.get("completion_tokens", 0.0) + statistics.mean((result.state_interpretation or {}).get("completion_tokens", 0) for result, _ in runs), 3),
        "state_interpreter_latency_mean_ms": round(statistics.mean((result.state_interpretation or {}).get("latency_ms", 0.0) for result, _ in runs), 3),
        "state_interpreter_prompt_tokens_mean": round(statistics.mean((result.state_interpretation or {}).get("prompt_tokens", 0) for result, _ in runs), 3),
        "state_interpreter_completion_tokens_mean": round(statistics.mean((result.state_interpretation or {}).get("completion_tokens", 0) for result, _ in runs), 3),
        "tool_calls_mean": metrics.get("tool_calls", 0.0),
        "retrieval_success": metrics.get("retrieval_success", 0.0),
        "ndcg_at_3": metrics.get("ndcg_at_3", 0.0),
        "valid_action_rate": round(statistics.mean(
            float(step.policy_valid) for result, _ in runs for step in result.trajectory
        ), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare finite-state action scoring with free-form ReAct generation.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--state-interpreter", choices=["passthrough", "qwen"], default="passthrough",
                        help="Build typed task context once before either controller starts.")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "demo_corpus.jsonl")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data" / "demo_benchmark.jsonl")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--react-max-new-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path, default=Path("runs/speed_comparison.json"))
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 0:
        parser.error("repetitions must be >= 1 and warmups must be >= 0")

    items = [json.loads(line) for line in args.benchmark.read_text().splitlines() if line.strip()]
    runtime = TransformersCausalLM(args.model, args.device)
    tools = RetrievalTools.from_jsonl(args.corpus)
    interpreter = QwenTaskStateInterpreter(runtime) if args.state_interpreter == "qwen" else None
    loops = {
        "finite_state": AgentLoop(QwenConstrainedPolicy(runtime=runtime), tools, state_interpreter=interpreter),
        "react": AgentLoop(QwenReActPolicy(runtime=runtime, max_new_tokens=args.react_max_new_tokens), tools, state_interpreter=interpreter),
    }

    for _ in range(args.warmups):
        for loop in loops.values():
            loop.run(items[0]["question"], args.max_steps)

    all_runs: dict[str, list[tuple]] = {name: [] for name in loops}
    serialized = []
    # Alternate controller order by repetition to reduce thermal/order bias.
    for repetition in range(args.repetitions):
        order = list(loops) if repetition % 2 == 0 else list(reversed(loops))
        for item in items:
            for name in order:
                result = loops[name].run(item["question"], args.max_steps)
                all_runs[name].append((result, item))
                serialized.append({"controller": name, "repetition": repetition, "benchmark": item,
                                   "trajectory": result.to_dict()})

    summaries = {name: summarize(name, runs) for name, runs in all_runs.items()}
    finite_p50 = summaries["finite_state"]["latency_p50_ms"]
    react_p50 = summaries["react"]["latency_p50_ms"]
    comparison = {
        "model": args.model,
        "device": str(runtime.device),
        "state_interpreter": args.state_interpreter,
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "finite_state": summaries["finite_state"],
        "react": summaries["react"],
        "finite_state_speedup_p50": round(react_p50 / finite_p50, 3) if finite_p50 else None,
        "runs": serialized,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps({key: value for key, value in comparison.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
