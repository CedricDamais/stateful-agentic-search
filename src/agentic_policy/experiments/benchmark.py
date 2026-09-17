from __future__ import annotations

import json
from pathlib import Path

from agentic_policy.evaluation import evaluate
from .common import ROOT, add_policy_arguments, make_loop


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a policy against the small reproducible benchmark.")
    add_policy_arguments(parser)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data" / "demo_benchmark.jsonl")
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark.jsonl"))
    args = parser.parse_args()
    items = [json.loads(line) for line in args.benchmark.read_text().splitlines() if line.strip()]
    loop, records = make_loop(args), []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for item in items:
            result = loop.run(item["question"])
            records.append((result, item))
            output.write(json.dumps({"benchmark": item, "trajectory": result.to_dict()}) + "\n")
    print(json.dumps(evaluate(records), indent=2))


if __name__ == "__main__":
    main()
