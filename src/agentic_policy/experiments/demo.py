from __future__ import annotations

import json
from pathlib import Path

from .common import add_policy_arguments, make_loop


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run one constrained-policy trajectory.")
    add_policy_arguments(parser)
    parser.add_argument("--question", default="What does BM25 use to rank documents?")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = make_loop(args).run(args.question)
    record = result.to_dict()
    print(json.dumps(record, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
