# agentic-policy

Experiment 0 for an agentic-search policy: given an explicit state \(S_t\), select one action from a finite set, execute it, append the observation, and repeat. This is deliberately a **vanilla constrained-decoding baseline**—no SFT or RL is included.

The project separates the policy from the sampler. A Qwen adapter maps the actions to single-token codes, reads their logits in one forward pass, normalizes those scores into a distribution, and the sampler chooses an action. Invalid actions are impossible by construction and the prompt prefill is performed only once per decision.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
agentic-policy-demo --policy heuristic --seed 7
agentic-policy-benchmark --policy heuristic --output runs/benchmark.jsonl
pytest
```

The default heuristic policy makes the included demo fully reproducible and works without a downloaded model. To run the constrained Qwen policy (a GPU is strongly recommended):

```bash
pip install -e '.[qwen]'
agentic-policy-demo --policy qwen --model Qwen/Qwen2.5-1.5B-Instruct \
  --device auto --sampling greedy
```

`--sampling greedy` selects the highest-probability valid action; `--sampling categorical` samples from the same masked distribution. The Qwen policy applies a softmax across only the seven single-token action codes and then chooses. It is the Experiment 0 comparator for later SFT/RL policies.

## Layout

```text
src/agentic_policy/
  actions.py        finite action schema
  state.py          serializable S_t and trajectories
  policy.py         policy interface, heuristic, Qwen constrained adapter
  loop.py           select → tool → observation loop
  retrieval.py      deterministic local BM25-like, vector, hybrid, rerank tools
  evaluation.py     retrieval/policy/efficiency metrics
  experiments/      demo and benchmark entry points
data/demo_corpus.jsonl
tests/              loop and metric tests
```

## Outputs and metrics

Each run emits a JSONL trajectory (one record per question). A step retains the full action probability distribution, selected action, observation summary, elapsed milliseconds, model token counts when available, and tool call count. The benchmark summarizes answer/retrieval success, Recall@k, nDCG@k, expected-action accuracy, mean tool calls, wall latency, and model-token totals.

This project intentionally uses a tiny in-repo corpus so baseline changes remain reproducible. Replace `data/demo_corpus.jsonl` with your corpus or implement a production retriever behind `RetrievalTools`; the state and evaluation contract stays unchanged.

## Finite-state vs ReAct speed benchmark

The comparison uses one shared Qwen model and identical state, retrieval tools, questions, top-k, and step budget. The finite controller scores all action candidates in one batched forward pass. The ReAct controller freely generates a short reasoning trace ending in `Action: <ACTION>`. Warm-up runs are excluded, controller order alternates, and accelerator synchronization is included in timings.

```bash
agentic-policy-speed \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device auto \
  --warmups 1 \
  --repetitions 5 \
  --output runs/speed_comparison.json
```

The summary reports end-to-end mean/p50/p95 latency, model time, retrieval-tool time, steps, tokens, tool calls, valid-action rate, retrieval quality, and the p50 finite-state speedup. Per-run trajectories are retained in the output for auditing. This measures **controller inference overhead**, not answer-generation quality: both controllers use the same deterministic answer operation after selecting `ANSWER`.
