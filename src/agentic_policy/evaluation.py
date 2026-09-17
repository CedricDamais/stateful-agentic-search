from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .state import AgentResult


def reciprocal_rank(result: AgentResult, relevant_ids: set[str]) -> float:
    for rank, document in enumerate(result.final_documents, start=1):
        if document.id in relevant_ids:
            return 1 / rank
    return 0.0


def ndcg_at_k(result: AgentResult, relevant_ids: set[str], k: int = 3) -> float:
    import math
    dcg = sum((1 / math.log2(rank + 1)) for rank, d in enumerate(result.final_documents[:k], 1) if d.id in relevant_ids)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant_ids)) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(records: Iterable[tuple[AgentResult, dict]]) -> dict[str, float]:
    values = defaultdict(float)
    count = 0
    for result, item in records:
        count += 1
        relevant = set(item["relevant_ids"])
        retrieved = {d.id for d in result.final_documents}
        values["retrieval_success"] += float(bool(retrieved & relevant))
        values["recall_at_3"] += len(retrieved & relevant) / len(relevant)
        values["ndcg_at_3"] += ndcg_at_k(result, relevant)
        values["mrr"] += reciprocal_rank(result, relevant)
        if result.trajectory:
            values["first_action_accuracy"] += float(result.trajectory[0].action == item.get("expected_first_action"))
        values["tool_calls"] += sum(s.tool_calls for s in result.trajectory)
        values["latency_ms"] += result.total_latency_ms
        values["prompt_tokens"] += sum(s.prompt_tokens for s in result.trajectory)
        values["completion_tokens"] += sum(s.completion_tokens for s in result.trajectory)
    return {key: round(value / count, 5) for key, value in values.items()} | {"examples": count}
