from agentic_policy.evaluation import evaluate
from agentic_policy.loop import AgentLoop
from agentic_policy.policy import HeuristicPolicy
from agentic_policy.retrieval import RetrievalTools


def test_retrieval_metrics():
    loop = AgentLoop(HeuristicPolicy(), RetrievalTools.from_jsonl("data/demo_corpus.jsonl"))
    result = loop.run("What does BM25 use to rank documents?")
    metrics = evaluate([(result, {"relevant_ids": ["bm25"], "expected_first_action": "BM25_SEARCH"})])
    assert metrics["recall_at_3"] == 1
    assert metrics["first_action_accuracy"] == 1
