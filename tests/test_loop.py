from agentic_policy.loop import AgentLoop
from agentic_policy.policy import HeuristicPolicy
from agentic_policy.retrieval import RetrievalTools


def test_loop_produces_logged_trajectory():
    tools = RetrievalTools.from_jsonl("data/demo_corpus.jsonl")
    result = AgentLoop(HeuristicPolicy(), tools).run("What does BM25 use to rank documents?")
    assert result.trajectory[0].action == "BM25_SEARCH"
    assert result.trajectory[-1].action == "ANSWER"
    assert abs(sum(result.trajectory[0].probabilities.values()) - 1) < 1e-6
    assert result.final_documents[0].id == "bm25"
