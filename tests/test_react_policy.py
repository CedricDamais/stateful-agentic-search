from agentic_policy.actions import Action
from agentic_policy.model_adapters import Generation
from agentic_policy.policy import QwenReActPolicy
from agentic_policy.state import AgentState


class FakeRuntime:
    def chat_prompt(self, messages):
        return "prompt"

    def generate(self, prompt, max_new_tokens):
        return Generation("Thought: lexical match.\nAction: BM25_SEARCH", 20, 8, 5.0)


def test_react_policy_parses_generated_action():
    decision = QwenReActPolicy(runtime=FakeRuntime()).decide(AgentState("question"))
    assert decision.selected is Action.BM25_SEARCH
    assert decision.completion_tokens == 8
    assert decision.raw_output.startswith("Thought:")


def test_invalid_react_output_is_not_silently_converted_to_stop():
    runtime = FakeRuntime()
    runtime.generate = lambda prompt, max_new_tokens: Generation(
        "Action: QUERY", 20, 3, 5.0
    )
    decision = QwenReActPolicy(runtime=runtime).decide(AgentState("question"))
    assert decision.selected is None
    assert not decision.valid
    assert decision.raw_output == "Action: QUERY"


def test_available_actions_follow_state():
    initial = AgentState("question")
    assert Action.ANSWER not in initial.available_actions
    assert Action.RERANK not in initial.available_actions
    assert Action.STOP not in initial.available_actions

    searched_empty = AgentState(
        "question",
        history=[{"action": Action.BM25_SEARCH.value, "observation": {"result_ids": []}}],
        step=3,
        max_steps=4,
    )
    assert Action.STOP in searched_empty.available_actions

    with_documents = AgentState(
        "question",
        documents=[],
        history=[{"action": Action.BM25_SEARCH.value, "observation": {"result_ids": ["d1"]}}],
    )
    from agentic_policy.state import Document
    with_documents.documents = [Document("d1", "title", "body")]
    assert Action.ANSWER in with_documents.available_actions
    assert Action.RERANK in with_documents.available_actions
    assert Action.STOP not in with_documents.available_actions
