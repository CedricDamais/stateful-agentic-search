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
