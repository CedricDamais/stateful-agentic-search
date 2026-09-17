from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from .actions import ALL_ACTIONS, Action
from .model_adapters import TransformersCausalLM
from .state import ActionDecision, AgentState


class ActionPolicy(ABC):
    @abstractmethod
    def decide(self, state: AgentState) -> ActionDecision: ...


def choose(probabilities: dict[Action, float], sampling: str, rng) -> Action:
    if sampling == "greedy":
        return max(probabilities, key=probabilities.get)
    cutoff, cumulative = rng.random(), 0.0
    for action, probability in probabilities.items():
        cumulative += probability
        if cumulative >= cutoff:
            return action
    return list(probabilities)[-1]


class HeuristicPolicy(ActionPolicy):
    """Deterministic no-model baseline; deliberately exposes a full action distribution."""

    def __init__(self, sampling: str = "greedy", seed: int = 0) -> None:
        import random
        self.sampling, self.rng = sampling, random.Random(seed)

    def decide(self, state: AgentState) -> ActionDecision:
        if state.documents:
            raw = {Action.ANSWER: 0.72, Action.RERANK: 0.20, Action.STOP: 0.08}
        elif state.step >= state.max_steps - 1:
            raw = {Action.ANSWER: 0.7, Action.STOP: 0.3}
        else:
            words = set(state.query.lower().split())
            preferred = Action.BM25_SEARCH if "bm25" in words else (Action.VECTOR_SEARCH if {"gil", "cpython"} & words else Action.HYBRID_SEARCH)
            raw = {Action.VECTOR_SEARCH: 0.2, Action.HYBRID_SEARCH: 0.13, Action.REWRITE_QUERY: 0.05}
            raw[preferred] = 0.62
        probabilities = {action: raw.get(action, 0.0) for action in ALL_ACTIONS}
        return ActionDecision(probabilities, choose(probabilities, self.sampling, self.rng))


class QwenConstrainedPolicy(ActionPolicy):
    """Scores only valid action labels under a causal Qwen model; no free-text parsing."""

    def __init__(self, model_name: str | None = None, device: str = "auto", sampling: str = "greedy", seed: int = 0,
                 runtime: TransformersCausalLM | None = None) -> None:
        import random
        if runtime is None and model_name is None:
            raise ValueError("model_name is required when runtime is not supplied")
        self.runtime = runtime or TransformersCausalLM(model_name, device)
        self.sampling, self.rng = sampling, random.Random(seed)

    def _prompt(self, state: AgentState) -> str:
        state_json = json.dumps(state.policy_view(), ensure_ascii=False)
        mapping = ", ".join(f"{code}={action.value}" for code, action in zip("ABCDEFG", ALL_ACTIONS))
        messages = [{"role": "system", "content": "Select the best next action code. Output one letter only."},
                    {"role": "user", "content": f"State: {state_json}\nAction codes: {mapping}\nCode:"}]
        return self.runtime.chat_prompt(messages)

    def decide(self, state: AgentState) -> ActionDecision:
        prompt = self._prompt(state)
        candidates = [" " + code for code in "ABCDEFG"]
        scores, prompt_tokens, latency_ms = self.runtime.score_next_tokens(prompt, candidates)
        values = self.runtime.torch.tensor(scores)
        normalized = self.runtime.torch.softmax(values, dim=0).tolist()
        probabilities = dict(zip(ALL_ACTIONS, normalized))
        selected = choose(probabilities, self.sampling, self.rng)
        return ActionDecision(probabilities, selected, prompt_tokens=prompt_tokens, completion_tokens=1,
                              model_latency_ms=latency_ms)


class QwenReActPolicy(ActionPolicy):
    """Free-form reasoning/action baseline using the same state and model runtime."""

    def __init__(self, model_name: str | None = None, device: str = "auto", max_new_tokens: int = 96,
                 runtime: TransformersCausalLM | None = None) -> None:
        if runtime is None and model_name is None:
            raise ValueError("model_name is required when runtime is not supplied")
        self.runtime = runtime or TransformersCausalLM(model_name, device)
        self.max_new_tokens = max_new_tokens

    def _prompt(self, state: AgentState) -> str:
        tools = """VECTOR_SEARCH: semantic retrieval
BM25_SEARCH: exact keyword retrieval
HYBRID_SEARCH: combine semantic and keyword retrieval
RERANK: reorder the current documents
REWRITE_QUERY: simplify the current query
ANSWER: answer when the retrieved documents are sufficient
STOP: stop only if no useful next action exists"""
        messages = [
            {"role": "system", "content": (
                "You control a retrieval agent. Think briefly, then choose exactly one tool name from this list:\n"
                f"{tools}\nNever invent or abbreviate a tool name. End with exactly `Action: <TOOL_NAME>`."
            )},
            {"role": "user", "content": (
                "Example response: `Thought: I need exact keyword evidence. Action: BM25_SEARCH`\n"
                "Current state:\n" + json.dumps(state.policy_view(), ensure_ascii=False)
            )},
        ]
        return self.runtime.chat_prompt(messages)

    def decide(self, state: AgentState) -> ActionDecision:
        generation = self.runtime.generate(self._prompt(state), self.max_new_tokens)
        matches = re.findall(r"Action\s*:\s*(" + "|".join(a.value for a in ALL_ACTIONS) + r")", generation.text, re.I)
        valid = bool(matches)
        selected = Action(matches[-1].upper()) if valid else Action.STOP
        probabilities = {action: float(action is selected) for action in ALL_ACTIONS}
        return ActionDecision(probabilities, selected, generation.prompt_tokens, generation.completion_tokens,
                              generation.latency_ms, generation.text, valid)
