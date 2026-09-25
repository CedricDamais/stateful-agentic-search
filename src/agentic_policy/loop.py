from __future__ import annotations

import time

from .actions import Action
from .policy import ActionPolicy
from .retrieval import RetrievalTools
from .state import AgentResult, AgentState, TrajectoryStep


class AgentLoop:
    def __init__(self, policy: ActionPolicy, tools: RetrievalTools, top_k: int = 3) -> None:
        self.policy, self.tools, self.top_k = policy, tools, top_k

    def run(self, question: str, max_steps: int = 4) -> AgentResult:
        state, steps = AgentState(question=question, max_steps=max_steps), []
        started, terminated_by = time.perf_counter(), "MAX_STEPS"
        for index in range(max_steps):
            state.step = index
            action_started = time.perf_counter()
            decision = self.policy.decide(state)
            action, tool_calls = decision.selected, 0
            if action is None:
                action_name = "INVALID_OUTPUT"
                observation = {"error": "Policy output did not contain an allowed action; see policy_output."}
            elif action is Action.BM25_SEARCH:
                state.documents, tool_calls = self.tools.bm25(state.query, self.top_k), 1
                observation = {"result_ids": [d.id for d in state.documents], "query": state.query}
            elif action is Action.VECTOR_SEARCH:
                state.documents, tool_calls = self.tools.vector(state.query, self.top_k), 1
                observation = {"result_ids": [d.id for d in state.documents], "query": state.query}
            elif action is Action.HYBRID_SEARCH:
                state.documents, tool_calls = self.tools.hybrid(state.query, self.top_k), 1
                observation = {"result_ids": [d.id for d in state.documents], "query": state.query}
            elif action is Action.RERANK:
                state.documents, tool_calls = self.tools.rerank(state.query, state.documents, self.top_k), 1
                observation = {"result_ids": [d.id for d in state.documents]}
            elif action is Action.REWRITE_QUERY:
                state.query, tool_calls = self.tools.rewrite(state.query), 1
                observation = {"rewritten_query": state.query}
            elif action is Action.ANSWER:
                state.answer = state.documents[0].text if state.documents else "Insufficient evidence retrieved."
                observation, terminated_by = {"answer": state.answer}, action.value
            else:
                observation, terminated_by = {"reason": "policy stopped"}, action.value
            if action is not None:
                action_name = action.value
            step = TrajectoryStep(
                step=index, action=action_name,
                probabilities={a.value: round(p, 8) for a, p in decision.probabilities.items()},
                observation=observation, latency_ms=(time.perf_counter() - action_started) * 1000,
                model_latency_ms=decision.model_latency_ms, prompt_tokens=decision.prompt_tokens,
                completion_tokens=decision.completion_tokens, tool_calls=tool_calls,
                policy_output=decision.raw_output,
                policy_valid=decision.valid,
            )
            steps.append(step)
            state.history.append({"action": action_name, "observation": observation})
            if action in {Action.ANSWER, Action.STOP}:
                break
            if action is None and index == max_steps - 1:
                terminated_by = "INVALID_POLICY_OUTPUT"
        return AgentResult(question, state.answer, state.documents, steps, terminated_by, (time.perf_counter() - started) * 1000)
