from __future__ import annotations

import time

from .actions import Action
from .arguments import build_tool_call
from .policy import ActionPolicy
from .retrieval import RetrievalTools
from .schemas import TaskContext
from .state import AgentResult, AgentState, TrajectoryStep
from .state_interpreter import PassthroughStateInterpreter, TaskStateInterpreter


class AgentLoop:
    def __init__(
        self,
        policy: ActionPolicy,
        tools: RetrievalTools,
        top_k: int = 3,
        state_interpreter: TaskStateInterpreter | None = None,
    ) -> None:
        self.policy, self.tools, self.top_k = policy, tools, top_k
        self.state_interpreter = state_interpreter or PassthroughStateInterpreter()

    def run(self, question: str, max_steps: int = 4) -> AgentResult:
        started = time.perf_counter()
        interpretation = self.state_interpreter.interpret(question)
        task_context: TaskContext = interpretation.context
        state = AgentState(
            question=question,
            query=task_context.query,
            task_context=task_context,
            max_steps=max_steps,
        )
        steps, terminated_by = [], "MAX_STEPS"
        for index in range(max_steps):
            state.step = index
            action_started = time.perf_counter()
            decision = self.policy.decide(state)
            action, tool_calls = decision.selected, 0
            tool_call = build_tool_call(state, action, self.top_k) if action is not None else None
            args = tool_call.arguments if tool_call else {}
            if action is None:
                action_name = "INVALID_OUTPUT"
                observation = {"error": "Policy output did not contain an allowed action; see policy_output."}
            elif action in {Action.BM25_SEARCH, Action.VECTOR_SEARCH, Action.HYBRID_SEARCH}:
                search = getattr(self.tools, {
                    Action.BM25_SEARCH: "bm25",
                    Action.VECTOR_SEARCH: "vector",
                    Action.HYBRID_SEARCH: "hybrid",
                }[action])
                state.documents, tool_calls = search(args["query"], args["top_k"]), 1
                observation = {
                    "result_ids": [document.id for document in state.documents],
                    "query": args["query"],
                }
            elif action is Action.RERANK:
                state.documents = self.tools.rerank(args["query"], state.documents, args["top_k"])
                tool_calls = 1
                observation = {"result_ids": [document.id for document in state.documents]}
            elif action is Action.REWRITE_QUERY:
                state.query = self.tools.rewrite(args["current_query"])
                tool_calls = 1
                observation = {"rewritten_query": state.query}
            elif action is Action.ANSWER:
                state.answer = state.documents[0].text
                observation, terminated_by = {"answer": state.answer}, action.value
            else:
                observation, terminated_by = {"reason": args["reason"]}, action.value

            action_name = action.value if action is not None else action_name
            trajectory_step = TrajectoryStep(
                step=index,
                action=action_name,
                probabilities={candidate.value: round(probability, 8)
                               for candidate, probability in decision.probabilities.items()},
                observation=observation,
                latency_ms=(time.perf_counter() - action_started) * 1000,
                model_latency_ms=decision.model_latency_ms,
                prompt_tokens=decision.prompt_tokens,
                completion_tokens=decision.completion_tokens,
                tool_calls=tool_calls,
                policy_output=decision.raw_output,
                policy_valid=decision.valid,
                tool_call=tool_call.to_log_dict() if tool_call else None,
            )
            steps.append(trajectory_step)
            state.history.append({
                "action": action_name,
                "arguments": tool_call.arguments if tool_call else None,
                "observation": observation,
            })
            if action in {Action.ANSWER, Action.STOP}:
                break
            if action is None and index == max_steps - 1:
                terminated_by = "INVALID_POLICY_OUTPUT"

        return AgentResult(
            question=question,
            answer=state.answer,
            final_documents=state.documents,
            trajectory=steps,
            terminated_by=terminated_by,
            total_latency_ms=(time.perf_counter() - started) * 1000,
            state_interpretation=interpretation.to_dict(),
        )
