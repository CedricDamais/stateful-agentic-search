from __future__ import annotations

from .actions import Action
from .schemas import ToolCall
from .state import AgentState


def build_tool_call(state: AgentState, action: Action, top_k: int = 3) -> ToolCall:
    """Construct and validate action-specific arguments from the current state."""
    context = state.task_context
    sources: dict[str, str] = {}
    if action in {Action.VECTOR_SEARCH, Action.BM25_SEARCH, Action.HYBRID_SEARCH}:
        sources.update({"query": "current state query", "top_k": "loop configuration"})
        arguments = {"query": state.query, "top_k": top_k}
    elif action is Action.RERANK:
        sources.update({"query": "current state query", "document_ids": "current retrieved evidence", "top_k": "loop configuration"})
        arguments = {
            "query": state.query,
            "document_ids": [document.id for document in state.documents],
            "top_k": top_k,
        }
    elif action is Action.REWRITE_QUERY:
        sources.update({"goal": "initial task context", "current_query": "current state query"})
        arguments = {"goal": context.goal, "current_query": state.query}
    elif action is Action.ANSWER:
        sources.update({"goal": "initial task context", "document_ids": "current retrieved evidence"})
        arguments = {"goal": context.goal, "document_ids": [document.id for document in state.documents]}
    elif action is Action.STOP:
        sources["reason"] = "state and action history"
        arguments = {"reason": "search budget exhausted without usable evidence"}
    else:
        raise ValueError(f"No argument schema is registered for action {action!r}")
    return ToolCall(action=action, arguments=arguments, argument_sources=sources)
