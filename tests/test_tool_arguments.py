import pytest
from pydantic import ValidationError

from agentic_policy.actions import Action
from agentic_policy.arguments import build_tool_call
from agentic_policy.schemas import TaskContext, ToolCall
from agentic_policy.state import AgentState, Document
from agentic_policy.state_interpreter import PassthroughStateInterpreter


def test_passthrough_interpreter_preserves_question_as_goal_and_query():
    question = "Find where the retry policy is configured"
    interpreted = PassthroughStateInterpreter().interpret(question)

    assert interpreted.context.goal == question
    assert interpreted.context.query == question
    assert interpreted.interpreter == "passthrough"


def test_search_call_uses_current_query_and_records_argument_sources():
    state = AgentState(
        question="original request",
        query="retry policy configuration",
        task_context=TaskContext(goal="locate retry configuration", query="retry policy configuration"),
    )

    call = build_tool_call(state, Action.BM25_SEARCH, top_k=7)

    assert call.arguments == {"query": "retry policy configuration", "top_k": 7}
    assert call.argument_sources["query"] == "current state query"


def test_rerank_call_uses_ids_from_current_evidence():
    state = AgentState(
        question="find evidence",
        documents=[Document(id="doc-a", title="A", text="first"), Document(id="doc-b", title="B", text="second")],
    )

    call = build_tool_call(state, Action.RERANK)

    assert call.arguments["document_ids"] == ["doc-a", "doc-b"]
    assert call.arguments["query"] == "find evidence"


def test_tool_call_rejects_arguments_that_do_not_match_action_schema():
    with pytest.raises(ValidationError):
        ToolCall(action=Action.BM25_SEARCH, arguments={"pattern": "retry"})
