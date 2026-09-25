from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .actions import Action
from .schemas import TaskContext


@dataclass
class Document:
    id: str
    title: str
    text: str
    score: float = 0.0

    def compact(self, chars: int = 220) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "score": round(self.score, 5), "text": self.text[:chars]}


@dataclass
class AgentState:
    question: str
    query: str | None = None
    documents: list[Document] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    step: int = 0
    max_steps: int = 5
    answer: str | None = None
    task_context: TaskContext | None = None

    def __post_init__(self) -> None:
        self.query = self.query or (self.task_context.query if self.task_context else self.question)
        if self.task_context is None:
            self.task_context = TaskContext(goal=self.question, query=self.query)

    @property
    def available_actions(self) -> tuple[Action, ...]:
        """Actions that are meaningful in this state."""
        actions = [
            Action.VECTOR_SEARCH,
            Action.BM25_SEARCH,
            Action.HYBRID_SEARCH,
            Action.REWRITE_QUERY,
        ]
        if self.documents:
            actions.extend((Action.RERANK, Action.ANSWER))
        searched = any(
            item.get("action") in {
                Action.VECTOR_SEARCH.value,
                Action.BM25_SEARCH.value,
                Action.HYBRID_SEARCH.value,
            }
            for item in self.history
        )
        if searched and not self.documents and self.max_steps - self.step <= 1:
            actions.append(Action.STOP)
        return tuple(actions)

    def policy_view(self) -> dict[str, Any]:
        """Bounded, JSON-serializable state representation given to a policy."""
        return {
            "question": self.question,
            "query": self.query,
            "task_context": self.task_context.model_dump(mode="json"),
            "step": self.step,
            "remaining_steps": self.max_steps - self.step,
            "available_actions": [action.value for action in self.available_actions],
            "previous_actions": [item["action"] for item in self.history],
            "documents": [document.compact() for document in self.documents[:5]],
        }


@dataclass
class ActionDecision:
    probabilities: dict[Action, float]
    selected: Action | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_latency_ms: float = 0.0
    raw_output: str | None = None
    valid: bool = True


@dataclass
class TrajectoryStep:
    step: int
    action: str
    probabilities: dict[str, float]
    observation: dict[str, Any]
    latency_ms: float
    model_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    tool_calls: int
    policy_output: str | None = None
    policy_valid: bool = True
    tool_call: dict[str, Any] | None = None


@dataclass
class AgentResult:
    question: str
    answer: str | None
    final_documents: list[Document]
    trajectory: list[TrajectoryStep]
    terminated_by: str
    total_latency_ms: float
    state_interpretation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["final_documents"] = [d.compact(1000) for d in self.final_documents]
        return result
