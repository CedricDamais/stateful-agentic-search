from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .actions import Action


class TaskContext(BaseModel):
    """Structured, query-level interpretation of the user's search request."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, description="What the user wants to find or establish.")
    query: str = Field(min_length=1, description="Initial concise retrieval query.")
    scope: str = Field(default="corpus", description="Searchable source or repository scope.")
    include_globs: list[str] = Field(default_factory=list)
    exclude_globs: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class SearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=100)


class RerankArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    document_ids: list[str] = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=100)


class RewriteArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1)
    current_query: str = Field(min_length=1)


class AnswerArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1)
    document_ids: list[str] = Field(min_length=1)


class StopArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)


class GrepArguments(BaseModel):
    """Typed contract for a future repository/text grep tool."""

    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(min_length=1, description="Literal text or regular expression.")
    path: str = Field(default=".", description="File or directory search root.")
    glob: str = Field(default="*", description="File-name glob filter.")
    fixed_strings: bool = False
    case_sensitive: bool = True
    context_lines: int = Field(default=0, ge=0, le=20)


_ARGUMENT_SCHEMAS: dict[Action, type[BaseModel]] = {
    Action.VECTOR_SEARCH: SearchArguments,
    Action.BM25_SEARCH: SearchArguments,
    Action.HYBRID_SEARCH: SearchArguments,
    Action.RERANK: RerankArguments,
    Action.REWRITE_QUERY: RewriteArguments,
    Action.ANSWER: AnswerArguments,
    Action.STOP: StopArguments,
}


class ToolCall(BaseModel):
    """Validated action plus exactly that tool's argument object."""

    model_config = ConfigDict(extra="forbid")

    action: Action
    arguments: dict[str, Any]
    argument_sources: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_arguments_for_action(self) -> "ToolCall":
        schema = _ARGUMENT_SCHEMAS[self.action]
        validated = schema.model_validate(self.arguments)
        self.arguments = validated.model_dump()
        return self

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "arguments": self.arguments,
            "argument_sources": self.argument_sources,
        }
