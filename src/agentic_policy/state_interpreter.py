from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from .model_adapters import TransformersCausalLM
from .schemas import TaskContext


@dataclass
class StateInterpretation:
    context: TaskContext
    interpreter: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw_output: str | None = None

    def to_dict(self) -> dict:
        return {
            "context": self.context.model_dump(mode="json"),
            "interpreter": self.interpreter,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "raw_output": self.raw_output,
        }


class TaskStateInterpreter(Protocol):
    def interpret(self, question: str) -> StateInterpretation: ...


class PassthroughStateInterpreter:
    """No-model baseline: preserve the user's request as goal and initial query."""

    def interpret(self, question: str) -> StateInterpretation:
        return StateInterpretation(
            context=TaskContext(goal=question, query=question),
            interpreter="passthrough",
        )


class QwenTaskStateInterpreter:
    """One initial SLM call that converts the request into validated task context."""

    def __init__(self, runtime: TransformersCausalLM) -> None:
        self.runtime = runtime

    def interpret(self, question: str) -> StateInterpretation:
        schema = json.dumps(TaskContext.model_json_schema(), ensure_ascii=False)
        prompt = self.runtime.chat_prompt([
            {
                "role": "system",
                "content": (
                    "Convert the user's search request into task context. "
                    "Return only one JSON object matching this JSON Schema. "
                    "Preserve explicit paths, patterns, and restrictions; do not invent them. "
                    "Use the original request as the query when no better concise query is clear.\n"
                    + schema
                ),
            },
            {"role": "user", "content": question},
        ])
        generated = self.runtime.generate(prompt, max_new_tokens=192)
        start, end = generated.text.find("{"), generated.text.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"State interpreter returned no JSON object: {generated.text!r}")
        try:
            context = TaskContext.model_validate_json(generated.text[start:end + 1])
        except Exception as exc:
            raise ValueError(f"State interpreter returned invalid task context: {generated.text!r}") from exc
        return StateInterpretation(
            context=context,
            interpreter="qwen",
            prompt_tokens=generated.prompt_tokens,
            completion_tokens=generated.completion_tokens,
            latency_ms=generated.latency_ms,
            raw_output=generated.text,
        )
