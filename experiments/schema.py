"""Versioned, validated data structures for experiment inputs and outputs."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ShiftProxy(StrEnum):
    RECENCY = "recency"
    RARITY = "rarity"
    SYNTAX = "syntax"
    API = "api"


class Condition(StrEnum):
    BASELINE = "baseline"
    SHIFTED_BASELINE = "shifted_baseline"
    SHIFTED_DOCGROUND = "shifted_docground"


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    language: str
    source_reference: str
    source_license: str
    original_prompt: str
    shifted_prompt: str
    shift_proxy: ShiftProxy
    transformation: str
    semantic_invariant: str
    expected_symbol: str
    test_code: str
    reference_solution: str | None = None
    required_apis: list[str] = Field(default_factory=list)
    status: str

    @field_validator("expected_symbol")
    @classmethod
    def require_python_identifier(cls, value: str) -> str:
        if not value.replace("_", "").isalnum() or not value:
            raise ValueError("must contain only letters, digits, and underscores")
        return value

    @field_validator("task_id")
    @classmethod
    def require_stable_task_id(cls, value: str) -> str:
        if not value.replace("-", "").replace("_", "").isalnum() or not value:
            raise ValueError("must contain only letters, digits, hyphens, and underscores")
        return value

    @model_validator(mode="after")
    def require_syntax_evidence(self) -> "Task":
        if self.shift_proxy is ShiftProxy.SYNTAX and not self.semantic_invariant.strip():
            raise ValueError("syntax shifts require a semantic invariant")
        if self.shift_proxy is ShiftProxy.API and not self.required_apis:
            raise ValueError("API shifts require an explicit list of required API calls")
        return self

    def prompt_for(self, condition: Condition) -> str:
        return self.original_prompt if condition is Condition.BASELINE else self.shifted_prompt


class TaskSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    taskset_id: str
    source: str
    tasks: list[Task] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_task_ids(self) -> "TaskSet":
        identifiers = [task.task_id for task in self.tasks]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("task IDs must be unique")
        return self

    @classmethod
    def from_path(cls, path: Path) -> "TaskSet":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class RunManifest(BaseModel):
    """Configuration that must remain identical across paired conditions."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    protocol_version: str
    status: str = "pilot"
    taskset_id: str
    task_id: str
    source_reference: str = ""
    source_license: str = ""
    shift_proxy: str = ""
    transformation: str = ""
    semantic_invariant: str = ""
    original_prompt_sha256: str = ""
    shifted_prompt_sha256: str = ""
    required_apis: list[str] = Field(default_factory=list)
    documentation_snapshot_sha256: str | None = None
    docground_version: str | None = None
    prompt_decision: str | None = None
    docground_review_sha256: str | None = None
    evaluator_image: str = "python:3.14-slim"
    attempt: int = Field(default=1, ge=1)
    generation_route: str = "chat_completions"
    reasoning_effort: str | None = None
    condition: Condition
    model_provider: str
    model_id: str
    replicate: int = Field(ge=1)
    temperature: float = Field(ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int = Field(gt=0)
    seed: int | None = None
    evaluator_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    prompt_sha256: str

    @classmethod
    def create(
        cls, *, taskset_id: str, task: Task, condition: Condition, provider: str,
        model_id: str, replicate: int, temperature: float, top_p: float | None,
        max_tokens: int, seed: int | None, protocol_version: str = "1.0",
        attempt: int = 1, generation_route: str = "chat_completions",
        reasoning_effort: str | None = None,
        prompt_text: str | None = None,
        documentation_snapshot_sha256: str | None = None,
        docground_version: str | None = None,
        prompt_decision: str | None = None,
        docground_review_sha256: str | None = None,
        evaluator_image: str = "python:3.14-slim",
    ) -> "RunManifest":
        prompt_hash = sha256((prompt_text or task.prompt_for(condition)).encode()).hexdigest()
        date = datetime.now(UTC).date().isoformat()
        run_id = "__".join((date, model_id.lower().replace("/", "-"), condition.value,
                             task.shift_proxy.value, f"{taskset_id}-{task.task_id}",
                             f"r{replicate}"))
        if attempt > 1:
            run_id += f"__a{attempt}"
        return cls(
            run_id=run_id, protocol_version=protocol_version, taskset_id=taskset_id,
            task_id=task.task_id, source_reference=task.source_reference,
            source_license=task.source_license, shift_proxy=task.shift_proxy.value,
            transformation=task.transformation, semantic_invariant=task.semantic_invariant,
            original_prompt_sha256=sha256(task.original_prompt.encode()).hexdigest(),
            shifted_prompt_sha256=sha256(task.shifted_prompt.encode()).hexdigest(),
            required_apis=task.required_apis,
            documentation_snapshot_sha256=documentation_snapshot_sha256,
            docground_version=docground_version,
            prompt_decision=prompt_decision,
            docground_review_sha256=docground_review_sha256,
            evaluator_image=evaluator_image,
            attempt=attempt, generation_route=generation_route,
            reasoning_effort=reasoning_effort,
            condition=condition, model_provider=provider,
            model_id=model_id, replicate=replicate, temperature=temperature,
            top_p=top_p, max_tokens=max_tokens, seed=seed,
            evaluator_version="baseline-evaluator-1.1", prompt_sha256=prompt_hash,
        )
