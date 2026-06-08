"""NeMo Gym-shaped dataset contract for Polar-backed NeMo Async GRPO runs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class VerifierResult:
    """Result of scoring one live model completion against one task."""

    passed: bool
    reward: float
    reason: str
    normalized_completion: str
    normalized_answer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reward": self.reward,
            "reason": self.reason,
            "normalized_completion": self.normalized_completion,
            "normalized_answer": self.normalized_answer,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TaskSpec:
    """One prompt/task row that can be rendered into a Polar rollout task."""

    task_id: str
    responses_create_params: dict[str, Any]
    answer: str
    dataset_id: str
    prompt: str = ""
    split: str = "train"
    config: str = "default"
    row_idx: int | None = None
    source_dataset: str | None = None
    cohort: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_ref: dict[str, Any] | None = None
    verifier_name: str = "exact_normalized"
    license: str | None = None

    def stable_id(self) -> str:
        if self.task_id:
            return self.task_id
        row_part = "unknown" if self.row_idx is None else str(self.row_idx)
        source = self.source_dataset or self.cohort or "default"
        return f"{self.dataset_id}:{self.split}:{source}:{row_part}"

    def to_train_row(self) -> dict[str, Any]:
        """Return the minimal JSONL row NeMo's placeholder dataloader needs."""

        return {
            "input": self.prompt,
            "output": self.answer,
            "dataset_id": self.dataset_id,
            "task_id": self.stable_id(),
            "source_dataset": self.source_dataset,
        }

    def to_attempt(self) -> dict[str, Any]:
        """Return a JSON-serializable task record consumed by the Polar collector."""

        task_json = {
            "task_id": self.stable_id(),
            "responses_create_params": self.responses_create_params,
            "prompt": self.prompt,
            "answer": self.answer,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "config": self.config,
            "row_idx": self.row_idx,
            "source_dataset": self.source_dataset,
            "cohort": self.cohort,
            "metadata": self.metadata,
            "agent_ref": self.agent_ref,
            "verifier_name": self.verifier_name,
            "license": self.license,
        }
        return {
            "name": _safe_name(self.stable_id()),
            "task_uid": self.stable_id(),
            "dataset_id": self.dataset_id,
            "split": self.split,
            "config": self.config,
            "row_idx": "" if self.row_idx is None else str(self.row_idx),
            "source_dataset": self.source_dataset or "",
            "cohort": self.cohort or self.source_dataset or "",
            "agent_ref": json.dumps(self.agent_ref or {}, sort_keys=True),
            "answer": self.answer,
            "verifier_name": self.verifier_name,
            "task_spec_json": json.dumps(task_json, sort_keys=True),
        }


class DatasetAdapter(ABC):
    """Abstract source for verifier-backed RL tasks."""

    name: str

    @abstractmethod
    def load_tasks(self, *, limit: int, scan_rows: int) -> list[TaskSpec]:
        """Load up to ``limit`` usable tasks after scanning at most ``scan_rows`` rows."""

    @abstractmethod
    def audit(self) -> dict[str, Any]:
        """Return loader/schema/sample information for the run manifest."""


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() else "-" for char in value.lower())
    safe = "-".join(part for part in safe.split("-") if part)
    return safe[:96] or "task"
