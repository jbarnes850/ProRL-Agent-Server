"""Generate reasoning-gym procedural tasks and normalize them into Polar task specs.

Unlike ``NeMoGymDatasetAdapter`` (which pulls pre-materialized NeMo Gym-shaped rows
from an HF dataset or local JSONL), this adapter calls directly into the
``reasoning-gym`` package (https://github.com/open-thought/reasoning-gym) to
generate tasks procedurally via ``reasoning_gym.create_dataset(name, **kwargs)``.
``reasoning-gym`` is an optional dependency (see the ``reasoning-gym`` extra in
``pyproject.toml``); importing this module must not fail when it is not
installed, so the import is guarded the same way ``slime_bridge/data_source.py``
guards its optional ``slime`` import.
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata
from typing import Any

from nemo_polar_bridge.datasets.base import DatasetAdapter, TaskSpec

try:
    import reasoning_gym
except ImportError as exc:  # pragma: no cover - exercised only when the extra is absent
    _REASONING_GYM_IMPORT_MESSAGE = str(exc)

    class _MissingReasoningGym:
        """Placeholder that raises a clear error the first time it is used."""

        def create_dataset(self, *args: Any, **kwargs: Any) -> Any:
            raise ImportError(
                "reasoning-gym is required to use ReasoningGymDatasetAdapter. Install it "
                "with `pip install reasoning-gym` or `uv sync --extra reasoning-gym`: "
                f"{_REASONING_GYM_IMPORT_MESSAGE}"
            )

    reasoning_gym = _MissingReasoningGym()  # type: ignore[assignment]
else:
    _REASONING_GYM_IMPORT_MESSAGE = ""


# Verified against reasoning-gym==0.1.25 (PyPI, fetched 2026-06-30):
# `reasoning_gym.create_dataset("basic_arithmetic", size=..., seed=...)` returns a
# `reasoning_gym.arithmetic.basic_arithmetic.BasicArithmeticDataset` instance whose
# `DATASET_NAME` class constant is the literal string "basic_arithmetic" (registered
# via `register_dataset(DATASET_NAME, BasicArithmeticDataset, ...)` in
# reasoning_gym/arithmetic/basic_arithmetic.py). Confirmed live by constructing the
# dataset and indexing it -- not assumed from documentation.
DEFAULT_TASK_NAME = "basic_arithmetic"

# New verifier_name (per adapter contract point (d)): dispatched in verifiers.py to a
# faithful port of reasoning-gym's own `score_answer` default implementation, kept
# distinct from the base "exact_normalized" verifier because the scoring semantics
# (fractional substring-containment partial credit) differ from exact/normalized match.
REASONING_GYM_VERIFIER_NAME = "reasoning_gym_basic_arithmetic"

REASONING_GYM_LICENSE = "Apache-2.0"


class ReasoningGymDatasetAdapter(DatasetAdapter):
    """Generate reasoning-gym tasks procedurally via ``reasoning_gym.create_dataset``.

    Only ``basic_arithmetic`` has a matching portable verifier wired into
    ``verifiers.py`` today (``reasoning_gym_basic_arithmetic``). The ``task_name``
    constructor argument exists so other reasoning-gym procedural datasets can be
    swapped in later without touching the ``DatasetAdapter`` contract, but selecting
    a different ``task_name`` will produce tasks whose ``verifier_name`` still points
    at the basic_arithmetic port -- callers must add a matching verifier branch
    before using a different task family for real training.
    """

    name = "reasoning_gym"

    def __init__(
        self,
        *,
        task_name: str = DEFAULT_TASK_NAME,
        seed: int = 0,
        split: str = "train",
        config: str = "default",
        task_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.task_name = task_name
        self.seed = seed
        self.split = split
        self.config = config
        self.task_kwargs = dict(task_kwargs or {})
        self.dataset_id = f"reasoning-gym/{task_name}"
        self._audit: dict[str, Any] = {}

    def load_tasks(self, *, limit: int, scan_rows: int) -> list[TaskSpec]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if scan_rows <= 0:
            raise ValueError("scan_rows must be positive")

        dataset = reasoning_gym.create_dataset(
            self.task_name,
            size=scan_rows,
            seed=self.seed,
            **self.task_kwargs,
        )

        inspected_rows: list[dict[str, Any]] = []
        tasks: list[TaskSpec] = []
        skipped = 0
        for row_idx in range(scan_rows):
            item = dataset[row_idx]
            inspected_rows.append({"row_idx": row_idx, "item": item})
            task = task_from_reasoning_gym_item(
                item,
                task_name=self.task_name,
                dataset_id=self.dataset_id,
                config=self.config,
                split=self.split,
                row_idx=row_idx,
                seed=self.seed,
            )
            if task is None:
                skipped += 1
                continue
            tasks.append(task)
            if len(tasks) >= limit:
                break

        self._audit = {
            "dataset_id": self.dataset_id,
            "config": self.config,
            "split": self.split,
            "loader": self.name,
            "canonical_schema": "reasoning_gym_procedural",
            "required_fields": ["question", "answer"],
            "optional_fields": ["metadata"],
            "reasoning_gym_task_name": self.task_name,
            "reasoning_gym_seed": self.seed,
            "reasoning_gym_task_kwargs": self.task_kwargs,
            "reasoning_gym_package_version": _reasoning_gym_version(),
            "scanned_rows": len(inspected_rows),
            "skipped_unmappable_rows": skipped,
            "selected_tasks": [task.stable_id() for task in tasks],
            "inspected_rows": [_summarize_row(wrapped) for wrapped in inspected_rows[:10]],
        }
        if len(tasks) < limit:
            raise ValueError(
                f"only found {len(tasks)} usable reasoning-gym tasks for task_name="
                f"{self.task_name!r} after scanning {len(inspected_rows)} rows"
            )
        return tasks

    def audit(self) -> dict[str, Any]:
        return dict(self._audit)


def task_from_reasoning_gym_item(
    item: dict[str, Any],
    *,
    task_name: str,
    dataset_id: str,
    config: str,
    split: str,
    row_idx: int,
    seed: int,
) -> TaskSpec | None:
    """Build a ``TaskSpec`` from one ``reasoning_gym`` dataset item.

    Item shape (verified live against reasoning-gym==0.1.25's
    ``BasicArithmeticDataset.__getitem__``): ``{"question": str, "answer": str,
    "metadata": dict}``. Mirrors ``task_from_gym_row``'s field semantics in
    ``data_loader.py`` so ``prepare_dataset.py``'s artifact-writing logic needs no
    changes to consume tasks from this adapter.
    """

    question = item.get("question")
    answer = item.get("answer")
    if not question or answer in (None, ""):
        return None

    prompt = str(question).strip()
    responses_create_params = {"input": [{"role": "user", "content": prompt}]}

    metadata = dict(item.get("metadata") or {})
    metadata["reasoning_gym_task_name"] = task_name
    metadata["reasoning_gym_seed"] = seed

    task_id = f"reasoning-gym:{task_name}:{seed}:{row_idx}"

    return TaskSpec(
        task_id=task_id,
        responses_create_params=responses_create_params,
        prompt=prompt,
        answer=str(answer),
        dataset_id=dataset_id,
        split=split,
        config=config,
        row_idx=row_idx,
        source_dataset=task_name,
        cohort=task_name,
        metadata=metadata,
        agent_ref=None,
        verifier_name=REASONING_GYM_VERIFIER_NAME,
        license=REASONING_GYM_LICENSE,
    )


def _summarize_row(wrapped: dict[str, Any]) -> dict[str, Any]:
    item = wrapped["item"]
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "row_idx": wrapped["row_idx"],
        "keys": sorted(item.keys()) if isinstance(item, dict) else [],
        "question_head": str(item.get("question", ""))[:240],
        "answer": item.get("answer"),
        "metadata_head": {key: metadata[key] for key in sorted(metadata)[:8]},
    }


def _reasoning_gym_version() -> str | None:
    try:
        return importlib_metadata.version("reasoning-gym")
    except importlib_metadata.PackageNotFoundError:
        return None
