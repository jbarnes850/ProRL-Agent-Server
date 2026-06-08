"""Run-matrix metadata for dataset/verifier/execution ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunMatrixCell:
    """One experiment-matrix cell, independent of trainer and rollout plumbing."""

    name: str
    dataset_family: str
    verifier_type: str
    execution_type: str
    adapter: str
    difficulty_band: str = "unspecified"
    source_dataset_filter: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "dataset_family",
            "verifier_type",
            "execution_type",
            "adapter",
            "difficulty_band",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"run matrix {field_name} must be a non-empty string")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset_family": self.dataset_family,
            "verifier_type": self.verifier_type,
            "execution_type": self.execution_type,
            "adapter": self.adapter,
            "difficulty_band": self.difficulty_band,
            "source_dataset_filter": list(self.source_dataset_filter),
            "tags": list(self.tags),
            "notes": self.notes,
            "organization_contract": (
                "Ablations are grouped by dataset family, verifier type, and "
                "execution type. Task names or source_dataset filters are selection "
                "metadata, not the primary matrix axis."
            ),
        }


def build_run_matrix_cell(
    *,
    matrix_name: str,
    matrix_cell: str,
    dataset_family: str,
    verifier_type: str,
    execution_type: str,
    adapter: str,
    difficulty_band: str,
    source_dataset_filter: list[str],
    tags: list[str],
    notes: str,
) -> RunMatrixCell:
    cell_name = matrix_cell.strip() if matrix_cell.strip() else _default_cell_name(
        verifier_type=verifier_type,
        execution_type=execution_type,
        difficulty_band=difficulty_band,
    )
    return RunMatrixCell(
        name=f"{matrix_name.strip()}:{cell_name}",
        dataset_family=dataset_family.strip(),
        verifier_type=verifier_type.strip(),
        execution_type=execution_type.strip(),
        adapter=adapter.strip(),
        difficulty_band=difficulty_band.strip(),
        source_dataset_filter=tuple(value for value in source_dataset_filter if value),
        tags=tuple(value for value in tags if value),
        notes=notes.strip(),
    )


def _default_cell_name(
    *,
    verifier_type: str,
    execution_type: str,
    difficulty_band: str,
) -> str:
    parts = [verifier_type, execution_type]
    if difficulty_band and difficulty_band != "unspecified":
        parts.append(difficulty_band)
    return "-".join(_safe_part(part) for part in parts if part)


def _safe_part(value: str) -> str:
    safe = "".join(char if char.isalnum() else "-" for char in value.lower())
    safe = "-".join(part for part in safe.split("-") if part)
    return safe or "unspecified"
