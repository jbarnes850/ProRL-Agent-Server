from __future__ import annotations

import pytest

from nemo_polar_bridge.datasets.run_matrix import RunMatrixCell, build_run_matrix_cell


def test_build_run_matrix_cell_defaults_to_verifier_execution_axis() -> None:
    cell = build_run_matrix_cell(
        matrix_name="baseline",
        matrix_cell="",
        dataset_family="nemo_gym",
        verifier_type="exact_answer",
        execution_type="single_turn_chat",
        adapter="nemo_gym_jsonl",
        difficulty_band="unspecified",
        source_dataset_filter=["aiw"],
        tags=["smoke"],
        notes="small plumbing run",
    )

    payload = cell.to_json_dict()

    assert cell.name == "baseline:exact-answer-single-turn-chat"
    assert payload["dataset_family"] == "nemo_gym"
    assert payload["verifier_type"] == "exact_answer"
    assert payload["execution_type"] == "single_turn_chat"
    assert payload["source_dataset_filter"] == ["aiw"]
    assert "Task names or source_dataset filters" in payload["organization_contract"]


def test_run_matrix_cell_rejects_empty_primary_axes() -> None:
    with pytest.raises(ValueError, match="verifier_type"):
        RunMatrixCell(
            name="baseline:bad",
            dataset_family="nemo_gym",
            verifier_type="",
            execution_type="single_turn_chat",
            adapter="nemo_gym_jsonl",
        )
