from __future__ import annotations

from nemo_polar_bridge.datasets.base import TaskSpec
from nemo_polar_bridge.datasets.verifiers import portable_verifier_source, verify_completion


def _task(*, answer: str, source_dataset: str) -> TaskSpec:
    return TaskSpec(
        task_id="task-1",
        responses_create_params={"input": [{"role": "user", "content": "Solve."}]},
        prompt="Solve.",
        answer=answer,
        dataset_id="dataset",
        source_dataset=source_dataset,
    )


def test_cryptarithm_verifier_compares_assignment_maps() -> None:
    task = _task(answer="D=8,H=4,J=0,K=1,U=2,W=5", source_dataset="cryptarithm")

    result = verify_completion(
        "Reasoning omitted.\nFinal answer: W = 5, U=2, D=8, J=0, K=1, H=4",
        task,
    )

    assert result.passed is True
    assert result.reward == 1.0
    assert result.normalized_answer == "D=8,H=4,J=0,K=1,U=2,W=5"
    assert result.metadata["verifier"] == "cryptarithm_assignment"


def test_cryptarithm_verifier_rejects_wrong_assignment() -> None:
    task = _task(answer="A=1,B=2", source_dataset="cryptarithm")

    result = verify_completion("Final answer: A=1,B=3", task)

    assert result.passed is False
    assert result.reward == 0.0
    assert result.reason == "cryptarithm_assignment_mismatch"


def test_matrix_verifier_accepts_whitespace_grid() -> None:
    task = _task(answer="0 0 0 7 5 0\n2 5 7 0 0 7", source_dataset="manipulate_matrix")

    result = verify_completion(
        "Final answer:\n0, 0, 0, 7, 5, 0\n2, 5, 7, 0, 0, 7",
        task,
    )

    assert result.passed is True
    assert result.reward == 1.0
    assert result.normalized_answer == "0 0 0 7 5 0\n2 5 7 0 0 7"
    assert result.metadata["verifier"] == "matrix_grid"


def test_portable_verifier_matches_task_specific_hooks() -> None:
    namespace: dict[str, object] = {}
    exec(portable_verifier_source(), namespace)
    verify = namespace["verify_completion"]

    cryptarithm = verify(
        "Final answer: B=2, A=1",
        {"source_dataset": "cryptarithm", "answer": "A=1,B=2"},
    )
    matrix = verify(
        "Final answer:\n1, 2\n3, 4",
        {"source_dataset": "manipulate_matrix", "answer": "1 2\n3 4"},
    )

    assert cryptarithm["passed"] is True
    assert cryptarithm["metadata"]["verifier"] == "cryptarithm_assignment"
    assert matrix["passed"] is True
    assert matrix["metadata"]["verifier"] == "matrix_grid"
