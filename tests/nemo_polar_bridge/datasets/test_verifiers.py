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


def test_calendar_gym_verifier_accepts_valid_state_and_ignores_null_placeholders() -> None:
    task = TaskSpec(
        task_id="calendar-1",
        responses_create_params={"input": [{"role": "user", "content": "Schedule it."}]},
        prompt="Schedule it.",
        answer=(
            '{"0":{"event_id":0,"duration":60,"constraint":"between 2pm and 4pm",'
            '"min_time":"10:00","max_time":"16:00"},"1":null}'
        ),
        dataset_id="nvidia/Nemotron-RL-agent-calendar_scheduling",
        source_dataset="calendar",
        verifier_name="calendar_gym",
    )

    result = verify_completion(
        '[{"event_id": 0, "event_name": "Review", "start_time": "14:00", "duration": 60}]',
        task,
    )

    assert result.passed is True
    assert result.reward == 1.0
    assert result.reason == "calendar_gym_pass"
    assert result.metadata["verifier"] == "calendar_gym"
    assert result.metadata["expected_event_count"] == 1


def test_calendar_gym_verifier_rejects_constraint_violation_and_think_tags() -> None:
    task = TaskSpec(
        task_id="calendar-1",
        responses_create_params={"input": [{"role": "user", "content": "Schedule it."}]},
        prompt="Schedule it.",
        answer=(
            '{"0":{"event_id":0,"duration":60,"constraint":"at 10am",'
            '"min_time":"10:00","max_time":"16:00"}}'
        ),
        dataset_id="nvidia/Nemotron-RL-agent-calendar_scheduling",
        source_dataset="calendar",
        verifier_name="calendar_gym",
    )

    wrong_time = verify_completion(
        '[{"event_id": 0, "event_name": "Review", "start_time": "11:00", "duration": 60}]',
        task,
    )
    think_tag = verify_completion(
        '<think>hidden</think>[{"event_id": 0, "start_time": "10:00", "duration": 60}]',
        task,
    )

    assert wrong_time.passed is False
    assert wrong_time.reason == "calendar_gym_constraint_violated"
    assert think_tag.passed is False
    assert think_tag.reason == "calendar_gym_think_found"


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
    calendar = verify(
        '[{"event_id": 0, "start_time": "2pm", "duration": 60}]',
        {
            "source_dataset": "calendar",
            "verifier_name": "calendar_gym",
            "answer": (
                '{"0":{"event_id":0,"duration":60,"constraint":"between 2pm and 4pm",'
                '"min_time":"10:00","max_time":"16:00"},"1":null}'
            ),
        },
    )

    assert cryptarithm["passed"] is True
    assert cryptarithm["metadata"]["verifier"] == "cryptarithm_assignment"
    assert matrix["passed"] is True
    assert matrix["metadata"]["verifier"] == "matrix_grid"
    assert calendar["passed"] is True
    assert calendar["metadata"]["verifier"] == "calendar_gym"
