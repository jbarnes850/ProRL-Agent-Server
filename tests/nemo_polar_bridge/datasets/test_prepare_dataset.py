from __future__ import annotations

from types import SimpleNamespace

from nemo_polar_bridge.datasets.base import TaskSpec
from nemo_polar_bridge.datasets.prepare_dataset import build_attempt_matrix, build_task_template


def test_build_attempt_matrix_declares_group_cycle_contract() -> None:
    task = TaskSpec(
        task_id="task-1",
        responses_create_params={"input": [{"role": "user", "content": "2+2?"}]},
        prompt="2+2?",
        answer="4",
        dataset_id="dataset",
    )

    matrix = build_attempt_matrix([task.to_attempt()])

    assert matrix["schema"] == "nemo_gym_to_polar_attempt_matrix"
    assert matrix["selection_mode"] == "group_cycle"
    assert matrix["attempts"][0]["task_uid"] == "task-1"


def test_build_task_template_uses_live_verifier_evaluator() -> None:
    args = SimpleNamespace(
        runtime_image="polar-spark-calculator:latest",
        task_timeout_seconds=240.0,
        model_name="Qwen/Qwen3-0.6B",
        test_timeout_seconds=60.0,
        model_max_tokens=32,
        model_temperature=0.6,
        model_top_p=0.95,
        model_request_timeout_seconds=120.0,
    )

    template = build_task_template(args)

    assert template["evaluator"]["strategy"] == "test_on_output"
    assert template["evaluator"]["config"]["expected_output_json"] == {"live_verifier": "PASSED"}
    command = template["agent"]["custom_shell"]["command"]
    assert "responses_create_params" in command
    assert "verify_completion" in command
