from __future__ import annotations

from types import SimpleNamespace

from nemo_polar_bridge.datasets.base import TaskSpec
from nemo_polar_bridge.datasets.prepare_dataset import (
    FINAL_ANSWER_INSTRUCTION,
    apply_answer_format,
    build_attempt_matrix,
    build_task_template,
    _summarize_task_provenance,
)
from nemo_polar_bridge.datasets.run_matrix import RunMatrixCell


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


def test_build_attempt_matrix_carries_run_matrix_metadata() -> None:
    task = TaskSpec(
        task_id="task-1",
        responses_create_params={"input": "2+2?"},
        prompt="2+2?",
        answer="4",
        dataset_id="dataset",
    )
    run_matrix = RunMatrixCell(
        name="baseline:exact-answer-single-turn-chat",
        dataset_family="nemo_gym",
        verifier_type="exact_answer",
        execution_type="single_turn_chat",
        adapter="nemo_gym_jsonl",
    )

    matrix = build_attempt_matrix([task.to_attempt()], run_matrix=run_matrix)

    assert matrix["run_matrix"]["name"] == "baseline:exact-answer-single-turn-chat"
    assert matrix["run_matrix"]["verifier_type"] == "exact_answer"
    assert matrix["run_matrix"]["execution_type"] == "single_turn_chat"


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
        answer_format="none",
        execution_type="single_turn_chat",
    )

    template = build_task_template(args)

    assert template["evaluator"]["strategy"] == "verifier_result_file"
    assert template["evaluator"]["config"]["path"] == "/polar/session/workspace/verifier_result.json"
    assert template["evaluator"]["config"]["reward_key"] == "reward"
    command = template["agent"]["custom_shell"]["command"]
    assert "responses_create_params" in command
    assert "verify_completion" in command
    assert "POLAR_ORIGINAL_TASK_SPEC_JSON" in command
    assert "load_verifier_task" in command


def test_task_template_strips_unsupported_top_k_from_model_request() -> None:
    args = SimpleNamespace(
        runtime_image="polar-spark-calculator:latest",
        task_timeout_seconds=240.0,
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        test_timeout_seconds=60.0,
        model_max_tokens=32,
        model_temperature=0.6,
        model_top_p=0.95,
        model_request_timeout_seconds=120.0,
        answer_format="none",
        execution_type="single_turn_chat",
    )

    command = build_task_template(args)["agent"]["custom_shell"]["command"]

    assert 'if key == "top_k" and value not in (None, -1):' in command


def test_build_task_template_supports_multi_turn_chat_tool_execution() -> None:
    args = SimpleNamespace(
        runtime_image="polar-spark-calculator:latest",
        task_timeout_seconds=240.0,
        model_name="Qwen/Qwen3-1.7B",
        test_timeout_seconds=60.0,
        model_max_tokens=64,
        model_temperature=0.6,
        model_top_p=0.95,
        model_request_timeout_seconds=120.0,
        answer_format="none",
        execution_type="multi_turn_chat_tool",
    )

    template = build_task_template(args)

    assert template["metadata"]["execution_type"] == "multi_turn_chat_tool"
    command = template["agent"]["custom_shell"]["command"]
    assert "multi_turn_transcript.json" in command
    assert "tool_observations" in command
    assert "verify_task(" in command
    assert "POLAR_GYM_RESOURCE_VERIFY_URL" in command
    assert "nemo_gym_response" in command


def test_apply_final_answer_format_appends_to_last_user_message() -> None:
    task = TaskSpec(
        task_id="task-1",
        responses_create_params={
            "input": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is 2+2?"},
            ]
        },
        prompt="What is 2+2?",
        answer="4",
        dataset_id="dataset",
    )

    [formatted] = apply_answer_format([task], "final_answer")

    messages = formatted.responses_create_params["input"]
    assert messages[0]["content"] == "Be concise."
    assert messages[1]["content"].endswith(FINAL_ANSWER_INSTRUCTION)
    assert formatted.prompt.endswith(FINAL_ANSWER_INSTRUCTION)
    assert formatted.metadata["answer_format"] == "final_answer"


def test_apply_final_answer_format_appends_to_string_input_once() -> None:
    task = TaskSpec(
        task_id="task-1",
        responses_create_params={"input": "What is 2+2?"},
        prompt="What is 2+2?",
        answer="4",
        dataset_id="dataset",
    )

    [formatted] = apply_answer_format([task], "final_answer")
    [formatted_again] = apply_answer_format([formatted], "final_answer")

    assert formatted.responses_create_params["input"].endswith(FINAL_ANSWER_INSTRUCTION)
    assert formatted_again.responses_create_params["input"].count(FINAL_ANSWER_INSTRUCTION) == 1


def test_summarize_task_provenance_carries_materials_hashes() -> None:
    task = TaskSpec(
        task_id="materials-1",
        responses_create_params={"input": "Predict."},
        prompt="Predict.",
        answer="{}",
        dataset_id="materials_replay",
        source_dataset="nist_ambench_in718_mds2_3735",
        metadata={
            "integrity_policy_id": "ambench_in718_posthoc_public_replay_v0",
            "dataset_hashes": {"readme.pdf": "abc123"},
        },
    )

    summary = _summarize_task_provenance([task])

    assert summary["source_datasets"] == ["nist_ambench_in718_mds2_3735"]
    assert summary["benchmark_integrity_policy_ids"] == [
        "ambench_in718_posthoc_public_replay_v0"
    ]
    assert summary["dataset_hashes"] == {"readme.pdf": "abc123"}
