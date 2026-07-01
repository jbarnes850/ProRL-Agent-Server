from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nemo_polar_bridge.datasets.base import TaskSpec
from nemo_polar_bridge.datasets.data_loader import NeMoGymDatasetAdapter
from nemo_polar_bridge.datasets.prepare_dataset import (
    FINAL_ANSWER_INSTRUCTION,
    apply_answer_format,
    build_attempt_matrix,
    build_task_template,
    _build_adapter,
    _build_config_dict,
    _summarize_task_provenance,
)
from nemo_polar_bridge.datasets.reasoning_gym_adapter import ReasoningGymDatasetAdapter
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


def test_final_answer_instruction_has_no_literal_placeholder() -> None:
    # A literal "<answer>" placeholder gets copied verbatim by small models,
    # producing "Final answer: <answer>" -> zero reward. The instruction must
    # direct the model to write its actual result after "Final answer:".
    assert "<answer>" not in FINAL_ANSWER_INSTRUCTION
    assert "final answer:" in FINAL_ANSWER_INSTRUCTION.lower()


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


def test_build_adapter_defaults_to_nemo_gym() -> None:
    args = SimpleNamespace(
        dataset_family="nemo_gym",
        dataset_id="nvidia/Nemotron-RL-ReasoningGym-v1",
        config="default",
        split="train",
        local_jsonl=None,
        source_dataset=[],
    )
    adapter = _build_adapter(args)
    assert isinstance(adapter, NeMoGymDatasetAdapter)


def test_build_adapter_selects_reasoning_gym() -> None:
    args = SimpleNamespace(
        dataset_family="reasoning_gym",
        dataset_id="ignored-for-reasoning-gym",
        config="default",
        split="train",
        local_jsonl=None,
        source_dataset=[],
        reasoning_gym_task_name="basic_arithmetic",
        reasoning_gym_seed=7,
    )
    adapter = _build_adapter(args)
    assert isinstance(adapter, ReasoningGymDatasetAdapter)
    assert adapter.task_name == "basic_arithmetic"
    assert adapter.seed == 7


def _config_dict_args(**overrides: object) -> SimpleNamespace:
    defaults = dict(
        dataset_id="nvidia/Nemotron-RL-ReasoningGym-v1",
        config="default",
        split="train",
        limit=2,
        scan_rows=200,
        local_jsonl=None,
        source_dataset=[],
        answer_format="none",
        image="local/nemo-rl-main-cu132:c236061b",
        model_name="Qwen/Qwen3-1.7B",
        model_path="/host-hf/hub/models--Qwen--Qwen3-1.7B",
        model_max_tokens=1024,
        model_temperature=1.0,
        model_top_p=1.0,
        vllm_precision="bfloat16",
        vllm_kv_cache_dtype="auto",
        vllm_gpu_memory_utilization=0.35,
        vllm_enforce_eager="true",
        vllm_max_num_seqs=None,
        vllm_max_num_batched_tokens=None,
        nemo_rl_ref="c236061b250e97638722292ab8a54d5eb47ae00f",
        polar_rollout_port=19080,
        polar_gateway_port=19100,
        vllm_http_port=31000,
        repo_host="/home/jarrodbarnes/ProRL-Agent-Server",
        script_path="scripts/smoke/run_nemo_polar_external_collector_spark_smoke.sh",
        worker_hostname="spark-cfd0",
        worker_ip="192.168.100.11",
        head_hostname="spark-f7e2",
        head_ip="192.168.100.10",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_build_config_dict_reports_the_adapters_real_dataset_id_not_the_cli_arg() -> None:
    # args.dataset_id is the NeMo Gym CLI default and is ignored by
    # ReasoningGymDatasetAdapter (see _build_adapter). The manifest must not
    # echo that ignored value back as though it were the real source -- it
    # must report what the adapter actually used, so `validation_summary.json`
    # stays trustworthy evidence rather than a misleading label.
    args = _config_dict_args(dataset_id="nvidia/Nemotron-RL-ReasoningGym-v1")
    adapter = ReasoningGymDatasetAdapter(task_name="basic_arithmetic", seed=0)
    run_matrix = RunMatrixCell(
        name="smoke:exact-answer-single-turn-chat",
        dataset_family="reasoning_gym",
        verifier_type="exact_answer",
        execution_type="single_turn_chat",
        adapter="nemo_gym_jsonl",
    )

    config = _build_config_dict(args, adapter, Path("/tmp/run"), run_matrix, [])

    assert config["dataset"]["id"] == "reasoning-gym/basic_arithmetic"
    assert config["dataset"]["id"] != args.dataset_id


def test_build_config_dict_reports_nemo_gym_dataset_id_unchanged() -> None:
    args = _config_dict_args(dataset_id="nvidia/Nemotron-RL-ReasoningGym-v1")
    adapter = NeMoGymDatasetAdapter(dataset_id=args.dataset_id)
    run_matrix = RunMatrixCell(
        name="smoke:exact-answer-single-turn-chat",
        dataset_family="nemo_gym",
        verifier_type="exact_answer",
        execution_type="single_turn_chat",
        adapter="nemo_gym_jsonl",
    )

    config = _build_config_dict(args, adapter, Path("/tmp/run"), run_matrix, [])

    assert config["dataset"]["id"] == "nvidia/Nemotron-RL-ReasoningGym-v1"


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
