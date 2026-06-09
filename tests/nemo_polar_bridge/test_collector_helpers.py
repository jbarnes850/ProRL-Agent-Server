from __future__ import annotations

from types import SimpleNamespace

from nemo_polar_bridge.collector import (
    _flatten_traces_for_nemo,
    env_flag,
    env_int,
    normalize_openai_base_url,
    normalize_attempt_matrix,
    prompts_per_target,
    reward_std,
    select_attempts_for_group,
    stable_bucket,
    target_weights_for_generation,
)


def test_env_flag_parses_enabled_values(monkeypatch) -> None:
    monkeypatch.setenv("NEMO_POLAR_EXAMPLE_FLAG", "true")
    assert env_flag("NEMO_POLAR_EXAMPLE_FLAG")
    monkeypatch.setenv("NEMO_POLAR_EXAMPLE_FLAG", "0")
    assert not env_flag("NEMO_POLAR_EXAMPLE_FLAG")


def test_env_int_uses_default_and_env_value(monkeypatch) -> None:
    assert env_int("NEMO_POLAR_GROUP_WORKERS", default=3) == 3
    monkeypatch.setenv("NEMO_POLAR_GROUP_WORKERS", "4")
    assert env_int("NEMO_POLAR_GROUP_WORKERS", default=3) == 4


def test_normalize_openai_base_url_strips_v1_suffix() -> None:
    assert normalize_openai_base_url("http://10.0.0.2:31000/v1") == "http://10.0.0.2:31000"
    assert normalize_openai_base_url("http://10.0.0.2:31000/v1/") == "http://10.0.0.2:31000"
    assert normalize_openai_base_url("http://10.0.0.2:31000") == "http://10.0.0.2:31000"


def test_reward_std_detects_non_degenerate_group() -> None:
    assert reward_std([1.0, 0.0]) == 0.5
    assert reward_std([1.0, 1.0]) == 0.0


def test_prompts_per_target_uses_grpo_num_prompts_per_step() -> None:
    assert prompts_per_target(SimpleNamespace(grpo={"num_prompts_per_step": 2})) == 2
    assert prompts_per_target(SimpleNamespace(grpo={"num_prompts_per_step": 0})) == 1


def test_target_weights_respect_training_horizon() -> None:
    assert target_weights_for_generation(
        generation_weight_version=0,
        initial_weight_version=0,
        max_age=1,
        max_num_steps=1,
    ) == [0]
    assert target_weights_for_generation(
        generation_weight_version=0,
        initial_weight_version=0,
        max_age=1,
        max_num_steps=2,
    ) == [0, 1]
    assert target_weights_for_generation(
        generation_weight_version=1,
        initial_weight_version=0,
        max_age=1,
        max_num_steps=2,
    ) == []


def test_stable_bucket_is_deterministic() -> None:
    assert stable_bucket("abc", 7) == stable_bucket("abc", 7)
    assert 0 <= stable_bucket("abc", 7) < 7


def test_attempt_matrix_group_cycle_repeats_one_task_per_group() -> None:
    attempts, mode = normalize_attempt_matrix(
        {
            "selection_mode": "group_cycle",
            "attempts": [{"name": "a"}, {"name": "b"}],
        }
    )

    assert mode == "group_cycle"
    assert select_attempts_for_group(
        attempts or [],
        selection_mode=mode,
        group_id=1,
        num_generations=3,
    ) == [{"name": "b"}, {"name": "b"}, {"name": "b"}]


def test_legacy_attempt_matrix_cycles_per_generation() -> None:
    attempts, mode = normalize_attempt_matrix([{"name": "pass"}, {"name": "fail"}])

    assert mode == "generation_cycle"
    assert select_attempts_for_group(
        attempts or [],
        selection_mode=mode,
        group_id=4,
        num_generations=3,
    ) == [{"name": "pass"}, {"name": "fail"}, {"name": "pass"}]


def test_flatten_traces_preserves_two_turn_assistant_tokens_and_masks() -> None:
    trace = _flatten_traces_for_nemo(
        [
            {
                "prompt_ids": [1, 2, 3],
                "response_ids": [10, 11, 99],
                "response_logprobs": [-0.1, -0.2, -0.3],
                "loss_mask": [1, 1, 1],
                "prompt_messages": [{"role": "user", "content": "Q1"}],
                "response_messages": [{"role": "assistant", "content": "A1"}],
                "reward": None,
                "finish_reason": "stop",
            },
            {
                "prompt_ids": [1, 2, 3, 10, 11, 99, 50, 51],
                "response_ids": [20, 21, 99],
                "response_logprobs": [-0.5, -0.6, -0.7],
                "loss_mask": [1, 1, 1],
                "prompt_messages": [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "tool", "content": "result"},
                ],
                "response_messages": [{"role": "assistant", "content": "A2"}],
                "reward": 1.0,
                "finish_reason": "stop",
            },
        ]
    )

    assert trace["prompt_ids"] == [1, 2, 3]
    assert trace["response_ids"] == [10, 11, 99, 50, 51, 20, 21, 99]
    assert trace["response_logprobs"] == [-0.1, -0.2, -0.3, 0.0, 0.0, -0.5, -0.6, -0.7]
    assert trace["loss_mask"] == [1, 1, 1, 0, 0, 1, 1, 1]
    assert trace["reward"] == 1.0
    assert trace["metadata"]["polar_trace_count"] == 2
