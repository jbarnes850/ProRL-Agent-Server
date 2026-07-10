from __future__ import annotations

import json
from types import SimpleNamespace
from urllib import error

import pytest

from nemo_polar_bridge.collector import (
    _PolarAsyncTrajectoryCollector,
    _flatten_traces_for_nemo,
    _post_control,
    _render_attempt_payload,
    _result_trace,
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


def test_polar_collector_accepts_latest_nemo_empty_distillation_args(monkeypatch) -> None:
    monkeypatch.setenv("NEMO_POLAR_ROLLOUT_URL", "http://rollout")
    monkeypatch.setenv("NEMO_POLAR_TASK_TEMPLATE_PATH", __file__)
    monkeypatch.setattr(
        "nemo_polar_bridge.collector._load_json",
        lambda _path: {"runtime": {"env": {}}, "metadata": {}},
    )
    collector = _PolarAsyncTrajectoryCollector(
        policy_generation=object(),
        tokenizer=object(),
        task_to_env={},
        master_config=object(),
        replay_buffer=object(),
        teacher_worker_groups={},
        alias_to_group_alias={},
        on_policy_distillation_cfg={},
    )

    assert collector.get_efficiency_metrics() == {}


def test_polar_collector_rejects_nonempty_distillation_metadata() -> None:
    with pytest.raises(NotImplementedError, match="distillation"):
        _PolarAsyncTrajectoryCollector(
            policy_generation=object(),
            tokenizer=object(),
            task_to_env={},
            master_config=object(),
            replay_buffer=object(),
            teacher_worker_groups={"teacher": object()},
        )


def test_resume_after_refit_invalidates_cache_before_gateway_resume(monkeypatch) -> None:
    events = []

    class PolicyGeneration:
        def invalidate_kv_cache(self):
            events.append("invalidate")
            return True

    collector = object.__new__(_PolarAsyncTrajectoryCollector)
    collector.policy_generation = PolicyGeneration()
    collector.master_config = SimpleNamespace(
        grpo={
            "async_grpo": {
                "in_flight_weight_updates": True,
                "recompute_kv_cache_after_weight_updates": True,
            }
        }
    )
    collector._gateway_url = "http://gateway"
    collector._refit_pause = __import__("threading").Event()
    monkeypatch.setattr(
        "nemo_polar_bridge.collector._post_control",
        lambda url, timeout: events.append((url, timeout)),
    )

    collector.resume_after_refit()

    assert events == ["invalidate", ("http://gateway/admin/inference/resume", 30)]
    assert collector._refit_pause.is_set()


def test_resume_after_refit_keeps_gateway_paused_if_cache_invalidation_fails(
    monkeypatch,
) -> None:
    class PolicyGeneration:
        def invalidate_kv_cache(self):
            return False

    collector = object.__new__(_PolarAsyncTrajectoryCollector)
    collector.policy_generation = PolicyGeneration()
    collector.master_config = SimpleNamespace(
        grpo={
            "async_grpo": {
                "in_flight_weight_updates": True,
                "recompute_kv_cache_after_weight_updates": True,
            }
        }
    )
    collector._gateway_url = "http://gateway"
    collector._refit_pause = __import__("threading").Event()
    monkeypatch.setattr(
        "nemo_polar_bridge.collector._post_control",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("gateway must remain paused")
        ),
    )

    with pytest.raises(RuntimeError, match="not successful on all workers"):
        collector.resume_after_refit()

    assert not collector._refit_pause.is_set()


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


def test_render_attempt_payload_exports_original_task_spec_env() -> None:
    task_spec = {
        "task_id": "materials-1",
        "responses_create_params": {"input": "Predict."},
        "answer": '{"answer_values":{}}',
        "verifier_name": "materials_tensile_numeric",
    }
    template = {"runtime": {"env": {}}, "metadata": {}, "task_id": "{task_id}"}

    payload = _render_attempt_payload(
        template,
        context={"task_id": "group-task"},
        num_samples=1,
        attempt={"name": "attempt", "task_spec_json": json.dumps(task_spec)},
    )

    env = payload["runtime"]["env"]
    assert json.loads(env["POLAR_ORIGINAL_TASK_SPEC_JSON"]) == task_spec
    assert payload["metadata"]["attempt_task_spec_json"] == json.dumps(task_spec)


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


def test_flatten_traces_preserves_sampling_support() -> None:
    trace = _flatten_traces_for_nemo(
        [
            {
                "prompt_ids": [1],
                "response_ids": [10],
                "response_logprobs": [0.0],
                "sampling_support_token_ids": [[10]],
                "sampling_support_mass": [1.0],
                "loss_mask": [1],
            },
            {
                "prompt_ids": [1, 10, 50],
                "response_ids": [20],
                "response_logprobs": [-0.2],
                "sampling_support_token_ids": [[20, 21]],
                "sampling_support_mass": [1.0],
                "loss_mask": [1],
            },
        ],
        outcome_reward=1.0,
    )

    assert trace["response_ids"] == [10, 50, 20]
    assert trace["sampling_support_token_ids"] == [[10], [], [20, 21]]
    assert trace["sampling_support_mass"] == [1.0, 0.0, 1.0]


def test_result_trace_requires_complete_sampling_support(monkeypatch) -> None:
    monkeypatch.setenv("NEMO_POLAR_REQUIRE_SAMPLING_SUPPORT_REPLAY", "1")
    result = {
        "status": "COMPLETED",
        "trajectory": {
            "traces": [
                {
                    "prompt_ids": [1],
                    "response_ids": [10],
                    "response_logprobs": [0.0],
                    "loss_mask": [1],
                    "reward": 1.0,
                }
            ]
        },
    }

    _, valid = _result_trace(result, tokenizer=None)

    assert valid is False


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_post_control_retries_transient_failures_then_succeeds(monkeypatch) -> None:
    # A momentary gateway blip during the pause/resume-around-refit call
    # (prepare_for_refit/resume_after_refit) previously aborted the training
    # step on the first URLError. It must retry before giving up.
    calls = []
    sleeps = []

    def fake_urlopen(req, timeout):
        calls.append(timeout)
        if len(calls) < 3:
            raise error.URLError("connection refused")
        return _FakeResponse(b'{"paused": true}')

    monkeypatch.setattr("nemo_polar_bridge.collector.request.urlopen", fake_urlopen)
    monkeypatch.setattr("nemo_polar_bridge.collector.time.sleep", lambda s: sleeps.append(s))

    result = _post_control("http://gateway/admin/inference/pause", timeout=30, attempts=3)

    assert result == {"paused": True}
    assert len(calls) == 3
    assert len(sleeps) == 2


def test_post_control_raises_after_exhausting_retries(monkeypatch) -> None:
    def fake_urlopen(req, timeout):
        raise error.URLError("connection refused")

    monkeypatch.setattr("nemo_polar_bridge.collector.request.urlopen", fake_urlopen)
    monkeypatch.setattr("nemo_polar_bridge.collector.time.sleep", lambda s: None)

    with pytest.raises(error.URLError):
        _post_control("http://gateway/admin/inference/pause", timeout=30, attempts=3)


def test_post_control_succeeds_immediately_without_retry(monkeypatch) -> None:
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(timeout)
        return _FakeResponse(b'{"resumed": true}')

    monkeypatch.setattr("nemo_polar_bridge.collector.request.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "nemo_polar_bridge.collector.time.sleep",
        lambda s: (_ for _ in ()).throw(AssertionError("should not sleep on first success")),
    )

    result = _post_control("http://gateway/admin/inference/resume", timeout=30)

    assert result == {"resumed": True}
    assert len(calls) == 1
