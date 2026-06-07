from __future__ import annotations

import json

import pytest

from nemo_bridge.adapter import NeMoRolloutSample
from nemo_bridge.rollout_actor import (
    build_default_shell_agent,
    build_nemo_payload_columns,
    message_log_to_openai_messages,
    select_one_sample_per_session,
)


def _sample(*, reward: float = 1.0, sample_mask: int = 1) -> NeMoRolloutSample:
    return NeMoRolloutSample(
        group_index=0,
        trajectory_index=0,
        trace_index=0,
        session_id="session-1",
        task_id="task-1",
        prompt_ids=[10, 11],
        response_ids=[20, 21],
        input_ids=[10, 11, 20, 21],
        sequence_length=4,
        prompt_length=2,
        response_length=2,
        generation_logprobs=[0.0, 0.0, -0.1, -0.2],
        token_mask=[0, 0, 1, 1],
        sample_mask=sample_mask,
        total_reward=reward,
        prompt_ids_for_adv=[10, 11],
        truncated=False,
        metadata={
            "trace_debug": {
                "prompt_messages": [{"role": "user", "content": "Pick a digit"}],
                "response_text": "4",
            }
        },
    )


def test_message_log_to_openai_messages_drops_tensor_fields() -> None:
    messages = message_log_to_openai_messages(
        [
            {"role": "system", "content": "short"},
            {"role": "user", "content": "return 4", "token_ids": object()},
        ]
    )

    assert messages == [
        {"role": "system", "content": "short"},
        {"role": "user", "content": "return 4"},
    ]


def test_default_shell_agent_carries_messages_and_sampling_env() -> None:
    agent = build_default_shell_agent(
        messages=[{"role": "user", "content": "return 4"}],
        model_name="Qwen/Qwen3-0.6B",
        max_tokens=8,
        temperature=0.7,
        top_p=0.9,
        request_timeout_seconds=30.0,
    )

    assert agent["harness"] == "shell"
    env = agent["env"]
    assert json.loads(env["POLAR_NEMO_MESSAGES_JSON"]) == [
        {"role": "user", "content": "return 4"}
    ]
    assert env["POLAR_MODEL_NAME"] == "Qwen/Qwen3-0.6B"
    assert env["POLAR_MAX_TOKENS"] == "8"
    assert agent["custom_shell"]["cwd"] == "/polar/session"
    assert "chat/completions" in agent["custom_shell"]["command"]


def test_select_one_sample_prefers_trainable_last_trace() -> None:
    masked = _sample(reward=0.0, sample_mask=0)
    trainable = _sample(reward=1.0, sample_mask=1)

    selected = select_one_sample_per_session([masked, trainable], trace_selection="last")

    assert selected is trainable


def test_build_nemo_payload_columns_matches_transfer_queue_shapes() -> None:
    torch = pytest.importorskip("torch")

    bulk, carry, tags = build_nemo_payload_columns(
        [_sample()],
        pad_token_id=0,
        pad_to_multiple=8,
    )

    assert bulk["input_ids"].shape == (1, 8)
    assert bulk["input_lengths"].tolist() == [4]
    assert bulk["generation_logprobs"].shape == (1, 8)
    assert bulk["token_mask"].tolist()[0][:4] == [0.0, 0.0, 1.0, 1.0]
    assert bulk["sample_mask"].dtype == torch.float32
    assert carry["loss_multiplier"].tolist() == [1.0]
    assert carry["response_token_lengths"].tolist() == [2]
    assert tags[0]["sample_mask"] == 1
