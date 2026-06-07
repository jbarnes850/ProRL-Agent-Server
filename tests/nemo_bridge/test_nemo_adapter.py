from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_bridge.adapter import (
    NeMoRolloutContractError,
    reward_group_stats,
    session_result_to_nemo_samples,
    task_result_to_nemo_batch,
    validate_policy_staleness,
    validate_reward_variance,
)
from polar.rollout.models import SessionResult, SessionStatus, SessionTiming, TaskResult
from polar.trajectory.models import Trace, Trajectory


def _session_result(
    *,
    trace: Trace,
    status: SessionStatus = SessionStatus.COMPLETED,
    session_id: str = "session-1",
    policy_version: int = 5,
    rollout_step: int = 7,
) -> SessionResult:
    trajectory_status = "COMPLETED" if status == SessionStatus.COMPLETED else status.value
    return SessionResult(
        session_id=session_id,
        task_id="task-1",
        status=status,
        node_id="node-a",
        timing=SessionTiming(init_ms=1.0, run_ms=2.0, postrun_ms=3.0),
        metadata={"policy_version": policy_version},
        trajectory=Trajectory(
            status=trajectory_status,
            metadata={"rollout_step": rollout_step},
            traces=[trace],
        ),
    )


def _trace(
    *,
    reward: float = 1.0,
    prompt_ids: list[int] | None = None,
    response_ids: list[int] | None = None,
) -> Trace:
    return Trace(
        prompt_ids=prompt_ids or [10, 11],
        response_ids=response_ids or [20, 21, 22],
        loss_mask=[1, 0, 1],
        prompt_messages=[{"role": "user", "content": "Say hi"}],
        response_messages=[{"role": "assistant", "content": "Hi"}],
        response_logprobs=[-0.1, -0.2, -0.3],
        reward=reward,
        metadata={"group_id": "group-a"},
    )


def test_session_result_to_nemo_samples_preserves_tokens_logprobs_masks_and_policy_metadata() -> None:
    samples = session_result_to_nemo_samples(
        _session_result(trace=_trace()),
        group_index=11,
        trajectory_index=2,
        accepted_rollout_id=9,
        scheduler_group_id=13,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.group_index == 11
    assert sample.trajectory_index == 2
    assert sample.trace_index == 0
    assert sample.prompt_ids == [10, 11]
    assert sample.response_ids == [20, 21, 22]
    assert sample.input_ids == [10, 11, 20, 21, 22]
    assert sample.sequence_length == 5
    assert sample.prompt_length == 2
    assert sample.response_length == 3
    assert sample.generation_logprobs == [0.0, 0.0, -0.1, -0.2, -0.3]
    assert sample.token_mask == [0, 0, 1, 0, 1]
    assert sample.sample_mask == 1
    assert sample.total_reward == 1.0
    assert sample.prompt_ids_for_adv == [10, 11]
    assert sample.metadata["policy_version"] == 5
    assert sample.metadata["rollout_step"] == 7
    assert sample.metadata["accepted_rollout_id"] == 9
    assert sample.metadata["policy_staleness"] == 4
    assert sample.metadata["scheduler_group_id"] == 13

    row = sample.as_training_row()
    assert row["input_lengths"] == 5
    assert row["loss_multiplier"] == 1
    assert row["prompt_ids_for_adv"] == [10, 11]


def test_session_result_to_nemo_samples_requires_logprobs_for_trainable_tokens() -> None:
    trace = Trace(
        prompt_ids=[1],
        response_ids=[2],
        loss_mask=[1],
        response_logprobs=None,
    )

    with pytest.raises(NeMoRolloutContractError, match="missing rollout logprobs"):
        session_result_to_nemo_samples(
            _session_result(trace=trace),
            group_index=1,
            trajectory_index=2,
        )


def test_task_result_to_nemo_batch_preserves_reward_grouping_from_source_group() -> None:
    first = _session_result(trace=_trace(reward=1.0), session_id="session-1")
    second = _session_result(trace=_trace(reward=0.0), session_id="session-2")
    task_result = TaskResult(task_id="task-1", status="completed", results=[first, second])
    source_group = [
        SimpleNamespace(group_index=42, index=4),
        SimpleNamespace(group_index=42, index=5),
    ]

    batch = task_result_to_nemo_batch(
        task_result,
        source_group,
        accepted_rollout_id=7,
        scheduler_group_id=42,
    )

    assert [sample.group_index for sample in batch.samples] == [42, 42]
    assert [sample.trajectory_index for sample in batch.samples] == [4, 5]
    assert [sample.prompt_ids_for_adv for sample in batch.samples] == [[10, 11], [10, 11]]
    stats = reward_group_stats(batch.samples)
    assert len(stats) == 1
    assert stats[0].group_index == 42
    assert stats[0].count == 2
    assert stats[0].std > 0.0


def test_failed_session_preserves_tokens_but_masks_training_and_reward() -> None:
    samples = session_result_to_nemo_samples(
        _session_result(trace=_trace(reward=3.0), status=SessionStatus.TIMEOUT),
        group_index=1,
        trajectory_index=2,
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.input_ids == [10, 11, 20, 21, 22]
    assert sample.generation_logprobs == [0.0, 0.0, -0.1, -0.2, -0.3]
    assert sample.token_mask == [0, 0, 0, 0, 0]
    assert sample.sample_mask == 0
    assert sample.total_reward == 0.0
    assert sample.metadata["raw_reward"] == 3.0


def test_validate_policy_staleness_rejects_stale_trainable_rollouts() -> None:
    samples = session_result_to_nemo_samples(
        _session_result(trace=_trace(), policy_version=2, rollout_step=2),
        group_index=1,
        trajectory_index=2,
        accepted_rollout_id=5,
    )

    with pytest.raises(NeMoRolloutContractError, match="policy staleness 3 exceeds"):
        validate_policy_staleness(samples, max_policy_staleness=2)


def test_reward_variance_gate_rejects_near_zero_grpo_groups() -> None:
    first = _session_result(trace=_trace(reward=1.0), session_id="session-1")
    second = _session_result(trace=_trace(reward=1.0), session_id="session-2")
    task_result = TaskResult(task_id="task-1", status="completed", results=[first, second])
    batch = task_result_to_nemo_batch(task_result, group_index=1)

    with pytest.raises(NeMoRolloutContractError, match="Near-zero reward std"):
        validate_reward_variance(batch.samples)


def test_placeholder_emitted_when_no_trace_tokens_are_usable() -> None:
    trace = Trace(prompt_ids=[], response_ids=[], loss_mask=[])
    samples = session_result_to_nemo_samples(
        _session_result(trace=trace),
        group_index=1,
        trajectory_index=2,
    )

    assert len(samples) == 1
    assert samples[0].metadata["placeholder"] is True
    assert samples[0].sample_mask == 0
    assert samples[0].token_mask == [0, 0]
