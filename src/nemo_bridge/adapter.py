"""Convert Polar rollout results into NeMo RL external-rollout rows.

The adapter does not import NeMo.  It emits the minimal data contract that a
NeMo ``SyncRolloutActor`` replacement must write to the data plane for GRPO:
full token sequences, full-length rollout logprobs, response-token masks,
sample masks, rewards, and prompt ids for per-prompt advantage grouping.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import math
import statistics
from typing import Any, TYPE_CHECKING

from slime_bridge._messages import messages_to_text

if TYPE_CHECKING:
    from polar.rollout.models import SessionResult, TaskResult
    from polar.trajectory.models import Trace

logger = logging.getLogger(__name__)


class NeMoRolloutContractError(ValueError):
    """Raised when a Polar rollout cannot safely become a NeMo training row."""


@dataclass(slots=True)
class NeMoRolloutSample:
    """One unpadded external-rollout sample for NeMo RL GRPO."""

    group_index: int
    trajectory_index: int
    trace_index: int
    session_id: str
    task_id: str
    prompt_ids: list[int]
    response_ids: list[int]
    input_ids: list[int]
    sequence_length: int
    prompt_length: int
    response_length: int
    generation_logprobs: list[float]
    token_mask: list[int]
    sample_mask: int
    total_reward: float
    prompt_ids_for_adv: list[int]
    truncated: bool
    metadata: dict[str, Any]

    def as_training_row(self) -> dict[str, Any]:
        """Return row-shaped values using NeMo's field names."""
        return {
            "input_ids": list(self.input_ids),
            "input_lengths": int(self.sequence_length),
            "generation_logprobs": list(self.generation_logprobs),
            "token_mask": list(self.token_mask),
            "sample_mask": int(self.sample_mask),
            "total_reward": float(self.total_reward),
            "prompt_ids_for_adv": list(self.prompt_ids_for_adv),
            "loss_multiplier": int(self.sample_mask),
            "truncated": bool(self.truncated),
            "length": int(self.sequence_length),
            "response_token_lengths": int(self.response_length),
            "metadata": deepcopy(self.metadata),
        }


@dataclass(slots=True)
class NeMoRolloutBatch:
    """A flat collection of Polar rollouts ready for a NeMo data-plane writer."""

    samples: list[NeMoRolloutSample]

    def as_training_rows(self) -> list[dict[str, Any]]:
        return [sample.as_training_row() for sample in self.samples]

    def validate(
        self,
        *,
        max_policy_staleness: int | None = None,
        require_reward_variance: bool = False,
    ) -> None:
        validate_rollout_rows(self.samples, max_policy_staleness=max_policy_staleness)
        if require_reward_variance:
            validate_reward_variance(self.samples)


@dataclass(frozen=True, slots=True)
class RewardGroupStats:
    """Reward statistics for one NeMo GRPO advantage group."""

    group_index: int
    prompt_ids_for_adv: tuple[int, ...]
    rewards: tuple[float, ...]

    @property
    def count(self) -> int:
        return len(self.rewards)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.rewards) if self.rewards else 0.0

    @property
    def std(self) -> float:
        if len(self.rewards) < 2:
            return 0.0
        return statistics.pstdev(self.rewards)


def task_result_to_nemo_batch(
    task_result: "TaskResult",
    source_group: list[Any] | None = None,
    *,
    group_index: int | None = None,
    reward_key: str = "score",
    max_tokens: int | None = None,
    accepted_rollout_id: int | None = None,
    scheduler_group_id: int | None = None,
) -> NeMoRolloutBatch:
    """Convert a Polar task result into flat NeMo external-rollout samples.

    ``source_group`` mirrors the Slime bridge convention: if provided, each
    source sample's ``index`` becomes the Polar trajectory index and the first
    source sample's ``group_index`` becomes the advantage group id.
    """
    resolved_group_index = _resolve_group_index(source_group, group_index)
    samples: list[NeMoRolloutSample] = []
    for position, result in enumerate(task_result.results):
        source = source_group[position] if source_group and position < len(source_group) else None
        trajectory_index = int(getattr(source, "index", position) if source is not None else position)
        samples.extend(
            session_result_to_nemo_samples(
                result,
                resolved_group_index,
                trajectory_index=trajectory_index,
                reward_key=reward_key,
                max_tokens=max_tokens,
                accepted_rollout_id=accepted_rollout_id,
                scheduler_group_id=scheduler_group_id,
            )
        )
    return NeMoRolloutBatch(samples=samples)


def session_result_to_nemo_samples(
    result: "SessionResult",
    group_index: int,
    *,
    trajectory_index: int,
    reward_key: str = "score",
    max_tokens: int | None = None,
    accepted_rollout_id: int | None = None,
    scheduler_group_id: int | None = None,
) -> list[NeMoRolloutSample]:
    """Convert one Polar session result into one NeMo row per trace."""
    del reward_key  # Evaluators already wrote the scalar reward onto each trace.
    samples: list[NeMoRolloutSample] = []
    for trace_index, trace in enumerate(result.trajectory.traces):
        sample = _build_sample(
            result=result,
            trace=trace,
            trace_index=trace_index,
            group_index=group_index,
            trajectory_index=trajectory_index,
            max_tokens=max_tokens,
            accepted_rollout_id=accepted_rollout_id,
            scheduler_group_id=scheduler_group_id,
        )
        if sample is not None:
            samples.append(sample)

    if samples:
        return samples

    logger.warning(
        "Session %s: no usable trace (traces=%d, max_tokens=%s); emitting masked placeholder",
        result.session_id,
        len(result.trajectory.traces),
        max_tokens,
    )
    return [
        _build_placeholder(
            result=result,
            group_index=group_index,
            trajectory_index=trajectory_index,
            accepted_rollout_id=accepted_rollout_id,
            scheduler_group_id=scheduler_group_id,
        )
    ]


def validate_rollout_rows(
    samples: list[NeMoRolloutSample],
    *,
    max_policy_staleness: int | None = None,
) -> None:
    """Validate field alignment and optional train/inference staleness."""
    for sample in samples:
        if len(sample.input_ids) != sample.sequence_length:
            raise NeMoRolloutContractError(
                f"Session {sample.session_id} trace {sample.trace_index}: "
                "input_ids length does not match sequence_length"
            )
        _require_len(sample, "generation_logprobs", sample.generation_logprobs)
        _require_len(sample, "token_mask", sample.token_mask)
        if sample.sequence_length != sample.prompt_length + sample.response_length:
            raise NeMoRolloutContractError(
                f"Session {sample.session_id} trace {sample.trace_index}: "
                "sequence length must equal prompt_length + response_length"
            )
        if sample.input_ids != sample.prompt_ids + sample.response_ids:
            raise NeMoRolloutContractError(
                f"Session {sample.session_id} trace {sample.trace_index}: "
                "input_ids must preserve prompt_ids + response_ids"
            )
        if sample.prompt_ids_for_adv != sample.prompt_ids:
            raise NeMoRolloutContractError(
                f"Session {sample.session_id} trace {sample.trace_index}: "
                "prompt_ids_for_adv must preserve the original prompt tokens"
            )
        if sample.sample_mask not in (0, 1):
            raise NeMoRolloutContractError("sample_mask must be 0 or 1")
    validate_policy_staleness(samples, max_policy_staleness=max_policy_staleness)


def validate_policy_staleness(
    samples: list[NeMoRolloutSample],
    *,
    max_policy_staleness: int | None,
) -> None:
    """Reject trainable samples whose rollout policy is too stale."""
    if max_policy_staleness is None:
        return
    for sample in samples:
        if sample.sample_mask == 0:
            continue
        staleness = _policy_staleness(sample.metadata)
        if staleness is None:
            raise NeMoRolloutContractError(
                f"Session {sample.session_id} trace {sample.trace_index}: "
                "missing policy_staleness or accepted_rollout_id/policy_version metadata"
            )
        if staleness > max_policy_staleness:
            raise NeMoRolloutContractError(
                f"Session {sample.session_id} trace {sample.trace_index}: "
                f"policy staleness {staleness} exceeds max_policy_staleness="
                f"{max_policy_staleness}"
            )


def reward_group_stats(samples: list[NeMoRolloutSample]) -> list[RewardGroupStats]:
    """Compute reward stats using NeMo's per-prompt GRPO grouping key."""
    grouped: dict[tuple[int, tuple[int, ...]], list[float]] = {}
    for sample in samples:
        if sample.sample_mask == 0:
            continue
        key = (sample.group_index, tuple(sample.prompt_ids_for_adv))
        grouped.setdefault(key, []).append(float(sample.total_reward))
    return [
        RewardGroupStats(group_index=group, prompt_ids_for_adv=prompt_ids, rewards=tuple(rewards))
        for (group, prompt_ids), rewards in grouped.items()
    ]


def validate_reward_variance(
    samples: list[NeMoRolloutSample],
    *,
    near_zero_threshold: float = 1e-5,
    max_near_zero_fraction: float = 0.5,
    min_group_size: int = 2,
) -> list[RewardGroupStats]:
    """Apply the GRPO reward-variance gate before normalization."""
    stats = reward_group_stats(samples)
    eligible = [stat for stat in stats if stat.count >= min_group_size]
    if not eligible:
        raise NeMoRolloutContractError(
            f"No reward groups have at least {min_group_size} trainable samples"
        )
    near_zero = [stat for stat in eligible if stat.std < near_zero_threshold]
    fraction = len(near_zero) / len(eligible)
    if fraction > max_near_zero_fraction:
        raise NeMoRolloutContractError(
            f"Near-zero reward std in {len(near_zero)}/{len(eligible)} groups "
            f"({fraction:.3f}) exceeds max_near_zero_fraction={max_near_zero_fraction}"
        )
    return stats


def _build_sample(
    *,
    result: "SessionResult",
    trace: "Trace",
    trace_index: int,
    group_index: int,
    trajectory_index: int,
    max_tokens: int | None,
    accepted_rollout_id: int | None,
    scheduler_group_id: int | None,
) -> NeMoRolloutSample | None:
    prompt_ids = list(trace.prompt_ids)
    response_ids = list(trace.response_ids)
    if not prompt_ids or not response_ids:
        logger.warning(
            "Dropping trace %d from session %s: missing tokens (prompt=%d, response=%d)",
            trace_index,
            result.session_id,
            len(prompt_ids),
            len(response_ids),
        )
        return None

    sequence_length = len(prompt_ids) + len(response_ids)
    if max_tokens is not None and sequence_length > max_tokens:
        logger.warning(
            "Dropping trace %d from session %s: sequence_length=%d > max_tokens=%d",
            trace_index,
            result.session_id,
            sequence_length,
            max_tokens,
        )
        return None

    failed = _is_failed_result(result)
    require_trainable = not failed
    loss_mask = _loss_mask_from_trace(
        trace,
        response_len=len(response_ids),
        require_loss_mask=require_trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )
    if failed:
        loss_mask = [0] * len(response_ids)
    response_logprobs = _extract_rollout_logprobs(
        trace,
        response_len=len(response_ids),
        loss_mask=loss_mask,
        require_trainable_logprobs=require_trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )
    reward_value = _reward_value(trace)
    sample_mask = 0 if failed else 1

    metadata = _metadata(
        result=result,
        trace=trace,
        trace_index=trace_index,
        accepted_rollout_id=accepted_rollout_id,
        scheduler_group_id=scheduler_group_id,
    )
    metadata["raw_reward"] = reward_value
    metadata["nemo_contract"] = {
        "input_lengths_semantics": "full_unpadded_sequence_length",
        "token_mask_semantics": "response_token_loss_mask_with_prompt_zeros",
    }

    return NeMoRolloutSample(
        group_index=group_index,
        trajectory_index=trajectory_index,
        trace_index=trace_index,
        session_id=result.session_id,
        task_id=result.task_id,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=prompt_ids + response_ids,
        sequence_length=sequence_length,
        prompt_length=len(prompt_ids),
        response_length=len(response_ids),
        generation_logprobs=[0.0] * len(prompt_ids) + response_logprobs,
        token_mask=[0] * len(prompt_ids) + loss_mask,
        sample_mask=sample_mask,
        total_reward=reward_value if sample_mask else 0.0,
        prompt_ids_for_adv=prompt_ids,
        truncated=trace.finish_reason == "length",
        metadata=metadata,
    )


def _build_placeholder(
    *,
    result: "SessionResult",
    group_index: int,
    trajectory_index: int,
    accepted_rollout_id: int | None,
    scheduler_group_id: int | None,
) -> NeMoRolloutSample:
    metadata = _metadata(
        result=result,
        trace=None,
        trace_index=-1,
        accepted_rollout_id=accepted_rollout_id,
        scheduler_group_id=scheduler_group_id,
    )
    metadata["placeholder"] = True
    metadata["nemo_contract"] = {
        "input_lengths_semantics": "full_unpadded_sequence_length",
        "token_mask_semantics": "fully_masked_placeholder",
    }
    return NeMoRolloutSample(
        group_index=group_index,
        trajectory_index=trajectory_index,
        trace_index=-1,
        session_id=result.session_id,
        task_id=result.task_id,
        prompt_ids=[0],
        response_ids=[0],
        input_ids=[0, 0],
        sequence_length=2,
        prompt_length=1,
        response_length=1,
        generation_logprobs=[0.0, 0.0],
        token_mask=[0, 0],
        sample_mask=0,
        total_reward=0.0,
        prompt_ids_for_adv=[0],
        truncated=False,
        metadata=metadata,
    )


def _metadata(
    *,
    result: "SessionResult",
    trace: "Trace | None",
    trace_index: int,
    accepted_rollout_id: int | None,
    scheduler_group_id: int | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_metadata": deepcopy(getattr(result, "metadata", {}) or {}),
        "result_error": result.error,
        "session_id": result.session_id,
        "session_status": _status_value(result.status),
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": trace_index,
        "trace_metadata": deepcopy(getattr(trace, "metadata", {}) or {}) if trace is not None else {},
        "trajectory_error": result.trajectory.error,
        "trajectory_metadata": deepcopy(result.trajectory.metadata),
        "trajectory_status": result.trajectory.status,
        "trace_debug": {
            "finish_reason": trace.finish_reason if trace is not None else None,
            "prompt_messages": deepcopy(trace.prompt_messages) if trace is not None else [],
            "response_messages": deepcopy(trace.response_messages) if trace is not None else [],
            "response_text": messages_to_text(trace.response_messages) if trace is not None else "",
        },
    }
    metadata.update(_scheduler_metadata(result, trace))
    if accepted_rollout_id is not None:
        metadata["accepted_rollout_id"] = int(accepted_rollout_id)
    if scheduler_group_id is not None:
        metadata["scheduler_group_id"] = int(scheduler_group_id)

    staleness = _policy_staleness(metadata)
    if staleness is not None:
        metadata["policy_staleness"] = int(staleness)
    return metadata


def _scheduler_metadata(result: "SessionResult", trace: "Trace | None") -> dict[str, Any]:
    keys = {
        "accepted_rollout_id",
        "group_id",
        "policy_staleness",
        "policy_version",
        "rollout_step",
        "scheduler_group_id",
    }
    merged: dict[str, Any] = {}
    for source in (
        getattr(result, "metadata", None),
        getattr(result.trajectory, "metadata", None),
        getattr(trace, "metadata", None) if trace is not None else None,
    ):
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source:
                merged[key] = source[key]
    return merged


def _extract_rollout_logprobs(
    trace: "Trace",
    *,
    response_len: int,
    loss_mask: list[int],
    require_trainable_logprobs: bool,
    session_id: str,
    trace_index: int,
) -> list[float]:
    logprobs = trace.response_logprobs
    if not logprobs:
        if require_trainable_logprobs and any(loss_mask):
            raise NeMoRolloutContractError(
                f"Session {session_id} trace {trace_index}: missing rollout logprobs "
                "for trainable response tokens"
            )
        return [0.0] * response_len
    if len(logprobs) != response_len:
        raise NeMoRolloutContractError(
            f"Session {session_id} trace {trace_index}: rollout logprobs length "
            f"{len(logprobs)} != response length {response_len}"
        )
    return [float(value) for value in logprobs]


def _loss_mask_from_trace(
    trace: "Trace",
    *,
    response_len: int,
    require_loss_mask: bool,
    session_id: str,
    trace_index: int,
) -> list[int]:
    mask = list(trace.loss_mask)
    if not mask:
        if require_loss_mask:
            raise NeMoRolloutContractError(
                f"Session {session_id} trace {trace_index}: missing loss_mask"
            )
        return [0] * response_len
    if len(mask) != response_len:
        raise NeMoRolloutContractError(
            f"Session {session_id} trace {trace_index}: loss_mask length "
            f"{len(mask)} != response length {response_len}"
        )
    return [1 if int(value) else 0 for value in mask]


def _reward_value(trace: "Trace") -> float:
    return float(trace.reward) if trace.reward is not None else 0.0


def _is_failed_result(result: "SessionResult") -> bool:
    return (
        _status_value(result.status).upper() in {"ERROR", "TIMEOUT"}
        or result.trajectory.status in {"ERROR", "TIMEOUT"}
        or bool(result.error)
        or bool(result.trajectory.error)
    )


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status))


def _resolve_group_index(source_group: list[Any] | None, group_index: int | None) -> int:
    if group_index is not None:
        return int(group_index)
    if source_group:
        value = getattr(source_group[0], "group_index", 0)
        if value is not None:
            return int(value)
    return 0


def _policy_staleness(metadata: dict[str, Any]) -> int | None:
    existing = _optional_int(metadata.get("policy_staleness"))
    if existing is not None:
        return max(0, existing)

    policy_version = _optional_int(metadata.get("policy_version"))
    if policy_version is None:
        return None
    accepted_rollout_id = _optional_int(metadata.get("accepted_rollout_id"))
    if accepted_rollout_id is not None:
        return max(0, accepted_rollout_id - policy_version)
    rollout_step = _optional_int(metadata.get("rollout_step"))
    if rollout_step is not None:
        return max(0, rollout_step - policy_version)
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(float(parsed)):
        return None
    return parsed


def _require_len(sample: NeMoRolloutSample, field: str, value: list[Any]) -> None:
    if len(value) != sample.sequence_length:
        raise NeMoRolloutContractError(
            f"Session {sample.session_id} trace {sample.trace_index}: {field} length "
            f"{len(value)} != sequence length {sample.sequence_length}"
        )
