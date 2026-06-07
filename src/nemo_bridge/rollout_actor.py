"""Polar-backed NeMo RL rollout actor.

This module gives NeMo's synchronous TransferQueue GRPO trainer an alternate
rollout actor with the same public shape as
``nemo_rl.experience.sync_rollout_actor.SyncRolloutActor``.  NeMo still owns
policy training, logprob recomputation, TransferQueue fanout, and vLLM weight
sync; Polar owns external agent/session rollout and trace construction.
"""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import statistics
import time
import uuid
from typing import Any, Optional

import httpx

from nemo_bridge.adapter import (
    NeMoRolloutContractError,
    NeMoRolloutSample,
    validate_reward_variance,
    validate_rollout_rows,
)

logger = logging.getLogger(__name__)

_DEFAULT_SHELL_COMMAND = r"""python3 - <<'PY'
import json
import os
import urllib.request

base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")
api_key = os.environ.get("OPENAI_API_KEY", "")
payload = {
    "model": os.environ.get("POLAR_MODEL_NAME") or "model",
    "messages": json.loads(os.environ["POLAR_NEMO_MESSAGES_JSON"]),
    "max_tokens": int(os.environ.get("POLAR_MAX_TOKENS", "16")),
    "temperature": float(os.environ.get("POLAR_TEMPERATURE", "1.0")),
    "top_p": float(os.environ.get("POLAR_TOP_P", "1.0")),
}
stop = os.environ.get("POLAR_STOP")
if stop:
    payload["stop"] = json.loads(stop)

request = urllib.request.Request(
    base_url + "/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "content-type": "application/json",
        "authorization": "Bearer " + api_key,
    },
)
with urllib.request.urlopen(
    request,
    timeout=float(os.environ.get("POLAR_REQUEST_TIMEOUT", "120")),
) as response:
    data = json.load(response)

message = (data["choices"][0].get("message") or {})
print(message.get("content") or "")
PY"""


def build_default_shell_agent(
    *,
    messages: list[dict[str, str]],
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    request_timeout_seconds: float,
    base_agent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a shell-agent payload that makes one OpenAI chat completion."""
    agent = deepcopy(base_agent or {})
    agent.setdefault("harness", "shell")
    custom_shell = dict(agent.get("custom_shell") or {})
    custom_shell.setdefault("command", _DEFAULT_SHELL_COMMAND)
    custom_shell.setdefault("cwd", "/polar/session")
    agent["custom_shell"] = custom_shell
    env = dict(agent.get("env") or {})
    env.update(
        {
            "POLAR_NEMO_MESSAGES_JSON": json.dumps(messages, ensure_ascii=False),
            "POLAR_MODEL_NAME": str(model_name),
            "POLAR_MAX_TOKENS": str(int(max_tokens)),
            "POLAR_TEMPERATURE": str(float(temperature)),
            "POLAR_TOP_P": str(float(top_p)),
            "POLAR_REQUEST_TIMEOUT": str(float(request_timeout_seconds)),
        }
    )
    agent["env"] = env
    agent.setdefault("model_name", str(model_name))
    return agent


def message_log_to_openai_messages(message_log: Any) -> list[dict[str, str]]:
    """Extract role/content messages from a NeMo message_log row."""
    messages: list[dict[str, str]] = []
    for raw in list(message_log or []):
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "user")
        content = _string_content(raw.get("content"))
        if content:
            messages.append({"role": role, "content": content})
    return messages or [{"role": "user", "content": ""}]


def select_one_sample_per_session(
    samples: list[NeMoRolloutSample],
    *,
    trace_selection: str,
) -> NeMoRolloutSample:
    """Select exactly one NeMo training row from one Polar session."""
    if not samples:
        raise NeMoRolloutContractError("No samples available for session selection")
    trainable = [sample for sample in samples if sample.sample_mask == 1]
    pool = trainable or samples
    if trace_selection == "first":
        return pool[0]
    if trace_selection == "last":
        return pool[-1]
    raise ValueError("trace_selection must be 'first' or 'last'")


def build_nemo_payload_columns(
    samples: list[NeMoRolloutSample],
    *,
    pad_token_id: int,
    pad_to_multiple: int = 1,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Build NeMo bulk columns, driver carry, and primitive TQ tags."""
    torch = _import_torch()
    np = _import_numpy()

    if not samples:
        raise ValueError("Cannot build a NeMo payload with zero samples")
    max_sequence_length = _round_up(
        max(sample.sequence_length for sample in samples),
        max(1, int(pad_to_multiple)),
    )
    max_prompt_length = max(len(sample.prompt_ids_for_adv) for sample in samples)
    max_turns = 2
    batch_size = len(samples)

    input_ids = torch.full(
        (batch_size, max_sequence_length),
        int(pad_token_id),
        dtype=torch.long,
    )
    generation_logprobs = torch.zeros((batch_size, max_sequence_length), dtype=torch.float32)
    token_mask = torch.zeros((batch_size, max_sequence_length), dtype=torch.float32)
    prompt_ids_for_adv = torch.full(
        (batch_size, max_prompt_length),
        int(pad_token_id),
        dtype=torch.long,
    )
    turn_lengths = torch.zeros((batch_size, max_turns), dtype=torch.long)
    turn_roles = np.empty(batch_size, dtype=object)
    turn_contents = np.empty(batch_size, dtype=object)

    input_lengths: list[int] = []
    total_reward: list[float] = []
    loss_multiplier: list[float] = []
    truncated: list[bool] = []
    response_lengths: list[int] = []
    sample_mask: list[float] = []
    content = np.empty(batch_size, dtype=object)
    tags: list[dict[str, Any]] = []

    for row, sample in enumerate(samples):
        seq_len = sample.sequence_length
        prompt_len = sample.prompt_length
        input_ids[row, :seq_len] = torch.tensor(sample.input_ids, dtype=torch.long)
        generation_logprobs[row, :seq_len] = torch.tensor(
            sample.generation_logprobs,
            dtype=torch.float32,
        )
        token_mask[row, :seq_len] = torch.tensor(sample.token_mask, dtype=torch.float32)
        prompt_ids_for_adv[row, : len(sample.prompt_ids_for_adv)] = torch.tensor(
            sample.prompt_ids_for_adv,
            dtype=torch.long,
        )
        input_lengths.append(seq_len)
        total_reward.append(float(sample.total_reward))
        loss_multiplier.append(float(sample.sample_mask))
        sample_mask.append(float(sample.sample_mask))
        truncated.append(bool(sample.truncated))
        response_lengths.append(int(sample.response_length))
        response_text = str(
            ((sample.metadata.get("trace_debug") or {}).get("response_text") or "")
        )
        prompt_text = _prompt_text_from_metadata(sample.metadata)
        content[row] = response_text
        turn_lengths[row, 0] = prompt_len
        turn_lengths[row, 1] = sample.response_length
        turn_roles[row] = ["user", "assistant"]
        turn_contents[row] = [prompt_text, response_text]
        tags.append(
            {
                "sample_mask": int(sample.sample_mask),
                "reward": float(sample.total_reward),
                "group_index": int(sample.group_index),
                "trajectory_index": int(sample.trajectory_index),
                "trace_index": int(sample.trace_index),
                "session_id": sample.session_id,
                "task_id": sample.task_id,
            }
        )

    bulk_batch = {
        "input_ids": input_ids,
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        "generation_logprobs": generation_logprobs,
        "token_mask": token_mask,
        "sample_mask": torch.tensor(sample_mask, dtype=torch.float32),
        "content": content,
        "turn_lengths": turn_lengths,
        "turn_roles": turn_roles,
        "turn_contents": turn_contents,
    }
    driver_carry = {
        "total_reward": torch.tensor(total_reward, dtype=torch.float32),
        "loss_multiplier": torch.tensor(loss_multiplier, dtype=torch.float32),
        "truncated": torch.tensor(truncated, dtype=torch.bool),
        "length": torch.tensor(input_lengths, dtype=torch.long),
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        "prompt_ids_for_adv": prompt_ids_for_adv,
        "response_token_lengths": torch.tensor(response_lengths, dtype=torch.long),
        "turn_roles": turn_roles,
        "turn_contents": turn_contents,
    }
    return bulk_batch, driver_carry, tags


class _PolarSyncRolloutActor:
    """Implementation class; wrapped with ``ray.remote`` when Ray is installed."""

    def __init__(
        self,
        policy_generation: Any,
        tokenizer: Any,
        task_to_env: dict[str, Any],
        master_config: Any,
        dp_cfg: dict[str, Any],
    ) -> None:
        del task_to_env
        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.master_config = master_config
        self.polar_config = _polar_config(master_config)
        self._rollout_step = 0

        from nemo_rl.data_plane import build_data_plane_client

        self._dp_client = build_data_plane_client(dp_cfg, bootstrap=False)

    def rollout_to_tq(
        self,
        input_batch: Any,
        *,
        partition_id: str,
        group_size: int = 1,
        first_iter: bool = True,
        finish_generation: bool = True,
        task_to_env_override: Optional[dict[str, Any]] = None,
        carry_keys: Optional[list[str]] = None,
    ) -> tuple[Any, dict[str, Any], dict[str, Any], Optional[dict[str, Any]]]:
        del task_to_env_override
        torch = _import_torch()
        del torch
        np = _import_numpy()
        del np

        if self.policy_generation is not None:
            if first_iter and hasattr(self.policy_generation, "snapshot_step_metrics"):
                self.policy_generation.snapshot_step_metrics()
            self.policy_generation.clear_logger_metrics()

        self._rollout_step += 1
        policy_version = int(self.polar_config.get("policy_version", self._rollout_step))
        accepted_rollout_id = int(
            self.polar_config.get("accepted_rollout_id", self._rollout_step)
        )

        messages_by_prompt = _group_prompt_messages(input_batch, group_size=group_size)
        print(
            "PolarSyncRolloutActor: submitting "
            f"{len(messages_by_prompt)} prompt groups x {group_size} attempts",
            flush=True,
        )
        samples = self._collect_samples(
            messages_by_prompt=messages_by_prompt,
            group_size=group_size,
            policy_version=policy_version,
            accepted_rollout_id=accepted_rollout_id,
        )

        max_policy_staleness = self.polar_config.get("max_policy_staleness")
        validate_rollout_rows(
            samples,
            max_policy_staleness=(
                None if max_policy_staleness is None else int(max_policy_staleness)
            ),
        )
        if bool(self.polar_config.get("require_reward_variance", False)):
            validate_reward_variance(
                samples,
                near_zero_threshold=float(
                    self.polar_config.get("reward_near_zero_threshold", 1e-5)
                ),
                max_near_zero_fraction=float(
                    self.polar_config.get("max_near_zero_reward_fraction", 0.5)
                ),
                min_group_size=int(
                    self.polar_config.get("reward_variance_min_group_size", 2)
                ),
            )

        from nemo_rl.data_plane.column_io import kv_first_write
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        pad_to_multiple = int(_cfg_get(self.master_config, "policy", {}).get(
            "make_sequence_length_divisible_by",
            1,
        ) or 1)
        bulk_columns, driver_carry, tags = build_nemo_payload_columns(
            samples,
            pad_token_id=int(getattr(self.tokenizer, "pad_token_id", 0) or 0),
            pad_to_multiple=pad_to_multiple,
        )
        bulk_batch = BatchedDataDict(bulk_columns)
        n_prompts = len(messages_by_prompt)
        n_samples = len(samples)
        if n_prompts == 0 or n_samples % n_prompts != 0:
            raise ValueError(
                f"Polar rollout produced {n_samples} samples for {n_prompts} prompts"
            )
        n_per_prompt = n_samples // n_prompts
        uids = [str(uuid.uuid4()) for _ in range(n_prompts)]
        sample_ids = [f"{uid}_g{i}" for uid in uids for i in range(n_per_prompt)]
        rollout_metrics = _rollout_metrics(samples)
        rollout_metrics.update(
            {
                "polar_rollout_step": self._rollout_step,
                "polar_policy_version": policy_version,
                "polar_accepted_rollout_id": accepted_rollout_id,
            }
        )

        meta = kv_first_write(
            bulk_batch,
            sample_ids=sample_ids,
            dp_client=self._dp_client,
            partition_id=partition_id,
            extra_info={"rollout_metrics": rollout_metrics},
            task_name=partition_id,
            pad_to_multiple=pad_to_multiple,
            tags=tags,
        )
        print(
            "PolarSyncRolloutActor: wrote "
            f"{len(samples)} samples to TransferQueue partition={partition_id}; "
            f"trainable={sum(sample.sample_mask for sample in samples)}",
            flush=True,
        )

        if carry_keys is not None:
            missing = set(carry_keys) - driver_carry.keys()
            if missing:
                raise KeyError(
                    f"Polar rollout carry_keys {sorted(missing)} not produced; "
                    f"valid keys: {sorted(driver_carry)}"
                )
            driver_carry = {key: driver_carry[key] for key in carry_keys}

        if self.policy_generation is not None:
            if finish_generation:
                self.policy_generation.finish_generation()
            gen_metrics = self.policy_generation.get_logger_metrics()
        else:
            gen_metrics = None
        return meta, BatchedDataDict(driver_carry), rollout_metrics, gen_metrics

    def shutdown(self) -> None:
        try:
            self._dp_client.close()
        except Exception:
            pass

    def _collect_samples(
        self,
        *,
        messages_by_prompt: list[list[dict[str, str]]],
        group_size: int,
        policy_version: int,
        accepted_rollout_id: int,
    ) -> list[NeMoRolloutSample]:
        from nemo_bridge.adapter import session_result_to_nemo_samples
        from polar.rollout.models import TaskResult

        rollout_url = str(self.polar_config.get("rollout_url") or "").rstrip("/")
        if not rollout_url:
            raise ValueError("master_config.polar.rollout_url is required")

        submitted: list[tuple[int, str]] = []
        timeout_seconds = float(self.polar_config.get("task_timeout_seconds", 300.0))
        poll_interval = float(self.polar_config.get("poll_interval_seconds", 1.0))
        http_timeout = httpx.Timeout(
            None,
            connect=float(self.polar_config.get("http_connect_timeout_seconds", 30.0)),
        )
        trace_selection = str(self.polar_config.get("trace_selection", "last"))
        deadline = time.monotonic() + timeout_seconds

        with httpx.Client(base_url=rollout_url, timeout=http_timeout) as client:
            for group_index, messages in enumerate(messages_by_prompt):
                payload = self._task_payload(
                    group_index=group_index,
                    messages=messages,
                    num_samples=group_size,
                    policy_version=policy_version,
                    accepted_rollout_id=accepted_rollout_id,
                )
                response = client.post("/rollout/task/submit", json=payload)
                response.raise_for_status()
                submitted.append((group_index, response.json()["task_id"]))

            task_results: dict[str, TaskResult] = {}
            while len(task_results) < len(submitted):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Timed out waiting for {len(submitted)} Polar tasks "
                        f"after {timeout_seconds:.1f}s"
                    )
                time.sleep(max(0.05, poll_interval))
                for _, task_id in submitted:
                    if task_id in task_results:
                        continue
                    response = client.get(f"/rollout/task/{task_id}")
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get("status") in {"completed", "failed"}:
                        task_results[task_id] = TaskResult.model_validate(payload)

        flat: list[NeMoRolloutSample] = []
        for group_index, task_id in submitted:
            task_result = task_results[task_id]
            for trajectory_index, result in enumerate(task_result.results):
                samples = session_result_to_nemo_samples(
                    result,
                    group_index,
                    trajectory_index=trajectory_index,
                    accepted_rollout_id=accepted_rollout_id,
                    scheduler_group_id=group_index,
                )
                flat.append(
                    select_one_sample_per_session(
                        samples,
                        trace_selection=trace_selection,
                    )
                )
        return flat

    def _task_payload(
        self,
        *,
        group_index: int,
        messages: list[dict[str, str]],
        num_samples: int,
        policy_version: int,
        accepted_rollout_id: int,
    ) -> dict[str, Any]:
        task_id_prefix = str(self.polar_config.get("task_id_prefix", "nemo-polar"))
        model_name = str(
            self.polar_config.get("model_name")
            or _cfg_get(self.master_config, "policy", {}).get("model_name")
            or "model"
        )
        runtime = deepcopy(self.polar_config.get("runtime") or {})
        agent = build_default_shell_agent(
            messages=messages,
            model_name=model_name,
            max_tokens=int(self.polar_config.get("max_tokens", 16)),
            temperature=float(self.polar_config.get("temperature", 1.0)),
            top_p=float(self.polar_config.get("top_p", 1.0)),
            request_timeout_seconds=float(
                self.polar_config.get("request_timeout_seconds", 120.0)
            ),
            base_agent=deepcopy(self.polar_config.get("agent") or {}),
        )
        evaluator = deepcopy(
            self.polar_config.get("evaluator")
            or {"strategy": "session_completed"}
        )
        builder = deepcopy(
            self.polar_config.get("builder")
            or {"strategy": "per_request"}
        )
        metadata = deepcopy(self.polar_config.get("metadata") or {})
        metadata.update(
            {
                "group_id": group_index,
                "policy_version": policy_version,
                "accepted_rollout_id": accepted_rollout_id,
                "rollout_step": self._rollout_step,
                "messages": messages,
            }
        )
        instruction = _instruction_from_messages(messages)
        return {
            "task_id": f"{task_id_prefix}-g{group_index}-{uuid.uuid4().hex[:10]}",
            "instruction": instruction,
            "num_samples": int(num_samples),
            "timeout_seconds": float(
                self.polar_config.get("session_timeout_seconds", 180.0)
            ),
            "runtime": runtime or None,
            "agent": agent,
            "builder": builder,
            "evaluator": evaluator,
            "metadata": metadata,
        }


def _group_prompt_messages(input_batch: Any, *, group_size: int) -> list[list[dict[str, str]]]:
    message_logs = input_batch["message_log"]
    input_size = int(getattr(input_batch, "size", len(message_logs)))
    if group_size <= 0 or input_size % group_size != 0:
        raise ValueError(f"input_batch.size={input_size} is not divisible by group_size={group_size}")
    n_prompts = input_size // group_size
    return [
        message_log_to_openai_messages(message_logs[group_index * group_size])
        for group_index in range(n_prompts)
    ]


def _rollout_metrics(samples: list[NeMoRolloutSample]) -> dict[str, Any]:
    response_lengths = [sample.response_length for sample in samples]
    rewards = [sample.total_reward for sample in samples if sample.sample_mask == 1]
    return {
        "mean_gen_tokens_per_sample": (
            statistics.fmean(response_lengths) if response_lengths else 0.0
        ),
        "polar_num_samples": len(samples),
        "polar_num_trainable_samples": sum(sample.sample_mask for sample in samples),
        "polar_num_masked_samples": sum(1 for sample in samples if sample.sample_mask == 0),
        "polar_mean_reward": statistics.fmean(rewards) if rewards else 0.0,
    }


def _polar_config(master_config: Any) -> dict[str, Any]:
    raw = getattr(master_config, "polar", None)
    if raw is None and hasattr(master_config, "model_extra"):
        raw = (master_config.model_extra or {}).get("polar")
    if raw is None and isinstance(master_config, dict):
        raw = master_config.get("polar")
    if raw is None:
        raw = {}
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="python")
    if not isinstance(raw, dict):
        raise TypeError("master_config.polar must be a mapping")
    return dict(raw)


def _cfg_get(master_config: Any, key: str, default: Any) -> Any:
    if isinstance(master_config, dict):
        return master_config.get(key, default)
    value = getattr(master_config, key, default)
    if value is default and hasattr(master_config, "model_extra"):
        value = (master_config.model_extra or {}).get(key, default)
    return value if value is not None else default


def _prompt_text_from_metadata(metadata: dict[str, Any]) -> str:
    trace_debug = metadata.get("trace_debug") or {}
    prompt_messages = trace_debug.get("prompt_messages") or metadata.get("messages") or []
    if not prompt_messages:
        result_metadata = metadata.get("result_metadata") or {}
        prompt_messages = result_metadata.get("messages") or []
    if prompt_messages:
        return _instruction_from_messages(prompt_messages)
    return ""


def _instruction_from_messages(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{message.get('role', 'user')}: {_string_content(message.get('content'))}"
        for message in messages
    )


def _string_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(value)


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in NeMo env
        raise RuntimeError("nemo_bridge.rollout_actor requires torch inside NeMo RL") from exc
    return torch


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised in NeMo env
        raise RuntimeError("nemo_bridge.rollout_actor requires numpy inside NeMo RL") from exc
    return np


try:  # pragma: no cover - local unit tests do not require Ray.
    import ray
except ImportError:  # pragma: no cover
    PolarSyncRolloutActor = _PolarSyncRolloutActor
else:  # pragma: no cover
    PolarSyncRolloutActor = ray.remote(_PolarSyncRolloutActor)


__all__ = [
    "PolarSyncRolloutActor",
    "_PolarSyncRolloutActor",
    "build_default_shell_agent",
    "build_nemo_payload_columns",
    "message_log_to_openai_messages",
    "select_one_sample_per_session",
]
