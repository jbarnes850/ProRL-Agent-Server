"""External Polar rollout collector for NeMo native Async GRPO.

This module intentionally keeps the integration surface small: NeMo still owns
the trainer, replay buffer, staleness window, importance-sampling correction,
and vLLM weight sync. The collector only replaces NeMo's default rollout actor
with one that submits Polar tasks and converts the resulting traces into the
same per-prompt replay-buffer group shape NeMo already samples.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback
from typing import Any
from urllib import error, request


def normalize_openai_base_url(base_url: str) -> str:
    """Return the upstream root URL Polar expects, without a trailing `/v1`."""

    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized.rstrip("/")


def _format_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**context)
        except (KeyError, IndexError, ValueError):
            return value
    if isinstance(value, list):
        return [_format_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _format_value(item, context) for key, item in value.items()}
    return value


def _load_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text())


def env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, *, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def normalize_attempt_matrix(raw_matrix: Any) -> tuple[list[dict[str, Any]] | None, str]:
    """Normalize legacy and dataset attempt matrices.

    Legacy smoke matrices are lists and cycle attempts across generations.
    Dataset matrices are envelopes with ``selection_mode=group_cycle`` so each
    GRPO group uses one task repeated for every sampled completion.
    """

    if raw_matrix is None:
        return None, "generation_cycle"
    if isinstance(raw_matrix, dict):
        attempts = raw_matrix.get("attempts") or []
        if not isinstance(attempts, list):
            raise ValueError("attempt_matrix.attempts must be a list")
        return list(attempts), str(raw_matrix.get("selection_mode") or "group_cycle")
    if isinstance(raw_matrix, list):
        return list(raw_matrix), "generation_cycle"
    raise ValueError("attempt_matrix must be a list or object")


def select_attempts_for_group(
    attempts: list[dict[str, Any]],
    *,
    selection_mode: str,
    group_id: int,
    num_generations: int,
) -> list[dict[str, Any]]:
    if selection_mode == "group_cycle":
        return [attempts[group_id % len(attempts)] for _ in range(num_generations)]
    return [attempts[i % len(attempts)] for i in range(num_generations)]


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    with request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_control(url: str, *, timeout: float) -> dict[str, Any] | None:
    req = request.Request(url, data=b"", method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8").strip()
            return json.loads(text) if text else None
    except error.URLError:
        raise


def _message_text(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""
    parts: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif content is not None:
            parts.append(json.dumps(content, sort_keys=True))
    return "\n".join(parts)


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    if tokenizer is None:
        return ""
    try:
        return str(tokenizer.decode(token_ids, skip_special_tokens=False))
    except Exception:
        return ""


def _assistant_segments(
    *,
    tokenizer: Any,
    response_ids: list[int],
    response_logprobs: list[float],
    loss_mask: list[int],
    fallback_text: str,
) -> list[dict[str, Any]]:
    """Split assistant tokens so NeMo can preserve Polar's per-token loss mask."""

    import torch

    if not response_ids:
        return []
    if len(response_logprobs) != len(response_ids):
        raise ValueError("response_logprobs length must match response_ids")
    if len(loss_mask) != len(response_ids):
        raise ValueError("loss_mask length must match response_ids")

    messages: list[dict[str, Any]] = []
    start = 0
    while start < len(response_ids):
        trainable = int(loss_mask[start]) == 1
        end = start + 1
        while end < len(response_ids) and (int(loss_mask[end]) == 1) == trainable:
            end += 1

        token_slice = response_ids[start:end]
        logprob_slice = response_logprobs[start:end]
        content = _decode_tokens(tokenizer, token_slice)
        if not content and start == 0:
            content = fallback_text
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "token_ids": torch.tensor(token_slice, dtype=torch.long),
        }
        if trainable:
            message["generation_logprobs"] = torch.tensor(
                logprob_slice,
                dtype=torch.float32,
            )
        messages.append(message)
        start = end
    return messages


def _dummy_trace(tokenizer: Any) -> dict[str, Any]:
    pad_id = int(getattr(tokenizer, "pad_token_id", None) or 0)
    eos_id = int(getattr(tokenizer, "eos_token_id", None) or pad_id)
    return {
        "prompt_ids": [pad_id],
        "response_ids": [eos_id],
        "loss_mask": [0],
        "response_logprobs": [0.0],
        "prompt_messages": [{"role": "user", "content": ""}],
        "response_messages": [{"role": "assistant", "content": ""}],
        "reward": 0.0,
        "finish_reason": "polar_placeholder",
    }


def _first_non_none(values: list[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    prefix_len = 0
    limit = min(len(left), len(right))
    while prefix_len < limit and left[prefix_len] == right[prefix_len]:
        prefix_len += 1
    return prefix_len


def _flatten_traces_for_nemo(
    traces: list[dict[str, Any]],
    *,
    outcome_reward: float | None = None,
) -> dict[str, Any]:
    """Flatten Polar completion traces into one NeMo multi-turn trajectory trace.

    Polar's prefix-merging builder already emits a single trace for a chained
    multi-turn session. This helper also handles per-request traces by
    concatenating later prompt suffixes as zero-loss interstitial tokens, then
    appending the sampled assistant tokens/logprobs for each turn.
    """

    if len(traces) == 1:
        trace = dict(traces[0])
        if trace.get("reward") is None and outcome_reward is not None:
            trace["reward"] = outcome_reward
        trace.setdefault("metadata", {})
        trace["metadata"] = {
            **dict(trace.get("metadata") or {}),
            "polar_trace_count": 1,
            "polar_trace_flattening": "single_trace",
        }
        return trace

    first_trace = traces[0]
    prompt_ids = list(first_trace.get("prompt_ids") or [])
    response_ids: list[int] = []
    response_logprobs: list[float] = []
    loss_mask: list[int] = []
    response_messages: list[dict[str, Any]] = []
    finish_reason: str | None = None
    reconstruction_warnings: list[str] = []

    for trace_index, trace in enumerate(traces):
        current_prompt_ids = list(trace.get("prompt_ids") or [])
        if trace_index > 0 and current_prompt_ids:
            current_stream = prompt_ids + response_ids
            prefix_len = _common_prefix_len(current_stream, current_prompt_ids)
            interstitial_ids = current_prompt_ids[prefix_len:]
            if prefix_len == 0:
                reconstruction_warnings.append(
                    f"trace {trace_index} prompt had no token prefix overlap"
                )
            response_ids.extend(interstitial_ids)
            response_logprobs.extend([0.0] * len(interstitial_ids))
            loss_mask.extend([0] * len(interstitial_ids))

        turn_response_ids = list(trace.get("response_ids") or [])
        turn_logprobs = trace.get("response_logprobs")
        if turn_logprobs is None:
            turn_logprobs = []
        turn_logprobs = [float(value) for value in turn_logprobs]
        turn_loss_mask = [int(value) for value in (trace.get("loss_mask") or [])]
        response_ids.extend(turn_response_ids)
        response_logprobs.extend(turn_logprobs)
        loss_mask.extend(turn_loss_mask)
        response_messages.extend(list(trace.get("response_messages") or []))
        finish_reason = trace.get("finish_reason") or finish_reason

    reward = _first_non_none([trace.get("reward") for trace in reversed(traces)])
    if reward is None:
        reward = outcome_reward
    return {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "loss_mask": loss_mask,
        "prompt_messages": list(first_trace.get("prompt_messages") or []),
        "response_messages": response_messages,
        "response_logprobs": response_logprobs,
        "reward": reward,
        "finish_reason": finish_reason,
        "metadata": {
            "polar_trace_count": len(traces),
            "polar_trace_flattening": "per_request_concat",
            "reconstruction_warnings": reconstruction_warnings,
        },
    }


def _result_trace(result: dict[str, Any], tokenizer: Any) -> tuple[dict[str, Any], bool]:
    trajectory = result.get("trajectory") or {}
    traces = trajectory.get("traces") or []
    if not traces:
        return _dummy_trace(tokenizer), False
    trace = _flatten_traces_for_nemo(
        list(traces),
        outcome_reward=(trajectory.get("metadata") or {})
        .get("evaluation", {})
        .get("outcome_reward"),
    )
    prompt_ids = list(trace.get("prompt_ids") or [])
    response_ids = list(trace.get("response_ids") or [])
    response_logprobs = list(trace.get("response_logprobs") or [])
    loss_mask = [int(value) for value in (trace.get("loss_mask") or [])]
    valid = (
        str(result.get("status")) == "COMPLETED"
        and bool(prompt_ids)
        and bool(response_ids)
        and len(response_logprobs) == len(response_ids)
        and len(loss_mask) == len(response_ids)
        and any(value == 1 for value in loss_mask)
        and trace.get("reward") is not None
    )
    if valid:
        return trace, True
    dummy = _dummy_trace(tokenizer)
    dummy["reward"] = float(trace.get("reward") or 0.0)
    return dummy, False


def build_nemo_trajectory_group(
    task_results: list[dict[str, Any]],
    *,
    tokenizer: Any = None,
    reward_key: str = "score",
) -> dict[str, Any]:
    """Convert Polar task status payloads into one NeMo replay-buffer group."""

    del reward_key
    import torch
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    message_logs: list[list[dict[str, Any]]] = []
    lengths: list[int] = []
    loss_multiplier: list[float] = []
    total_reward: list[float] = []
    truncated: list[bool] = []
    completion_count = 0
    valid_count = 0
    trace_count = 0

    for task_result in task_results:
        for result in task_result.get("results") or []:
            trace, valid = _result_trace(result, tokenizer)
            prompt_ids = list(trace.get("prompt_ids") or [])
            response_ids = list(trace.get("response_ids") or [])
            response_logprobs = [float(x) for x in (trace.get("response_logprobs") or [])]
            loss_mask = [int(x) for x in (trace.get("loss_mask") or [])]

            prompt_text = _message_text(trace.get("prompt_messages"))
            response_text = _message_text(trace.get("response_messages"))
            message_log: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": prompt_text,
                    "token_ids": torch.tensor(prompt_ids, dtype=torch.long),
                }
            ]
            message_log.extend(
                _assistant_segments(
                    tokenizer=tokenizer,
                    response_ids=response_ids,
                    response_logprobs=response_logprobs,
                    loss_mask=loss_mask,
                    fallback_text=response_text,
                )
            )
            message_logs.append(message_log)
            lengths.append(len(prompt_ids))
            loss_multiplier.append(1.0 if valid else 0.0)
            reward = float(trace.get("reward") or 0.0)
            total_reward.append(reward)
            truncated.append(False)
            completion_count += int((result.get("trajectory") or {}).get("metadata", {}).get("record_count") or 0)
            trace_count += int((trace.get("metadata") or {}).get("polar_trace_count") or 1)
            valid_count += int(valid)

    if not message_logs:
        raise ValueError("Polar task results did not contain any sessions")

    batch = BatchedDataDict(
        {
            "message_log": message_logs,
            "length": torch.tensor(lengths, dtype=torch.int32),
            "loss_multiplier": torch.tensor(loss_multiplier, dtype=torch.float32),
            "total_reward": torch.tensor(total_reward, dtype=torch.float32),
            "truncated": torch.tensor(truncated, dtype=torch.bool),
        }
    )
    response_lengths = [
        sum(len(message["token_ids"]) for message in log if message.get("role") == "assistant")
        for log in message_logs
    ]
    rewards_tensor = batch["total_reward"]
    metrics = {
        "mean_gen_tokens_per_sample": float(sum(response_lengths) / len(response_lengths)),
        "gen_tokens_per_sample/mean": float(sum(response_lengths) / len(response_lengths)),
        "total_reward/mean": float(rewards_tensor.mean().item()),
        "polar/reward_mean": float(rewards_tensor.mean().item()),
        "polar/reward_std": float(rewards_tensor.std(unbiased=False).item()),
        "polar/group_sessions": float(len(message_logs)),
        "polar/trainable_samples": float(batch["loss_multiplier"].sum().item()),
        "polar/completions": float(completion_count),
        "polar/traces": float(trace_count),
        "polar/valid_sessions": float(valid_count),
        "natural_termination_rate": 1.0,
        "truncation_rate": 0.0,
    }
    return {"batch": batch.to("cpu"), "rollout_metrics": metrics}


def _render_attempt_payload(
    template: dict[str, Any],
    *,
    context: dict[str, Any],
    num_samples: int,
    attempt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attempt = attempt or {}
    payload = _format_value(deepcopy(template), {**context, **attempt})
    payload["task_id"] = str(payload.get("task_id") or context["task_id"])
    payload["num_samples"] = int(num_samples)
    metadata = payload.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata.update(context)
        metadata.update({f"attempt_{key}": value for key, value in attempt.items()})
    runtime = payload.get("runtime")
    if isinstance(runtime, dict):
        env = runtime.setdefault("env", {})
        if isinstance(env, dict):
            for key, value in attempt.items():
                env[f"POLAR_{key.upper()}"] = str(value)
            if attempt.get("task_spec_json"):
                env["POLAR_ORIGINAL_TASK_SPEC_JSON"] = str(attempt["task_spec_json"])
    return payload


class _PolarAsyncTrajectoryCollector:
    def __init__(
        self,
        policy_generation: Any,
        tokenizer: Any,
        task_to_env: dict[str, Any],
        master_config: Any,
        replay_buffer: Any,
        start_step: int = 0,
    ) -> None:
        del task_to_env
        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.master_config = master_config
        self.replay_buffer = replay_buffer
        self.current_weight_version = int(start_step)
        self.initial_weight_version = int(start_step)
        self.running = False
        self.dataloader = None
        self._thread: threading.Thread | None = None
        self._pause = threading.Event()
        self._pause.set()
        self._refit_pause = threading.Event()
        self._refit_pause.set()
        self._lock = threading.Lock()
        self._generating_targets: set[int] = set()
        self._submitted_groups = 0
        self._request_timeout = float(os.environ.get("NEMO_POLAR_REQUEST_TIMEOUT", "900"))
        self._poll_interval = float(os.environ.get("NEMO_POLAR_POLL_INTERVAL", "2"))
        self._rollout_url = os.environ["NEMO_POLAR_ROLLOUT_URL"].rstrip("/")
        self._gateway_url = os.environ.get("NEMO_POLAR_GATEWAY_URL", "").rstrip("/")
        self._task_template = _load_json(os.environ["NEMO_POLAR_TASK_TEMPLATE_PATH"])
        matrix_path = os.environ.get("NEMO_POLAR_ATTEMPT_MATRIX_PATH")
        self._attempt_selection_mode = "generation_cycle"
        self._attempt_matrix = None
        self._group_collection_workers = max(1, env_int("NEMO_POLAR_GROUP_WORKERS", default=1))
        if matrix_path:
            self._attempt_matrix, self._attempt_selection_mode = normalize_attempt_matrix(
                _load_json(matrix_path)
            )
        print("🌐 Using Polar external rollout collector")
        print(f"   rollout_url={self._rollout_url}")
        print(f"   gateway_url={self._gateway_url or 'unset'}")
        print(f"   vllm_http={self._openai_server_base_url() or 'unknown'}")
        print(f"   group_collection_workers={self._group_collection_workers}")
        if self._attempt_matrix is not None:
            print(
                "   attempt_matrix="
                f"{len(self._attempt_matrix)} mode={self._attempt_selection_mode}"
            )

    def _openai_server_base_url(self) -> str | None:
        urls = getattr(self.policy_generation, "dp_openai_server_base_urls", None)
        if not urls:
            return None
        first = urls[0]
        if not first:
            return None
        return normalize_openai_base_url(str(first))

    def _max_age(self) -> int:
        return int(self.master_config.grpo["async_grpo"]["max_trajectory_age_steps"])

    def _target_weights(self, generation_weight_version: int) -> list[int]:
        return target_weights_for_generation(
            generation_weight_version=generation_weight_version,
            initial_weight_version=self.initial_weight_version,
            max_age=self._max_age(),
            max_num_steps=int(self.master_config.grpo["max_num_steps"]),
        )

    def _next_target(self, generation_weight_version: int) -> int | None:
        import ray

        last_generated = ray.get(
            self.replay_buffer.get_last_target_weight_already_generated.remote()
        )
        with self._lock:
            for target in self._target_weights(generation_weight_version):
                if target > last_generated and target not in self._generating_targets:
                    self._generating_targets.add(target)
                    return target
        return None

    def set_weight_version(self, version: int) -> None:
        self.current_weight_version = int(version)
        print(f"🔄 Polar collector weight version -> {version}")

    def pause(self) -> None:
        self._pause.clear()
        print("⏸️ Polar collector paused")

    def resume(self) -> None:
        self._pause.set()
        print("▶️ Polar collector resumed")

    def prepare_for_refit(self) -> None:
        self._refit_pause.clear()
        print("🔄 Polar collector pausing gateway generation before NeMo refit")
        if self._gateway_url:
            _post_control(
                f"{self._gateway_url}/admin/inference/pause?timeout_seconds=300",
                timeout=310,
            )

    def resume_after_refit(self) -> None:
        print("🔄 Polar collector resuming gateway generation after NeMo refit")
        if self._gateway_url:
            _post_control(f"{self._gateway_url}/admin/inference/resume", timeout=30)
        self._refit_pause.set()

    def get_dataloader_state(self) -> dict[str, Any]:
        if self.dataloader is not None and hasattr(self.dataloader, "state_dict"):
            try:
                return self.dataloader.state_dict()
            except Exception:
                return {}
        return {}

    def start_collection(self, dataloader: Any) -> None:
        self.running = True
        self.dataloader = dataloader
        self._thread = threading.Thread(target=self._loop, daemon=True, name="polar-nemo-collector")
        self._thread.start()
        print("Started Polar external trajectory collection")

    def _loop(self) -> None:
        try:
            while self.running:
                self._pause.wait()
                self._refit_pause.wait()
                target = self._next_target(self.current_weight_version)
                if target is None:
                    time.sleep(0.5)
                    continue
                groups_needed = prompts_per_target(self.master_config)
                for _ in range(groups_needed):
                    if not self.running:
                        break
                    self._pause.wait()
                    self._refit_pause.wait()
                    self._collect_group(
                        generation_weight_version=self.current_weight_version,
                        target_weight_version=target,
                    )
        except Exception:
            print("❌ Polar collector failed")
            traceback.print_exc()
            self.running = False

    def _collect_group(
        self,
        *,
        generation_weight_version: int,
        target_weight_version: int,
    ) -> None:
        import ray

        group_id = self._submitted_groups
        self._submitted_groups += 1
        num_generations = int(self.master_config.grpo["num_generations_per_prompt"])
        context = {
            "group_id": group_id,
            "task_id": f"nemo-polar-{generation_weight_version}-{target_weight_version}-{group_id}",
            "weight_version": generation_weight_version,
            "target_weight_version": target_weight_version,
            "num_samples": num_generations,
            "model_name": str(getattr(self.master_config.policy, "model_name", "")),
        }
        attempts = self._attempt_matrix
        if attempts:
            selected = select_attempts_for_group(
                attempts,
                selection_mode=self._attempt_selection_mode,
                group_id=group_id,
                num_generations=num_generations,
            )
            payloads = [
                _render_attempt_payload(
                    self._task_template,
                    context={
                        **context,
                        "attempt_index": i,
                        "task_id": f"{context['task_id']}-a{i}",
                    },
                    num_samples=1,
                    attempt=attempt,
                )
                for i, attempt in enumerate(selected)
            ]
            max_workers = min(num_generations, self._group_collection_workers)
            if max_workers > 1:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    task_results = list(executor.map(self._submit_attempt, payloads))
            else:
                task_results = [self._submit_attempt(payload) for payload in payloads]
        else:
            payload = _render_attempt_payload(
                self._task_template,
                context=context,
                num_samples=num_generations,
            )
            task_results = [self._submit_attempt(payload)]

        group = build_nemo_trajectory_group(task_results, tokenizer=self.tokenizer)
        rewards = group["batch"]["total_reward"].tolist()
        reward_std = group["rollout_metrics"]["polar/reward_std"]
        valid_sessions = group["rollout_metrics"]["polar/valid_sessions"]
        group_sessions = group["rollout_metrics"]["polar/group_sessions"]
        if valid_sessions < group_sessions:
            raise RuntimeError(
                "Polar collector refusing invalid group before replay-buffer add: "
                f"valid_sessions={valid_sessions} group_sessions={group_sessions}"
            )
        if reward_std < 1e-5:
            print(
                "⚠️ Polar collector observed near-zero reward variance: "
                f"rewards={rewards} reward_std={reward_std:.6f}"
            )
        print(
            "📦 Polar collector adding group "
            f"group_id={group_id} weight={generation_weight_version} "
            f"target={target_weight_version} rewards={rewards} reward_std={reward_std:.6f}"
        )
        status = ray.get(
            self.replay_buffer.add.remote(
                group,
                generation_weight_version,
                target_weight_version,
            )
        )
        if status != "success":
            raise RuntimeError(f"ReplayBuffer.add returned {status!r}")

    def _submit_attempt(self, payload: dict[str, Any]) -> dict[str, Any]:
        submit = _post_json(
            f"{self._rollout_url}/rollout/task/submit",
            payload,
            timeout=self._request_timeout,
        )
        task_id = str(submit["task_id"])
        deadline = time.monotonic() + self._request_timeout
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Timed out waiting for Polar task {task_id}")
            time.sleep(self._poll_interval)
            status = _get_json(
                f"{self._rollout_url}/rollout/task/{task_id}",
                timeout=min(self._request_timeout, 60),
            )
            if status.get("status") != "running":
                if status.get("status") not in {"completed", "failed"}:
                    raise RuntimeError(f"Unexpected Polar task status: {status}")
                return status


try:
    import ray

    _remote_options: dict[str, Any] = {}
    _collector_node_ip = os.environ.get("NEMO_POLAR_COLLECTOR_NODE_IP")
    if _collector_node_ip:
        _remote_options["resources"] = {f"node:{_collector_node_ip}": 0.001}
    PolarAsyncTrajectoryCollector = ray.remote(**_remote_options)(_PolarAsyncTrajectoryCollector)
except Exception:
    PolarAsyncTrajectoryCollector = _PolarAsyncTrajectoryCollector  # type: ignore[assignment]


def reward_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def prompts_per_target(master_config: Any) -> int:
    return max(1, int(master_config.grpo["num_prompts_per_step"]))


def target_weights_for_generation(
    *,
    generation_weight_version: int,
    initial_weight_version: int,
    max_age: int,
    max_num_steps: int,
) -> list[int]:
    if generation_weight_version == initial_weight_version:
        candidates = list(range(initial_weight_version, initial_weight_version + max_age + 1))
    else:
        candidates = [generation_weight_version + i for i in range(1, max_age + 1)]
    max_target = initial_weight_version + max_num_steps - 1
    return [target for target in candidates if target <= max_target]


def stable_bucket(value: str, modulo: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest, 16) % modulo
