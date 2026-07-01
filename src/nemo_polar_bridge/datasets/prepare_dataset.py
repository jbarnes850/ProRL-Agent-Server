"""Prepare NeMo Gym-shaped dataset artifacts for NeMo Async GRPO + Polar rollouts."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import textwrap
from typing import Any

from nemo_polar_bridge.datasets.base import DatasetAdapter, TaskSpec
from nemo_polar_bridge.datasets.data_loader import (
    DEFAULT_NEMO_GYM_DATASET_ID,
    NeMoGymDatasetAdapter,
)
from nemo_polar_bridge.datasets.reasoning_gym_adapter import ReasoningGymDatasetAdapter
from nemo_polar_bridge.datasets.run_matrix import RunMatrixCell, build_run_matrix_cell
from nemo_polar_bridge.datasets.verifiers import portable_verifier_source


FINAL_ANSWER_INSTRUCTION = (
    "On the last line of your response, write 'Final answer:' followed by your "
    "final result and nothing else."
)
MULTI_TURN_EXECUTION_TYPES = {"multi_turn_chat_tool", "multi_step_chat_tool"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_task_template(
    args: argparse.Namespace,
    *,
    run_matrix: RunMatrixCell | None = None,
) -> dict[str, Any]:
    execution_type = str(getattr(args, "execution_type", "single_turn_chat") or "single_turn_chat")
    multi_turn = execution_type in MULTI_TURN_EXECUTION_TYPES
    command = _build_multi_turn_agent_command(args) if multi_turn else _build_agent_command(args)
    return {
        "task_id": (
            "nemo-polar-dataset-{weight_version}-{target_weight_version}-"
            "{group_id}-{attempt_index}-{name}"
        ),
        "instruction": (
            "Run a multi-turn NeMo Gym-shaped task row through live model calls "
            "and score the final episode with the task verifier."
            if multi_turn
            else "Sample one live model completion from a NeMo Gym-shaped task row "
            "and score it with the task verifier."
        ),
        "timeout_seconds": args.task_timeout_seconds,
        "runtime": {
            "backend": "docker",
            "image": args.runtime_image,
            "prepare": [
                {
                    "type": "exec",
                    "command": (
                        "rm -rf /polar/session/workspace && "
                        "mkdir -p /polar/session/workspace /polar/session/logs/agent && "
                        "cd /polar/session/workspace && git init -q && "
                        "git config user.email 'polar@test' && "
                        "git config user.name 'Polar' && "
                        "touch .keep && git add .keep && git commit -qm initial"
                    ),
                }
            ],
            "network": "host",
            "workdir": "/polar/session/workspace",
        },
        "agent": {
            "harness": "shell",
            "model_name": args.model_name,
            "custom_shell": {"command": command},
        },
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {
            "strategy": "verifier_result_file",
            "config": {
                "path": "/polar/session/workspace/verifier_result.json",
                "reward_key": "reward",
            },
            "refresh_runtime": False,
        },
        "metadata": {
            "purpose": "NeMo native Async GRPO with Polar live verifier dataset tasks",
            "source_schema": "NeMo Gym JSONL",
            "answer_format": getattr(args, "answer_format", "none"),
            "execution_type": execution_type,
            "run_matrix": {} if run_matrix is None else run_matrix.to_json_dict(),
            "reward_contract": (
                "One NeMo Gym task row is sampled N times by the current policy; "
                "each completion is scored by a deterministic verifier against "
                "answer/expected_answer."
            ),
        },
    }


def _build_agent_command(args: argparse.Namespace) -> str:
    verifier_source = portable_verifier_source()
    return f"""python3 - <<'PY'
import json
import os
import urllib.request
from pathlib import Path

{verifier_source}

def normalize_messages(value):
    if isinstance(value, str):
        return [{{"role": "user", "content": value}}]
    if isinstance(value, list):
        messages = []
        for message in value:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True)
            messages.append({{"role": role, "content": content}})
        if messages:
            return messages
    return [{{"role": "user", "content": str(value or "")}}]

def model_request_params(responses_create_params):
    params = {{}}
    for key, value in responses_create_params.items():
        if key == "input":
            continue
        if key == "top_k" and value not in (None, -1):
            continue
        if key in ("tools", "tool_choice") and value in (None, [], {{}}):
            continue
        params[key] = value
    return params

def load_verifier_task(payload_task):
    candidates = [os.environ.get("POLAR_ORIGINAL_TASK_SPEC_JSON")]
    if isinstance(payload_task, dict):
        metadata = payload_task.get("metadata") or {{}}
        if isinstance(metadata, dict):
            candidates.append(metadata.get("attempt_task_spec_json"))
    for raw in candidates:
        if not raw:
            continue
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("responses_create_params"), dict):
            return parsed
    return payload_task

def nemo_gym_response(content, model_name, raw_response):
    return {{
        "id": str(raw_response.get("id") or "polar_response"),
        "created_at": float(raw_response.get("created") or raw_response.get("created_at") or 0.0),
        "model": str(raw_response.get("model") or model_name),
        "object": "response",
        "output": [
            {{
                "id": "polar_message_0",
                "content": [
                    {{
                        "annotations": [],
                        "text": str(content or ""),
                        "type": "output_text",
                    }}
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }}
        ],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }}

def resource_verify_url(task):
    metadata = task.get("metadata") or {{}}
    agentic = metadata.get("agentic") if isinstance(metadata, dict) else {{}}
    if not isinstance(agentic, dict):
        agentic = {{}}
    return (
        os.environ.get("POLAR_GYM_RESOURCE_VERIFY_URL")
        or agentic.get("resource_verify_url")
        or agentic.get("verifier_url")
    )

def verify_task(content, task, raw_response, model_name):
    url = resource_verify_url(task)
    if not url:
        return verify_completion(content, task)
    payload = {{
        "responses_create_params": task.get("responses_create_params") or {{}},
        "response": nemo_gym_response(content, model_name, raw_response),
    }}
    exp_cal_state = coerce_jsonish(str(task.get("answer") or ""))
    if str(task.get("source_dataset") or "").casefold() == "calendar":
        payload["exp_cal_state"] = normalize_calendar_state(exp_cal_state)
    req = urllib.request.Request(
        str(url),
        data=json.dumps(payload).encode(),
        headers={{"Content-Type": "application/json"}},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout={int(args.model_request_timeout_seconds)}) as resp:
        verify_response = json.loads(resp.read())
    reward = float(verify_response.get("reward", 0.0))
    return {{
        "passed": reward > 0.0,
        "reward": reward,
        "reason": "gym_resource_server_verify",
        "normalized_completion": str(content or ""),
        "normalized_answer": json.dumps(exp_cal_state, sort_keys=True),
        "metadata": {{"verifier": "gym_resource_server", "resource_verify_url": str(url)}},
    }}

artifacts = Path(os.environ.get("ARTIFACTS_DIR", "/polar/session/artifacts"))
artifacts.mkdir(parents=True, exist_ok=True)
workspace = Path("/polar/session/workspace")
workspace.mkdir(parents=True, exist_ok=True)

payload_task = json.loads(os.environ["POLAR_TASK_SPEC_JSON"])
task = load_verifier_task(payload_task)
responses_create_params = dict(task.get("responses_create_params") or {{}})
messages = normalize_messages(responses_create_params.get("input"))
payload = {{
    **model_request_params(responses_create_params),
    "model": os.environ.get("POLAR_MODEL_NAME", {args.model_name!r}),
    "messages": messages,
    "max_tokens": int(os.environ.get("POLAR_MODEL_MAX_TOKENS", "{args.model_max_tokens}")),
    "temperature": float(os.environ.get("POLAR_MODEL_TEMPERATURE", "{args.model_temperature}")),
    "top_p": float(os.environ.get("POLAR_MODEL_TOP_P", "{args.model_top_p}")),
}}
base = os.environ["OPENAI_BASE_URL"].rstrip("/")
req = urllib.request.Request(
    base + "/chat/completions",
    data=json.dumps(payload).encode(),
    headers={{
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
    }},
    method="POST",
)
with urllib.request.urlopen(req, timeout={int(args.model_request_timeout_seconds)}) as resp:
    data = json.loads(resp.read())

content = data["choices"][0]["message"].get("content", "")
result = verify_task(
    content,
    task,
    data,
    os.environ.get("POLAR_MODEL_NAME", {args.model_name!r}),
)
(workspace / "model_response.txt").write_text(str(content))
(workspace / "verifier_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
(workspace / "dataset_task_metadata.json").write_text(json.dumps(task, indent=2, sort_keys=True))
(artifacts / "llm_probe.json").write_text(json.dumps(data, indent=2, sort_keys=True))
(artifacts / "verifier_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
PY"""


def _build_multi_turn_agent_command(args: argparse.Namespace) -> str:
    verifier_source = portable_verifier_source()
    return f"""python3 - <<'PY'
import json
import os
import urllib.request
from pathlib import Path

{verifier_source}

def normalize_messages(value):
    if isinstance(value, str):
        return [{{"role": "user", "content": value}}]
    if isinstance(value, list):
        messages = []
        for message in value:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "user")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True)
            normalized = {{"role": role, "content": content}}
            for key in ("name", "tool_call_id"):
                if key in message:
                    normalized[key] = message[key]
            messages.append(normalized)
        if messages:
            return messages
    return [{{"role": "user", "content": str(value or "")}}]

def model_request_params(responses_create_params):
    params = {{}}
    for key, value in responses_create_params.items():
        if key == "input":
            continue
        if key == "top_k" and value not in (None, -1):
            continue
        if key in ("tools", "tool_choice") and value in (None, [], {{}}):
            continue
        params[key] = value
    return params

def load_verifier_task(payload_task):
    candidates = [os.environ.get("POLAR_ORIGINAL_TASK_SPEC_JSON")]
    if isinstance(payload_task, dict):
        metadata = payload_task.get("metadata") or {{}}
        if isinstance(metadata, dict):
            candidates.append(metadata.get("attempt_task_spec_json"))
    for raw in candidates:
        if not raw:
            continue
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("responses_create_params"), dict):
            return parsed
    return payload_task

def nemo_gym_response(content, model_name, raw_response):
    return {{
        "id": str(raw_response.get("id") or "polar_response"),
        "created_at": float(raw_response.get("created") or raw_response.get("created_at") or 0.0),
        "model": str(raw_response.get("model") or model_name),
        "object": "response",
        "output": [
            {{
                "id": "polar_message_0",
                "content": [
                    {{
                        "annotations": [],
                        "text": str(content or ""),
                        "type": "output_text",
                    }}
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }}
        ],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }}

def resource_verify_url(task):
    metadata = task.get("metadata") or {{}}
    agentic = metadata.get("agentic") if isinstance(metadata, dict) else {{}}
    if not isinstance(agentic, dict):
        agentic = {{}}
    return (
        os.environ.get("POLAR_GYM_RESOURCE_VERIFY_URL")
        or agentic.get("resource_verify_url")
        or agentic.get("verifier_url")
    )

def verify_task(content, task, raw_response, model_name):
    url = resource_verify_url(task)
    if not url:
        return verify_completion(content, task)
    payload = {{
        "responses_create_params": task.get("responses_create_params") or {{}},
        "response": nemo_gym_response(content, model_name, raw_response),
    }}
    exp_cal_state = coerce_jsonish(str(task.get("answer") or ""))
    if str(task.get("source_dataset") or "").casefold() == "calendar":
        payload["exp_cal_state"] = normalize_calendar_state(exp_cal_state)
    req = urllib.request.Request(
        str(url),
        data=json.dumps(payload).encode(),
        headers={{"Content-Type": "application/json"}},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout={int(args.model_request_timeout_seconds)}) as resp:
        verify_response = json.loads(resp.read())
    reward = float(verify_response.get("reward", 0.0))
    return {{
        "passed": reward > 0.0,
        "reward": reward,
        "reason": "gym_resource_server_verify",
        "normalized_completion": str(content or ""),
        "normalized_answer": json.dumps(exp_cal_state, sort_keys=True),
        "metadata": {{"verifier": "gym_resource_server", "resource_verify_url": str(url)}},
    }}

def agentic_metadata(task):
    metadata = task.get("metadata") or {{}}
    agentic = metadata.get("agentic") if isinstance(metadata, dict) else {{}}
    return agentic if isinstance(agentic, dict) else {{}}

def observation_for_turn(agentic, turn_index, content):
    observations = agentic.get("tool_observations") or agentic.get("observations") or []
    if isinstance(observations, list) and turn_index < len(observations):
        value = observations[turn_index]
        return value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    scripted_tools = agentic.get("tools") or []
    if scripted_tools and "<tool" in str(content).lower():
        return json.dumps({{"status": "ok", "turn": turn_index, "content": str(content)[:200]}})
    return None

def call_model(base, payload):
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={{
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
        }},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout={int(args.model_request_timeout_seconds)}) as resp:
        return json.loads(resp.read())

artifacts = Path(os.environ.get("ARTIFACTS_DIR", "/polar/session/artifacts"))
artifacts.mkdir(parents=True, exist_ok=True)
workspace = Path("/polar/session/workspace")
workspace.mkdir(parents=True, exist_ok=True)

payload_task = json.loads(os.environ["POLAR_TASK_SPEC_JSON"])
task = load_verifier_task(payload_task)
agentic = agentic_metadata(task)
responses_create_params = dict(task.get("responses_create_params") or {{}})
messages = normalize_messages(responses_create_params.get("input"))
max_turns = int(agentic.get("max_rollout_turns") or agentic.get("max_turns") or agentic.get("max_steps") or 2)
max_turns = max(1, max_turns)
base = os.environ["OPENAI_BASE_URL"].rstrip("/")
transcript = []
final_content = ""
last_probe = {{}}

for turn_index in range(max_turns):
    payload = {{
        **model_request_params(responses_create_params),
        "model": os.environ.get("POLAR_MODEL_NAME", {args.model_name!r}),
        "messages": messages,
        "max_tokens": int(os.environ.get("POLAR_MODEL_MAX_TOKENS", "{args.model_max_tokens}")),
        "temperature": float(os.environ.get("POLAR_MODEL_TEMPERATURE", "{args.model_temperature}")),
        "top_p": float(os.environ.get("POLAR_MODEL_TOP_P", "{args.model_top_p}")),
    }}
    last_probe = call_model(base, payload)
    message = last_probe["choices"][0].get("message") or {{}}
    final_content = str(message.get("content") or "")
    assistant_message = {{"role": "assistant", "content": final_content}}
    if message.get("tool_calls"):
        assistant_message["tool_calls"] = message["tool_calls"]
    messages.append(assistant_message)
    transcript.append({{"turn": turn_index, "role": "assistant", "content": final_content}})

    observation = observation_for_turn(agentic, turn_index, final_content)
    if observation is None or turn_index == max_turns - 1:
        break
    tool_message = {{
        "role": "tool",
        "content": observation,
        "tool_call_id": f"polar-tool-{{turn_index}}",
    }}
    messages.append(tool_message)
    transcript.append({{"turn": turn_index, "role": "tool", "content": observation}})

result = verify_task(
    final_content,
    task,
    last_probe,
    os.environ.get("POLAR_MODEL_NAME", {args.model_name!r}),
)
(workspace / "model_response.txt").write_text(str(final_content))
(workspace / "multi_turn_transcript.json").write_text(json.dumps(transcript, indent=2, sort_keys=True))
(workspace / "verifier_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
(workspace / "dataset_task_metadata.json").write_text(json.dumps(task, indent=2, sort_keys=True))
(artifacts / "llm_probe.json").write_text(json.dumps(last_probe, indent=2, sort_keys=True))
(artifacts / "multi_turn_transcript.json").write_text(json.dumps(transcript, indent=2, sort_keys=True))
(artifacts / "verifier_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
PY"""


def build_attempt_matrix(
    attempts: list[dict[str, Any]],
    *,
    run_matrix: RunMatrixCell | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "schema": "nemo_gym_to_polar_attempt_matrix",
        "selection_mode": "group_cycle",
        "run_matrix": {} if run_matrix is None else run_matrix.to_json_dict(),
        "attempts": attempts,
        "grouping_contract": (
            "Select one task per GRPO prompt group, then submit the same task once "
            "per requested generation so rewards are comparable within the group."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default=DEFAULT_NEMO_GYM_DATASET_ID)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--scan-rows", type=int, default=200)
    parser.add_argument("--source-dataset", action="append", default=[])
    parser.add_argument("--local-jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--nemo-rl-ref", default="")
    parser.add_argument("--repo-host", default="")
    parser.add_argument("--head-ip", default="")
    parser.add_argument("--worker-ip", default="")
    parser.add_argument("--head-hostname", default="spark-f7e2")
    parser.add_argument("--worker-hostname", default="spark-cfd0")
    parser.add_argument("--polar-rollout-port", type=int, default=19080)
    parser.add_argument("--polar-gateway-port", type=int, default=19100)
    parser.add_argument("--vllm-http-port", type=int, default=31000)
    parser.add_argument("--model-max-tokens", type=int, default=1024)
    parser.add_argument("--model-temperature", type=float, default=0.6)
    parser.add_argument("--model-top-p", type=float, default=0.95)
    parser.add_argument("--model-request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--task-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--test-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--answer-format", choices=("none", "final_answer"), default="none")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--vllm-enforce-eager", default="true")
    parser.add_argument("--vllm-precision", default="bfloat16")
    parser.add_argument("--vllm-kv-cache-dtype", default="auto")
    parser.add_argument("--vllm-max-num-seqs", type=int)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int)
    parser.add_argument("--script-path", default="")
    parser.add_argument("--matrix-name", default="smoke")
    parser.add_argument("--matrix-cell", default="")
    parser.add_argument(
        "--dataset-family", default="nemo_gym", choices=("nemo_gym", "reasoning_gym")
    )
    parser.add_argument(
        "--reasoning-gym-task-name",
        default="basic_arithmetic",
        help="reasoning-gym task name, only used when --dataset-family=reasoning_gym",
    )
    parser.add_argument(
        "--reasoning-gym-seed",
        type=int,
        default=0,
        help="reasoning_gym.create_dataset seed, only used when --dataset-family=reasoning_gym",
    )
    parser.add_argument("--verifier-type", default="exact_answer")
    parser.add_argument("--execution-type", default="single_turn_chat")
    parser.add_argument("--difficulty-band", default="unspecified")
    parser.add_argument("--adapter-name", default="nemo_gym_jsonl")
    parser.add_argument("--matrix-tag", action="append", default=[])
    parser.add_argument("--matrix-notes", default="")
    return parser.parse_args()


def apply_answer_format(tasks: list[TaskSpec], answer_format: str) -> list[TaskSpec]:
    if answer_format == "none":
        return tasks
    if answer_format != "final_answer":
        raise ValueError(f"Unsupported answer format: {answer_format}")
    return [_with_final_answer_instruction(task) for task in tasks]


def _with_final_answer_instruction(task: TaskSpec) -> TaskSpec:
    responses_create_params = dict(task.responses_create_params)
    original_input = responses_create_params.get("input")
    formatted_input = _append_instruction_to_input(original_input, FINAL_ANSWER_INSTRUCTION)
    responses_create_params["input"] = formatted_input
    prompt = _extract_prompt_text(formatted_input)
    return replace(
        task,
        responses_create_params=responses_create_params,
        prompt=prompt,
        metadata={**task.metadata, "answer_format": "final_answer"},
    )


def _append_instruction_to_input(value: Any, instruction: str) -> Any:
    if isinstance(value, str):
        return _append_instruction(value, instruction)
    if isinstance(value, list):
        messages = [dict(message) if isinstance(message, dict) else message for message in value]
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "user") != "user":
                continue
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True)
            messages[index] = {**message, "content": _append_instruction(content, instruction)}
            return messages
        return [*messages, {"role": "user", "content": instruction}]
    return instruction if value is None else _append_instruction(str(value), instruction)


def _append_instruction(text: str, instruction: str) -> str:
    text = str(text or "").rstrip()
    if instruction.casefold() in text.casefold():
        return text
    return f"{text}\n\n{instruction}" if text else instruction


def _extract_prompt_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for message in value:
            if not isinstance(message, dict):
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                parts.append(content)
        return "\n\n".join(parts)
    return str(value or "")


def _build_adapter(args: argparse.Namespace) -> DatasetAdapter:
    if args.dataset_family == "reasoning_gym":
        return ReasoningGymDatasetAdapter(
            task_name=args.reasoning_gym_task_name,
            seed=args.reasoning_gym_seed,
            split=args.split,
            config=args.config,
        )
    if args.dataset_family != "nemo_gym":
        raise ValueError(f"Unsupported --dataset-family: {args.dataset_family!r}")
    return NeMoGymDatasetAdapter(
        dataset_id=args.dataset_id,
        config=args.config,
        split=args.split,
        local_jsonl=args.local_jsonl,
        source_datasets=args.source_dataset,
    )


def _build_config_dict(
    args: argparse.Namespace,
    adapter: DatasetAdapter,
    output_dir: Path,
    run_matrix: RunMatrixCell,
    tasks: list[TaskSpec],
) -> dict:
    return {
        "run_matrix": run_matrix.to_json_dict(),
        "dataset": {
            "id": adapter.dataset_id,
            "config": args.config,
            "split": args.split,
            "limit": args.limit,
            "scan_rows": args.scan_rows,
            "local_jsonl": args.local_jsonl,
            "source_dataset_filter": args.source_dataset,
            "canonical_schema": adapter.audit()["canonical_schema"],
            "selection_mode": "group_cycle",
            "answer_format": args.answer_format,
            "provenance": _summarize_task_provenance(tasks),
        },
        "image": args.image,
        "model": {"name": args.model_name, "path": args.model_path},
        "generation": {
            "max_new_tokens": args.model_max_tokens,
            "temperature": args.model_temperature,
            "top_p": args.model_top_p,
            "top_k": None,
            "vllm_generation_config": "vllm",
            "vllm_runtime": {
                "precision": args.vllm_precision,
                "kv_cache_dtype": args.vllm_kv_cache_dtype,
                "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                "enforce_eager": args.vllm_enforce_eager,
                "max_num_seqs": args.vllm_max_num_seqs,
                "max_num_batched_tokens": args.vllm_max_num_batched_tokens,
            },
        },
        "nemo_rl_ref": args.nemo_rl_ref,
        "polar": {
            "task_template": str(output_dir / "polar" / "task_template.json"),
            "attempt_matrix": str(output_dir / "polar" / "attempt_matrix.json"),
            "rollout_port": args.polar_rollout_port,
            "gateway_port": args.polar_gateway_port,
            "vllm_http_port": args.vllm_http_port,
        },
        "repo_host": args.repo_host,
        "run_dir": str(output_dir),
        "script_path": args.script_path,
        "topology": {
            "train_node": args.worker_hostname,
            "train_ip": args.worker_ip,
            "inference_rollout_node": args.head_hostname,
            "inference_rollout_ip": args.head_ip,
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    adapter = _build_adapter(args)
    tasks = apply_answer_format(
        adapter.load_tasks(limit=args.limit, scan_rows=args.scan_rows),
        args.answer_format,
    )
    run_matrix = build_run_matrix_cell(
        matrix_name=args.matrix_name,
        matrix_cell=args.matrix_cell,
        dataset_family=args.dataset_family,
        verifier_type=args.verifier_type,
        execution_type=args.execution_type,
        adapter=args.adapter_name,
        difficulty_band=args.difficulty_band,
        source_dataset_filter=args.source_dataset,
        tags=args.matrix_tag,
        notes=args.matrix_notes,
    )
    attempts = [task.to_attempt() for task in tasks]

    write_jsonl(output_dir / "data" / "train.jsonl", [task.to_train_row() for task in tasks])
    write_json(
        output_dir / "polar" / "attempt_matrix.json",
        build_attempt_matrix(attempts, run_matrix=run_matrix),
    )
    write_json(
        output_dir / "polar" / "task_template.json",
        build_task_template(args, run_matrix=run_matrix),
    )
    write_json(
        output_dir / "config.json",
        _build_config_dict(args, adapter, output_dir, run_matrix, tasks),
    )
    write_json(
        output_dir / "train_data_audit.json",
        {
            **adapter.audit(),
            "run_matrix": run_matrix.to_json_dict(),
            "reward_contract": (
                "Live verifier scoring only: dataset answers define ground truth, "
                "but reward is computed from each sampled completion."
            ),
            "answer_format": {
                "mode": args.answer_format,
                "instruction": FINAL_ANSWER_INSTRUCTION
                if args.answer_format == "final_answer"
                else None,
            },
            "grouping_contract": (
                "One task per prompt group; num_generations_per_prompt live "
                "completions are scored independently for that same task."
            ),
            "selected_tasks": [
                {
                    "task_id": task.stable_id(),
                    "row_idx": task.row_idx,
                    "source_dataset": task.source_dataset,
                    "agent_ref": task.agent_ref,
                    "answer": task.answer,
                    "prompt_head": task.prompt[:500],
                    "metadata": task.metadata,
                }
                for task in tasks
            ],
        },
    )
    print(
        textwrap.dedent(
            f"""\
            prepared_dataset={output_dir}
            dataset={adapter.dataset_id}
            canonical_schema={adapter.audit()["canonical_schema"]}
            run_matrix={run_matrix.name}
            selected_tasks={[task.stable_id() for task in tasks]}
            selection_mode=group_cycle
            reward_contract=live_verifier
            """
        ).strip()
    )


def _summarize_task_provenance(tasks: list[TaskSpec]) -> dict[str, Any]:
    hashes: dict[str, Any] = {}
    policy_ids: set[str] = set()
    source_datasets: set[str] = set()
    for task in tasks:
        if task.source_dataset:
            source_datasets.add(task.source_dataset)
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        policy_id = metadata.get("integrity_policy_id") or metadata.get("benchmark_integrity_policy_id")
        if policy_id:
            policy_ids.add(str(policy_id))
        task_hashes = metadata.get("dataset_hashes") or metadata.get("provenance_hashes")
        if isinstance(task_hashes, dict):
            hashes.update(task_hashes)
    return {
        "source_datasets": sorted(source_datasets),
        "benchmark_integrity_policy_ids": sorted(policy_ids),
        "dataset_hashes": hashes,
    }


if __name__ == "__main__":
    main()
