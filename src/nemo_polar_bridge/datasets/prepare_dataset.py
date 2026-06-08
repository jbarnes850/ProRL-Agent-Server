"""Prepare NeMo Gym-shaped dataset artifacts for NeMo Async GRPO + Polar rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap
from typing import Any

from nemo_polar_bridge.datasets.data_loader import (
    DEFAULT_REASONING_GYM_DATASET_ID,
    NeMoGymDatasetAdapter,
)
from nemo_polar_bridge.datasets.verifiers import portable_verifier_source


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_task_template(args: argparse.Namespace) -> dict[str, Any]:
    command = _build_agent_command(args)
    test_command = r"""cd /polar/session/workspace && python3 - <<'PY'
import json
from pathlib import Path

result = json.loads(Path("verifier_result.json").read_text())
if result.get("passed") is True:
    print("PASSED live_verifier")
else:
    print("FAILED live_verifier")
PY"""
    return {
        "task_id": (
            "nemo-polar-dataset-{weight_version}-{target_weight_version}-"
            "{group_id}-{attempt_index}-{name}"
        ),
        "instruction": (
            "Sample one live model completion from a NeMo Gym-shaped task row "
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
            "strategy": "test_on_output",
            "config": {
                "repo_dir": "/polar/session/workspace",
                "patch_command": "cd /polar/session/workspace && git add -A && git diff --cached --binary",
                "test_command": test_command,
                "test_timeout": args.test_timeout_seconds,
                "expected_output_json": {"live_verifier": "PASSED"},
                "exclude_patterns": [
                    "__pycache__/**",
                    "**/__pycache__/**",
                    ".pytest_cache/**",
                    "**/.pytest_cache/**",
                ],
            },
            "refresh_runtime": True,
        },
        "metadata": {
            "purpose": "NeMo native Async GRPO with Polar live verifier dataset tasks",
            "source_schema": "NeMo Gym JSONL",
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

artifacts = Path(os.environ.get("ARTIFACTS_DIR", "/polar/session/artifacts"))
artifacts.mkdir(parents=True, exist_ok=True)
workspace = Path("/polar/session/workspace")
workspace.mkdir(parents=True, exist_ok=True)

task = json.loads(os.environ["POLAR_TASK_SPEC_JSON"])
responses_create_params = dict(task.get("responses_create_params") or {{}})
messages = normalize_messages(responses_create_params.get("input"))
payload = {{
    **{{key: value for key, value in responses_create_params.items() if key != "input"}},
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
result = verify_completion(content, task)
(workspace / "model_response.txt").write_text(str(content))
(workspace / "verifier_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
(workspace / "dataset_task_metadata.json").write_text(json.dumps(task, indent=2, sort_keys=True))
(artifacts / "llm_probe.json").write_text(json.dumps(data, indent=2, sort_keys=True))
(artifacts / "verifier_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
PY"""


def build_attempt_matrix(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": 1,
        "schema": "nemo_gym_to_polar_attempt_matrix",
        "selection_mode": "group_cycle",
        "attempts": attempts,
        "grouping_contract": (
            "Select one task per GRPO prompt group, then submit the same task once "
            "per requested generation so rewards are comparable within the group."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default=DEFAULT_REASONING_GYM_DATASET_ID)
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
    parser.add_argument("--polar-rollout-port", type=int, default=19080)
    parser.add_argument("--polar-gateway-port", type=int, default=19100)
    parser.add_argument("--vllm-http-port", type=int, default=31000)
    parser.add_argument("--model-max-tokens", type=int, default=1024)
    parser.add_argument("--model-temperature", type=float, default=0.6)
    parser.add_argument("--model-top-p", type=float, default=0.95)
    parser.add_argument("--model-request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--task-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--test-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--script-path", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    adapter = NeMoGymDatasetAdapter(
        dataset_id=args.dataset_id,
        config=args.config,
        split=args.split,
        local_jsonl=args.local_jsonl,
        source_datasets=args.source_dataset,
    )
    tasks = adapter.load_tasks(limit=args.limit, scan_rows=args.scan_rows)
    attempts = [task.to_attempt() for task in tasks]

    write_jsonl(output_dir / "data" / "train.jsonl", [task.to_train_row() for task in tasks])
    write_json(output_dir / "polar" / "attempt_matrix.json", build_attempt_matrix(attempts))
    write_json(output_dir / "polar" / "task_template.json", build_task_template(args))
    write_json(
        output_dir / "config.json",
        {
            "dataset": {
                "id": args.dataset_id,
                "config": args.config,
                "split": args.split,
                "limit": args.limit,
                "scan_rows": args.scan_rows,
                "source_dataset_filter": args.source_dataset,
                "canonical_schema": "nemo_gym_jsonl",
                "selection_mode": "group_cycle",
            },
            "image": args.image,
            "model": {"name": args.model_name, "path": args.model_path},
            "generation": {
                "max_new_tokens": args.model_max_tokens,
                "temperature": args.model_temperature,
                "top_p": args.model_top_p,
                "top_k": None,
                "vllm_generation_config": "vllm",
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
                "train_node": "spark-cfd0",
                "train_ip": args.worker_ip,
                "inference_rollout_node": "spark-f7e2",
                "inference_rollout_ip": args.head_ip,
            },
        },
    )
    write_json(
        output_dir / "train_data_audit.json",
        {
            **adapter.audit(),
            "reward_contract": (
                "Live verifier scoring only: dataset answers define ground truth, "
                "but reward is computed from each sampled completion."
            ),
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
            dataset={args.dataset_id}
            canonical_schema=nemo_gym_jsonl
            selected_tasks={[task.stable_id() for task in tasks]}
            selection_mode=group_cycle
            reward_contract=live_verifier
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
