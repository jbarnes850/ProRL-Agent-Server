from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_audit_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "smoke" / "audit_nemo_polar_run.py"
    spec = importlib.util.spec_from_file_location("audit_nemo_polar_run", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_smoke_audit_does_not_require_fp8_for_default_vllm_run(tmp_path: Path) -> None:
    module = _load_audit_module()
    run_dir = tmp_path
    logs = run_dir / "logs"
    train_dir = logs / "grpo-polar" / "exp_001"
    train_dir.mkdir(parents=True)
    logs.mkdir(exist_ok=True)
    (logs / "exit-code.log").write_text("docker_exec_exit_code=0\n")
    (run_dir / "config.json").write_text(
        json.dumps({"generation": {"vllm_runtime": {"gpu_memory_utilization": 0.45}}})
    )
    (logs / "grpo-nemo-polar.log").write_text(
        "\n".join(
            [
                "Using Polar external rollout collector",
                "Polar collector adding group group_id=0 weight=0 target=0 rewards=[0.2, 0.225] reward_std=0.0125",
                "ReplayBuffer.add: Adding trajectory",
                "✅ Sampled 1 trajectory groups from buffer",
                "▶ Training policy",
                "🔄 Synchronizing policy weights to trajectory collector",
                "Async GRPO training complete",
            ]
        )
    )
    row = {
        "token_ids": [[1, 2, 3]],
        "token_loss_mask": [[0, 1, 1]],
        "generation_logprobs": [[0.0, -0.2, -0.3]],
        "prev_logprobs": [[0.0, -0.21, -0.29]],
        "advantages": [[-0.01, 0.01]],
        "rewards": [[0.2, 0.225]],
    }
    (train_dir / "train_data_step1.jsonl").write_text(json.dumps(row) + "\n")

    summary = module.audit(run_dir)

    assert summary["status"] == "passed"
    assert summary["evidence"]["fp8_vllm_precision"] is False
    assert "fp8_vllm_precision" not in summary["required_evidence"]


def test_smoke_audit_requires_fp8_when_config_requests_fp8(tmp_path: Path) -> None:
    module = _load_audit_module()
    run_dir = tmp_path
    logs = run_dir / "logs"
    train_dir = logs / "grpo-polar" / "exp_001"
    train_dir.mkdir(parents=True)
    logs.mkdir(exist_ok=True)
    (logs / "exit-code.log").write_text("docker_exec_exit_code=0\n")
    (run_dir / "config.json").write_text(
        json.dumps({"generation": {"vllm_runtime": {"precision": "fp8"}}})
    )
    (logs / "grpo-nemo-polar.log").write_text(
        "\n".join(
            [
                "Using Polar external rollout collector",
                "ReplayBuffer.add: Adding trajectory",
                "✅ Sampled 1 trajectory groups from buffer",
                "▶ Training policy",
                "🔄 Synchronizing policy weights to trajectory collector",
                "Async GRPO training complete",
            ]
        )
    )
    row = {
        "token_ids": [[1, 2]],
        "token_loss_mask": [[0, 1]],
        "generation_logprobs": [[0.0, -0.2]],
        "prev_logprobs": [[0.0, -0.21]],
        "advantages": [[0.0]],
        "rewards": [[0.2]],
    }
    (train_dir / "train_data_step1.jsonl").write_text(json.dumps(row) + "\n")

    summary = module.audit(run_dir)

    assert summary["status"] == "failed"
    assert "fp8_vllm_precision" in summary["required_evidence"]
