from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_audit_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "smoke" / "audit_nemo_polar_run.py"
    spec = importlib.util.spec_from_file_location("audit_nemo_polar_run", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_failed_audit_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_audit_module()
    (tmp_path / "logs").mkdir()
    monkeypatch.setattr("sys.argv", ["audit_nemo_polar_run.py", str(tmp_path)])

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 1
    assert json.loads((tmp_path / "validation_summary.json").read_text())["status"] == "failed"


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


def test_learning_audit_rejects_majority_near_zero_reward_groups(
    tmp_path: Path,
) -> None:
    module = _load_audit_module()
    logs = tmp_path / "logs"
    train_dir = logs / "grpo-polar" / "exp_001"
    train_dir.mkdir(parents=True)
    (logs / "exit-code.log").write_text("docker_exec_exit_code=0\n")
    (tmp_path / "config.json").write_text(json.dumps({}))
    (logs / "grpo-nemo-polar.log").write_text(
        "\n".join(
            [
                "Using Polar external rollout collector",
                "Polar collector adding group group_id=0 rewards=[0, 0] reward_std=0.000000",
                "Polar collector adding group group_id=1 rewards=[1, 1] reward_std=0.000000",
                "ReplayBuffer.add: Adding trajectory",
                "✅ Sampled 2 trajectory groups from buffer",
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
        "prev_logprobs": [[0.0, -0.2]],
        "advantages": [[0.0, 0.0]],
        "rewards": [[0.0, 0.0]],
    }
    (train_dir / "train_data_step1.jsonl").write_text(json.dumps(row) + "\n")

    infrastructure = module.audit(tmp_path)
    learning = module.audit(tmp_path, require_learning_signal=True)

    assert infrastructure["status"] == "passed"
    assert infrastructure["validation_mode"] == "infrastructure"
    assert learning["status"] == "failed"
    assert learning["validation_mode"] == "learning"
    assert learning["reward_variance"]["near_zero_group_fraction"] == 1.0
    assert "reward_variance_gate" in learning["required_evidence"]
    assert "nonzero_advantages" in learning["required_evidence"]


def test_top_p_replay_audit_rejects_missing_rollout_support(tmp_path: Path) -> None:
    module = _load_audit_module()
    logs = tmp_path / "logs"
    train_dir = logs / "grpo-polar" / "exp_001"
    train_dir.mkdir(parents=True)
    (logs / "exit-code.log").write_text("docker_exec_exit_code=0\n")
    (tmp_path / "config.json").write_text(
        json.dumps({"generation": {"top_p": 0.95}})
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
        "prev_logprobs": [[0.0, -0.2]],
        "advantages": [[0.0]],
        "rewards": [[0.0]],
    }
    (train_dir / "train_data_step1.jsonl").write_text(json.dumps(row) + "\n")

    summary = module.audit(tmp_path, require_sampling_distribution_replay=True)

    assert summary["status"] == "failed"
    assert summary["sampling_distribution"]["rollout_support_replay_needed"] is True
    assert summary["evidence"]["sampling_distribution_replay"] is False
    assert "sampling_distribution_replay" in summary["required_evidence"]


def test_top_p_replay_audit_accepts_complete_rollout_support(tmp_path: Path) -> None:
    module = _load_audit_module()
    logs = tmp_path / "logs"
    train_dir = logs / "grpo-polar" / "exp_001"
    train_dir.mkdir(parents=True)
    (logs / "exit-code.log").write_text("docker_exec_exit_code=0\n")
    (tmp_path / "config.json").write_text(
        json.dumps({"generation": {"top_p": 0.95}})
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
        "token_ids": [[1, 2, 3]],
        "token_loss_mask": [[0, 1, 1]],
        "generation_logprobs": [[0.0, -0.2, -0.3]],
        "prev_logprobs": [[0.0, -0.2, -0.3]],
        "advantages": [[0.0, 0.0]],
        "rewards": [[0.0]],
        "sampling_support_token_ids": [[[], [2, 4], [3, 5, 7]]],
    }
    (train_dir / "train_data_step1.jsonl").write_text(json.dumps(row) + "\n")

    summary = module.audit(tmp_path, require_sampling_distribution_replay=True)

    assert summary["status"] == "passed"
    assert summary["evidence"]["sampling_distribution_replay"] is True
    assert summary["train_data_audit"]["sampling_support_train_coverage"] == 1.0
    assert summary["train_data_audit"]["sampled_token_support_fraction"] == 1.0
