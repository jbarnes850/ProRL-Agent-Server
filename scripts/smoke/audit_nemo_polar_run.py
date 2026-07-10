#!/usr/bin/env python3
"""Summarize NeMo + Polar smoke evidence and training telemetry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _clean(text: str) -> str:
    return ANSI_RE.sub("", text)


def _safe_load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _flatten_one(value: Any) -> list[Any]:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []


def _float_stats(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "std": statistics.pstdev(clean),
        "min": min(clean),
        "max": max(clean),
    }


def _extract_scalar(pattern: str, text: str, *, default: float | None = None) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return default
    return float(match.group(1))


def _extract_steps(log_text: str) -> list[dict[str, Any]]:
    blocks = re.split(r"\n📊 Training Results:\n", log_text)[1:]
    steps: list[dict[str, Any]] = []
    for index, block in enumerate(blocks, start=1):
        step: dict[str, Any] = {"event": "training_step", "step": index}
        scalar_patterns = {
            "loss": r"• Loss: ([\-0-9.eE]+)",
            "generation_kl_error": r"• Generation KL Error: ([\-0-9.eE]+)",
            "avg_reward": r"• Avg Reward: ([\-0-9.eE]+)",
            "buffer_size": r"• Buffer Size: ([\-0-9.eE]+)",
            "avg_trajectory_age": r"• Avg Trajectory Age: ([\-0-9.eE]+) steps",
            "total_step_time_s": r"• Total step time: ([\-0-9.eE]+)s",
            "weight_sync_s": r"• weight_sync: ([\-0-9.eE]+)s",
            "policy_training_s": r"• policy_training: ([\-0-9.eE]+)s",
            "logprobs_s": r"• policy_and_reference_logprobs: ([\-0-9.eE]+)s",
            "exposed_generation_s": r"• exposed_generation: ([\-0-9.eE]+)s",
            "e2e_samples_per_sec_per_gpu": r"- E2E \(Samples/sec/gpu\): ([\-0-9.eE]+)",
            "e2e_tokens_per_sec_per_gpu": r"- E2E \(Tokens/sec/gpu\): ([\-0-9.eE]+)",
            "policy_training_tokens_per_sec_per_gpu": (
                r"- Policy Training \(Tokens/sec/gpu\): ([\-0-9.eE]+)"
            ),
            "logprobs_tokens_per_sec_per_gpu": (
                r"- Policy and Reference Logprobs \(Tokens/sec/gpu\): ([\-0-9.eE]+)"
            ),
            "generation_worker_tokens_per_sec_per_gpu": (
                r"- Generation Worker Group \(Tokens/sec/gpu\): ([\-0-9.eE]+)"
            ),
        }
        for key, pattern in scalar_patterns.items():
            value = _extract_scalar(pattern, block)
            if value is not None:
                step[key] = value
        if step.get("total_step_time_s") and step.get("weight_sync_s"):
            step["weight_sync_pct"] = 100.0 * step["weight_sync_s"] / step["total_step_time_s"]
        steps.append(step)
    return steps


def _audit_train_data(run_dir: Path) -> dict[str, Any]:
    paths = sorted((run_dir / "logs" / "grpo-polar").glob("exp_*/train_data_step*.jsonl"))
    rewards: list[float] = []
    advantages: list[float] = []
    gen_lps: list[float] = []
    prev_lps: list[float] = []
    deltas: list[float] = []
    ratios: list[float] = []
    clipped_low = 0
    clipped_high = 0
    mask_tokens = 0
    total_tokens = 0
    sampling_support_positions = 0
    sampling_support_train_positions = 0
    sampled_tokens_in_support = 0
    rows = 0
    has_fields = {
        "token_ids": False,
        "token_loss_mask": False,
        "generation_logprobs": False,
        "prev_logprobs": False,
        "advantages": False,
        "rewards": False,
        "sampling_support_token_ids": False,
    }
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            for key in has_fields:
                has_fields[key] = has_fields[key] or key in row
            reward_values = [float(x) for x in _flatten_one(row.get("rewards"))]
            rewards.extend(reward_values)
            adv_values = _flatten_one(row.get("advantages"))
            for item in adv_values:
                if isinstance(item, list):
                    advantages.extend(float(x) for x in item)
                else:
                    advantages.append(float(item))
            token_ids = _flatten_one(row.get("token_ids"))
            loss_mask = [int(x) for x in _flatten_one(row.get("token_loss_mask"))]
            gen = [float(x) for x in _flatten_one(row.get("generation_logprobs"))]
            prev = [float(x) for x in _flatten_one(row.get("prev_logprobs"))]
            sampling_support = _flatten_one(row.get("sampling_support_token_ids"))
            total_tokens += len(token_ids)
            sampling_support_positions += len(sampling_support)
            for i, token_id in enumerate(token_ids):
                if i >= len(loss_mask) or loss_mask[i] != 1 or i >= len(sampling_support):
                    continue
                support_ids = sampling_support[i]
                if not isinstance(support_ids, list) or not support_ids:
                    continue
                sampling_support_train_positions += 1
                sampled_tokens_in_support += int(int(token_id) in {int(x) for x in support_ids})
            limit = min(len(loss_mask), len(gen), len(prev))
            for i in range(limit):
                if loss_mask[i] != 1:
                    continue
                if not math.isfinite(gen[i]) or not math.isfinite(prev[i]):
                    continue
                mask_tokens += 1
                gen_lps.append(gen[i])
                prev_lps.append(prev[i])
                delta = prev[i] - gen[i]
                deltas.append(delta)
                ratio = math.exp(max(min(delta, 20.0), -20.0))
                ratios.append(ratio)
                clipped_low += int(ratio < 0.8)
                clipped_high += int(ratio > 1.2)
    clip_count = clipped_low + clipped_high
    return {
        "exists": bool(paths),
        "paths": [str(path) for path in paths],
        "rows": rows,
        "total_tokens": total_tokens,
        "masked_train_tokens": mask_tokens,
        "sampling_support_positions": sampling_support_positions,
        "sampling_support_train_positions": sampling_support_train_positions,
        "sampling_support_train_coverage": (
            sampling_support_train_positions / mask_tokens if mask_tokens else None
        ),
        "sampled_tokens_in_support": sampled_tokens_in_support,
        "sampled_token_support_fraction": (
            sampled_tokens_in_support / sampling_support_train_positions
            if sampling_support_train_positions
            else None
        ),
        **{f"has_{key}": value for key, value in has_fields.items()},
        "reward_summary": _float_stats(rewards),
        "advantage_summary": _float_stats(advantages),
        "rollout_generation_logprob_summary": _float_stats(gen_lps),
        "trainer_prev_logprob_summary": _float_stats(prev_lps),
        "logprob_delta_prev_minus_generation": _float_stats(deltas),
        "importance_ratio_prev_over_generation": _float_stats(ratios),
        "importance_ratio_clip_fraction_0p8_1p2": (
            clip_count / len(ratios) if ratios else None
        ),
        "importance_ratio_clip_low_fraction": clipped_low / len(ratios) if ratios else None,
        "importance_ratio_clip_high_fraction": clipped_high / len(ratios) if ratios else None,
    }


def _read_exit_code(run_dir: Path) -> int | None:
    text = (run_dir / "logs" / "exit-code.log").read_text() if (run_dir / "logs" / "exit-code.log").exists() else ""
    match = re.search(r"docker_exec_exit_code=(\d+)", text)
    return int(match.group(1)) if match else None


def audit(
    run_dir: Path,
    *,
    require_learning_signal: bool = False,
    require_sampling_distribution_replay: bool = False,
    max_near_zero_group_fraction: float = 0.5,
) -> dict[str, Any]:
    log_path = run_dir / "logs" / "grpo-nemo-polar.log"
    log_text = _clean(log_path.read_text(errors="ignore")) if log_path.exists() else ""
    config = _safe_load_json(run_dir / "config.json") or {}
    train_data = _audit_train_data(run_dir)
    steps = _extract_steps(log_text)
    run_timing = _safe_load_json(run_dir / "run_timing.json")
    group_reward_stds = [
        float(value)
        for value in re.findall(
            r"Polar collector adding group[^\n]*reward_std=([0-9.eE+-]+)",
            log_text,
        )
    ]
    near_zero_groups = sum(value < 1e-5 for value in group_reward_stds)
    near_zero_group_fraction = (
        near_zero_groups / len(group_reward_stds) if group_reward_stds else None
    )
    generation = config.get("generation") if isinstance(config, dict) else {}
    top_p = (
        float(generation.get("top_p", 1.0))
        if isinstance(generation, dict)
        else 1.0
    )
    sampling_distribution_replay_needed = top_p < 1.0
    sampling_distribution_replay_valid = (
        not sampling_distribution_replay_needed
        or (
            train_data["has_sampling_support_token_ids"]
            and train_data["sampling_support_train_coverage"] == 1.0
            and train_data["sampled_token_support_fraction"] == 1.0
        )
    )
    evidence = {
        "fp8_vllm_precision": "'precision': 'fp8'" in log_text or '"precision": "fp8"' in log_text,
        "fp8_kv_cache": "'kv_cache_dtype': 'fp8'" in log_text or '"kv_cache_dtype": "fp8"' in log_text,
        "bf16_policy": "'precision': 'bfloat16'" in log_text or '"precision": "bfloat16"' in log_text,
        "polar_rollout": "Using Polar external rollout collector" in log_text,
        "polar_group_adds": len(re.findall(r"Polar collector adding group", log_text)),
        "tokens_logprobs_masks": (
            train_data["has_token_ids"]
            and train_data["has_token_loss_mask"]
            and train_data["has_generation_logprobs"]
        ),
        "trainer_prev_logprobs": train_data["has_prev_logprobs"],
        "nonzero_advantages": bool(
            (train_data["advantage_summary"].get("max") or 0) != 0
            or (train_data["advantage_summary"].get("min") or 0) != 0
        ),
        "reward_variance_gate": bool(
            group_reward_stds
            and near_zero_group_fraction is not None
            and near_zero_group_fraction <= max_near_zero_group_fraction
        ),
        "reward_grouping": "reward_std=" in log_text,
        "replay_add": "ReplayBuffer.add: Adding trajectory" in log_text,
        "replay_sample": "Sampled " in log_text and "trajectory groups from buffer" in log_text,
        "policy_update": "▶ Training policy" in log_text,
        "weight_sync": "Synchronizing policy weights to trajectory collector" in log_text,
        "async_grpo_complete": "Async GRPO training complete" in log_text,
        "sampling_distribution_replay": sampling_distribution_replay_valid,
    }
    docker_exit_code = _read_exit_code(run_dir)
    required_evidence = [
        "polar_rollout",
        "tokens_logprobs_masks",
        "trainer_prev_logprobs",
        "replay_add",
        "replay_sample",
        "policy_update",
        "weight_sync",
        "async_grpo_complete",
    ]
    vllm_runtime = generation.get("vllm_runtime") if isinstance(generation, dict) else {}
    if isinstance(vllm_runtime, dict) and str(vllm_runtime.get("precision", "")).casefold() == "fp8":
        required_evidence.append("fp8_vllm_precision")
    if require_learning_signal:
        required_evidence.extend(["reward_variance_gate", "nonzero_advantages"])
    if require_sampling_distribution_replay:
        required_evidence.append("sampling_distribution_replay")
    status = "passed" if docker_exit_code == 0 and all(
        bool(evidence[key]) for key in required_evidence
    ) else "failed"
    summary = {
        "status": status,
        "run_dir": str(run_dir),
        "docker_exec_exit_code": docker_exit_code,
        "config": config,
        "evidence": evidence,
        "required_evidence": required_evidence,
        "validation_mode": "learning" if require_learning_signal else "infrastructure",
        "reward_variance": {
            "group_count": len(group_reward_stds),
            "near_zero_group_count": near_zero_groups,
            "near_zero_group_fraction": near_zero_group_fraction,
            "max_near_zero_group_fraction": max_near_zero_group_fraction,
        },
        "sampling_distribution": {
            "top_p": top_p,
            "rollout_support_replay_needed": sampling_distribution_replay_needed,
            "contract": (
                "sampling_support_token_ids must be token-aligned, cover every "
                "trainable token, and contain the sampled token"
            ),
        },
        "steps": steps,
        "train_data_audit": train_data,
    }
    if run_timing:
        summary["run_timing"] = run_timing
    if steps:
        summary["last_step"] = steps[-1]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-learning-signal", action="store_true")
    parser.add_argument("--require-sampling-distribution-replay", action="store_true")
    parser.add_argument("--max-near-zero-group-fraction", type=float, default=0.5)
    args = parser.parse_args()
    summary = audit(
        args.run_dir,
        require_learning_signal=args.require_learning_signal,
        require_sampling_distribution_replay=args.require_sampling_distribution_replay,
        max_near_zero_group_fraction=args.max_near_zero_group_fraction,
    )
    run_dir = args.run_dir
    (run_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    metrics_path = run_dir / "metrics.jsonl"
    with metrics_path.open("w") as fh:
        if summary.get("run_timing"):
            fh.write(json.dumps(summary["run_timing"], sort_keys=True) + "\n")
        for step in summary["steps"]:
            fh.write(json.dumps(step, sort_keys=True) + "\n")
        fh.write(
            json.dumps(
                {"event": "train_data_audit", **summary["train_data_audit"]},
                sort_keys=True,
            )
            + "\n"
        )
        fh.write(
            json.dumps(
                {
                    "event": "run_status",
                    "status": summary["status"],
                    "docker_exec_exit_code": summary["docker_exec_exit_code"],
                    "evidence": summary["evidence"],
                },
                sort_keys=True,
            )
            + "\n"
        )
    print(json.dumps({"status": summary["status"], "run_dir": str(run_dir)}, sort_keys=True))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
