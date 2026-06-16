"""Validate a compiled experiment spec against NeMo RL's real loader (no GPU).

This is the no-GPU "dry-run" gate of the experiment-as-code execution layer. It
compiles a spec to ``(base_config, dotted overrides)`` and applies the overrides
to the base config through NeMo RL's own ``load_config`` + ``parse_hydra_overrides``
(``OmegaConf.set_struct(True)``), exactly as ``run_grpo.py`` does at startup
before any GPU work. A misspelled or non-existent key raises and fails the gate.

NeMo RL's ``nemo_rl/utils/config.py`` imports only ``hydra`` + ``omegaconf`` (no
torch), so it is loaded in isolation via importlib to avoid pulling the heavy
``nemo_rl`` package ``__init__``.

Usage:
    uv run --with omegaconf --with hydra-core \\
      python scripts/experiment/validate_against_nemo_loader.py \\
      examples/experiments/qwen3-1p7b-basic_arith-grpo-8x4.yaml

Env:
    NEMO_RL_ROOT  NeMo RL checkout (default: /tmp/nemo-rl-recon)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_nemo_config_module(nemo_root: Path):
    """Load nemo_rl/utils/config.py in isolation (hydra+omegaconf only)."""
    cfg_path = nemo_root / "nemo_rl" / "utils" / "config.py"
    if not cfg_path.exists():
        sys.exit(f"NeMo RL config loader not found: {cfg_path} (set NEMO_RL_ROOT)")
    spec = importlib.util.spec_from_file_location("_nemo_rl_config", cfg_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="path to an experiment spec YAML")
    parser.add_argument(
        "--nemo-root",
        default=os.environ.get("NEMO_RL_ROOT", "/tmp/nemo-rl-recon"),
        help="NeMo RL checkout providing the base configs and loader",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from nemo_polar_bridge.experiment import ExperimentSpec, compile_spec

    nemo_root = Path(args.nemo_root)
    nemo_cfg = _load_nemo_config_module(nemo_root)

    spec = ExperimentSpec.from_yaml(args.spec)
    compiled = compile_spec(spec)

    base_path = nemo_root / compiled.base_config
    if not base_path.exists():
        sys.exit(f"base config not found: {base_path}")

    nemo_cfg.register_omegaconf_resolvers()
    cfg = nemo_cfg.load_config(base_path)
    overrides = compiled.nemo_override_args()

    try:
        cfg = nemo_cfg.parse_hydra_overrides(cfg, overrides)
    except Exception as exc:  # noqa: BLE001 - surface the exact loader error
        print(f"FAIL: {compiled.spec_id}: NeMo loader rejected an override:\n  {exc}")
        sys.exit(1)

    print(
        f"OK: {compiled.spec_id} -> {len(overrides)} experiment overrides "
        f"applied against {compiled.base_config} (struct=True)"
    )
    print(f"  algorithm                : {spec.algorithm.name}/{spec.algorithm.advantage}")
    print(f"  grpo.async_grpo.enabled  : {cfg.grpo.async_grpo.enabled}")
    print(f"  ...max_trajectory_age    : {cfg.grpo.async_grpo.max_trajectory_age_steps}")
    print(f"  num_prompts_per_step     : {cfg.grpo.num_prompts_per_step}")
    print(f"  num_generations_per_prompt: {cfg.grpo.num_generations_per_prompt}")
    print(f"  loss_fn.use_is_correction: {cfg.loss_fn.use_importance_sampling_correction}")
    print(f"  policy.model_name        : {cfg.policy.model_name}")
    print(f"  policy.generation.backend: {cfg.policy.generation.backend}")
    print(f"  colocated.enabled        : {cfg.policy.generation.colocated.enabled}")


if __name__ == "__main__":
    main()
