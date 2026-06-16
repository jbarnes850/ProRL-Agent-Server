"""CLI end-to-end: spec -> compile -> register lineage -> emit launch (no GPU).

The CLI is emit-only: it must register a lineage record and write a runnable
launch.sh + plan.json, and never execute the smoke launcher.
"""

from __future__ import annotations

import json
from pathlib import Path

from nemo_polar_bridge.experiment.cli import main, run_launch

REPO_ROOT = Path(__file__).resolve().parents[3]
PROVEN_RUN_YAML = (
    REPO_ROOT / "examples" / "experiments" / "qwen3-1p7b-basic_arith-grpo-8x4.yaml"
)


def test_cli_launch_registers_lineage_and_emits_plan(tmp_path):
    store = tmp_path / "lineage"
    out = tmp_path / "out"
    main(["launch", str(PROVEN_RUN_YAML), "--store", str(store), "--out", str(out)])

    # lineage registered
    assert (store / "qwen3-1p7b-basic_arith-grpo-8x4.json").exists()

    # plan emitted (not executed)
    plan = json.loads((out / "plan.json").read_text())
    assert plan["env"]["NEMO_GRPO_NUM_PROMPTS_PER_STEP"] == "8"
    assert plan["extra_overrides"] == []
    assert plan["base_config"].endswith("grpo-qwen3-0.6b-1n8g-sglang.yaml")
    assert "lineage" in plan and len(plan["lineage"]["compiled_digest"]) == 64

    launch_sh = (out / "launch.sh").read_text()
    assert launch_sh.startswith("#!/usr/bin/env bash")
    assert "bash scripts/smoke/run_nemo_polar_external_collector_spark_smoke.sh" in launch_sh


def test_cli_returns_plan_without_executing(tmp_path):
    plan, record = run_launch(str(PROVEN_RUN_YAML), store_dir=str(tmp_path / "l"))
    # the launcher is emitted as argv, never invoked
    assert plan.argv()[0] == "bash"
    assert record.spec_id == "qwen3-1p7b-basic_arith-grpo-8x4"
