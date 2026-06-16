"""Experiment-as-code CLI: spec -> compile -> register lineage -> emit launch.

Emit-only by design: it never launches GPU work. It registers a content-addressed
lineage record and writes a runnable ``launch.sh`` (an env-prefixed invocation of
the proven smoke launcher) plus a ``plan.json``. Run the emitted ``launch.sh`` on
the Spark to actually execute, gated by the staged-validation ladder.

    python -m nemo_polar_bridge.experiment.cli launch <spec.yaml> \\
        --profile two_spark --store runs/lineage --out runs/<id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .compile import compile_spec
from .lineage import LineageRecord, register_experiment
from .runtime import LaunchPlan, RuntimeProfile, compose_launch
from .spec import ExperimentSpec


def _load_profile(value: Optional[str]) -> RuntimeProfile:
    if value in (None, "two_spark"):
        return RuntimeProfile.two_spark()
    return RuntimeProfile.from_yaml(value)


def run_launch(
    spec_path: str,
    *,
    profile: Optional[RuntimeProfile] = None,
    store_dir: Optional[str] = None,
    out_dir: Optional[str] = None,
    slug: Optional[str] = None,
    matrix_cell: str = "",
) -> tuple[LaunchPlan, Optional[LineageRecord]]:
    spec = ExperimentSpec.from_yaml(spec_path)
    profile = profile or RuntimeProfile.two_spark()
    compiled = compile_spec(spec)
    record = (
        register_experiment(spec, store_dir, compiled=compiled) if store_dir else None
    )
    plan = compose_launch(spec, profile, slug=slug, matrix_cell=matrix_cell)

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        launch_sh = out_path / "launch.sh"
        launch_sh.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n" + plan.shell() + "\n"
        )
        launch_sh.chmod(0o755)
        plan_json: dict = {
            "spec_id": plan.spec_id,
            "slug": plan.slug,
            "smoke_script": plan.smoke_script,
            "base_config": compiled.base_config,
            "env": plan.env,
            "extra_overrides": plan.extra_overrides,
        }
        if record is not None:
            plan_json["lineage"] = {
                "spec_digest": record.spec_digest,
                "compiled_digest": record.compiled_digest,
                "parent": record.parent,
            }
        (out_path / "plan.json").write_text(
            json.dumps(plan_json, indent=2, sort_keys=True) + "\n"
        )

    return plan, record


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="experiment", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    launch = sub.add_parser(
        "launch", help="compile + register lineage + emit launch command (no GPU)"
    )
    launch.add_argument("spec", help="path to an experiment spec YAML")
    launch.add_argument(
        "--profile", default="two_spark", help="'two_spark' or a profile YAML path"
    )
    launch.add_argument("--store", default=None, help="lineage store directory")
    launch.add_argument("--out", default=None, help="emit launch.sh + plan.json here")
    launch.add_argument("--slug", default=None, help="run slug (defaults to spec id)")
    launch.add_argument("--matrix-cell", default="", help="run-matrix cell label")

    args = parser.parse_args(argv)

    if args.cmd == "launch":
        plan, record = run_launch(
            args.spec,
            profile=_load_profile(args.profile),
            store_dir=args.store,
            out_dir=args.out,
            slug=args.slug,
            matrix_cell=args.matrix_cell,
        )
        print(f"# experiment: {plan.spec_id}")
        if record is not None:
            print(
                f"# lineage: spec={record.spec_digest[:12]} "
                f"compiled={record.compiled_digest[:12]} parent={record.parent}"
            )
        print(f"# experiment overrides beyond the smoke baseline: {len(plan.extra_overrides)}")
        print(plan.shell())


if __name__ == "__main__":
    main()
