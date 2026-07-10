"""Lineage store: content-addressed experiment records + bidirectional DAG.

Forward (spec -> run -> artifacts) and backward (parent chain + digests) per the
experiment-as-code lineage requirement. Built on the spec + compiler; no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nemo_polar_bridge.experiment import ExperimentSpec
from nemo_polar_bridge.experiment.lineage import (
    LineageStore,
    compiled_digest,
    register_experiment,
    spec_digest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROVEN_RUN_YAML = (
    REPO_ROOT / "examples" / "experiments" / "qwen3-1p7b-basic_arith-grpo-8x4.yaml"
)
GRPO_ID = "qwen3-1p7b-basic_arith-grpo-8x4"
DRGRPO_ID = "qwen3-1p7b-basic_arith-drgrpo-8x4"


def grpo_spec() -> ExperimentSpec:
    return ExperimentSpec.from_yaml(PROVEN_RUN_YAML)


def drgrpo_child() -> ExperimentSpec:
    """One-axis change off the GRPO baseline: same base, different objective."""
    spec = ExperimentSpec.from_yaml(PROVEN_RUN_YAML)
    spec.id = DRGRPO_ID
    spec.parent = GRPO_ID
    spec.algorithm.name = "drgrpo"
    return spec


def test_register_writes_record_and_digests_are_deterministic(tmp_path):
    rec = register_experiment(grpo_spec(), tmp_path)
    assert (tmp_path / f"{GRPO_ID}.json").exists()
    assert rec.spec_id == GRPO_ID
    assert rec.parent is None
    assert rec.base_config.endswith("grpo_math_1B_sglang.yaml")
    # same spec -> same digest (canonical, order-independent)
    assert spec_digest(grpo_spec()) == spec_digest(grpo_spec())
    assert rec.spec_digest == spec_digest(grpo_spec())


def test_one_axis_change_flips_compiled_digest_but_keeps_base(tmp_path):
    g = register_experiment(grpo_spec(), tmp_path)
    d = register_experiment(drgrpo_child(), tmp_path)
    assert g.base_config == d.base_config  # same recipe
    assert g.compiled_digest != d.compiled_digest  # different config actually ran
    assert g.spec_digest != d.spec_digest
    assert compiled_digest(grpo_spec()) == g.compiled_digest


def test_bidirectional_lineage_traversal(tmp_path):
    register_experiment(grpo_spec(), tmp_path)
    register_experiment(drgrpo_child(), tmp_path)
    store = LineageStore(tmp_path)
    assert store.get(GRPO_ID).parent is None
    assert store.get(DRGRPO_ID).parent == GRPO_ID
    # backward
    assert store.ancestors(DRGRPO_ID) == [GRPO_ID]
    # forward
    assert store.children(GRPO_ID) == [DRGRPO_ID]
    assert DRGRPO_ID in store.descendants(GRPO_ID)


def test_store_round_trips_records_from_disk(tmp_path):
    rec = register_experiment(grpo_spec(), tmp_path)
    reloaded = LineageStore(tmp_path).get(GRPO_ID)
    assert reloaded.compiled_digest == rec.compiled_digest
    assert reloaded.matrix == rec.matrix
    assert reloaded.algorithm == rec.algorithm


def test_artifacts_bind_forward_by_content_hash(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text('{"slug": "x"}\n')
    rec = register_experiment(grpo_spec(), tmp_path, run_dir=run_dir, artifacts=["config.json"])
    assert len(rec.artifacts) == 1
    art = rec.artifacts[0]
    assert art["path"] == "config.json"
    assert len(art["sha256"]) == 64


def test_artifact_binding_rejects_path_escape(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "secret").write_text("nope\n")
    with pytest.raises(ValueError, match="escapes"):
        register_experiment(
            grpo_spec(), tmp_path, run_dir=run_dir, artifacts=["../secret"]
        )


def test_artifact_binding_missing_file_errors_clearly(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        register_experiment(
            grpo_spec(), tmp_path, run_dir=run_dir, artifacts=["nope.json"]
        )


def test_artifact_binding_rejects_absolute_path(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    secret = tmp_path / "secret"
    secret.write_text("nope\n")
    with pytest.raises(ValueError, match="escapes"):
        register_experiment(
            grpo_spec(), tmp_path, run_dir=run_dir, artifacts=[str(secret)]
        )


def test_compiled_digest_is_injective_under_newline_values():
    """A newline inside an override value must not collide with a boundary (D1)."""
    from nemo_polar_bridge.experiment.compile import CompiledExperiment

    base = "base.yaml"
    one_override_with_newline = CompiledExperiment(
        "a", base, [("policy.tokenizer.name", "a\nz=1")]
    )
    two_overrides = CompiledExperiment(
        "b", base, [("policy.tokenizer.name", "a"), ("z", 1)]
    )
    assert compiled_digest(one_override_with_newline) != compiled_digest(two_overrides)
