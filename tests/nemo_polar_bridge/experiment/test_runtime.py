"""Runtime profile + launch composer.

The composer reuses the proven smoke launcher (the skill's "pass overrides
through existing config surfaces" rule): spec experiment knobs map to the smoke
env-var contract; the algorithm-axis keys the smoke does not know become
positional EXTRA_OVERRIDES. The GRPO baseline must compose to env-only with no
positional extras (the smoke reproduces it exactly).
"""

from __future__ import annotations

from pathlib import Path

from nemo_polar_bridge.experiment import ExperimentSpec
from nemo_polar_bridge.experiment.runtime import RuntimeProfile, compose_launch

REPO_ROOT = Path(__file__).resolve().parents[3]
PROVEN_RUN_YAML = (
    REPO_ROOT / "examples" / "experiments" / "qwen3-1p7b-basic_arith-grpo-8x4.yaml"
)


def grpo_spec() -> ExperimentSpec:
    return ExperimentSpec.from_yaml(PROVEN_RUN_YAML)


def ablation(name: str) -> ExperimentSpec:
    spec = ExperimentSpec.from_yaml(PROVEN_RUN_YAML)
    spec.id = f"qwen3-1p7b-basic_arith-{name}-8x4"
    spec.parent = "qwen3-1p7b-basic_arith-grpo-8x4"
    spec.algorithm.name = name
    return spec


def test_grpo_baseline_composes_to_env_only_no_positional_extras():
    plan = compose_launch(grpo_spec(), RuntimeProfile.two_spark())
    assert plan.env["NEMO_GRPO_NUM_PROMPTS_PER_STEP"] == "8"
    assert plan.env["NEMO_GRPO_NUM_GENERATIONS_PER_PROMPT"] == "4"
    assert plan.env["NEMO_GRPO_MAX_NUM_STEPS"] == "2"
    assert plan.env["NEMO_GRPO_MAX_TRAJECTORY_AGE_STEPS"] == "1"
    assert plan.env["POLAR_MODEL_NAME"] == "Qwen/Qwen3-1.7B"
    assert plan.env["POLAR_MODEL_MAX_TOTAL_SEQUENCE_LENGTH"] == "4096"
    assert plan.env["NEMO_VLLM_PRECISION"] == "bfloat16"
    assert plan.env["NEMO_VLLM_KV_CACHE_DTYPE"] == "auto"
    assert plan.env["POLAR_DATASET_FAMILY"] == "nemo_gym"
    assert plan.env["POLAR_VERIFIER_TYPE"] == "exact_answer"
    assert plan.env["POLAR_EXECUTION_TYPE"] == "single_turn_chat"
    assert plan.env["POLAR_SOURCE_DATASETS"] == "basic_arithmetic"
    # the smoke reproduces the GRPO baseline exactly -> no positional overrides
    assert plan.extra_overrides == []
    assert plan.argv()[:3] == ["bash", plan.smoke_script, "qwen3-1p7b-basic_arith-grpo-8x4"]


def test_model_host_and_cont_built_from_profile_and_spec():
    plan = compose_launch(grpo_spec(), RuntimeProfile.two_spark())
    snap = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert plan.env["MODEL_HOST"] == (
        f"/home/jarrodbarnes/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/{snap}"
    )
    assert plan.env["MODEL_CONT"] == (
        f"/host-hf/hub/models--Qwen--Qwen3-1.7B/snapshots/{snap}"
    )


def test_cispo_ablation_emits_algorithm_axis_as_positional_overrides():
    plan = compose_launch(ablation("cispo"), RuntimeProfile.two_spark())
    assert "loss_fn.use_cispo=true" in plan.extra_overrides
    assert "loss_fn.token_level_loss=true" in plan.extra_overrides
    assert "loss_fn.ratio_clip_min=1.0" in plan.extra_overrides
    assert "loss_fn.ratio_clip_max=4.0" in plan.extra_overrides
    # the async + rollout env contract is unchanged by the algorithm swap
    assert plan.env["NEMO_GRPO_NUM_PROMPTS_PER_STEP"] == "8"
    # algorithm keys live only as positional overrides, never in env
    assert not any("cispo" in v.lower() for v in plan.env.values())


def test_qwen3_4b_fp8_rollout_spec_maps_runtime_env():
    spec_path = (
        REPO_ROOT
        / "examples"
        / "experiments"
        / "qwen3-4b-instruct-basic_arith-grpo-fp8-rollout-8x4.yaml"
    )
    plan = compose_launch(ExperimentSpec.from_yaml(spec_path), RuntimeProfile.two_spark())

    assert plan.env["POLAR_MODEL_NAME"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert plan.env["NEMO_VLLM_PRECISION"] == "fp8"
    assert plan.env["NEMO_VLLM_KV_CACHE_DTYPE"] == "auto"
    # 0.5, not 0.9: DGX Spark's unified memory (CPU+GPU share ~119GB per host)
    # OOMs at 0.9 with Ray/Polar/host overhead sharing the box (live-verified
    # this session; see the spec file's own comment for the incident).
    assert plan.env["NEMO_VLLM_GPU_MEMORY_UTILIZATION"] == "0.5"
    assert plan.env["NEMO_VLLM_ENFORCE_EAGER"] == "false"
    assert plan.env["NEMO_VLLM_MAX_NUM_BATCHED_TOKENS"] == "16384"
    assert plan.env["NEMO_VLLM_MAX_NUM_SEQS"] == "256"
    assert plan.env["POLAR_DATASET_LOCAL_JSONL"] == (
        "examples/datasets/basic_arithmetic_curated.jsonl"
    )
    assert plan.env["MODEL_HOST"].endswith(
        "models--Qwen--Qwen3-4B-Instruct-2507/snapshots/"
        "cdbee75f17c01a7cc42f958dc650907174af0554"
    )
    assert plan.extra_overrides == []


def test_drgrpo_ablation_emits_single_extra_override():
    plan = compose_launch(ablation("drgrpo"), RuntimeProfile.two_spark())
    assert plan.extra_overrides == ["grpo.normalize_rewards=false"]


def test_in_flight_weight_updates_false_reaches_the_launch_command():
    # The smoke script hardcodes in_flight_weight_updates=true with no env-var
    # hook; this must reach the launch command as a positional override
    # (appended after the smoke script's own hardcoded Hydra args) or the
    # spec's value silently has no effect at GPU-launch time.
    spec = grpo_spec()
    spec.async_grpo.in_flight_weight_updates = False
    plan = compose_launch(spec, RuntimeProfile.two_spark())
    assert plan.extra_overrides == ["grpo.async_grpo.in_flight_weight_updates=false"]


def test_recompute_kv_cache_true_reaches_the_launch_command():
    spec = grpo_spec()
    spec.async_grpo.recompute_kv_cache = True
    plan = compose_launch(spec, RuntimeProfile.two_spark())
    assert plan.extra_overrides == [
        "grpo.async_grpo.recompute_kv_cache_after_weight_updates=true"
    ]


def test_shell_rendering_is_runnable_and_self_describing():
    plan = compose_launch(ablation("cispo"), RuntimeProfile.two_spark())
    shell = plan.shell()
    assert "NEMO_GRPO_NUM_PROMPTS_PER_STEP=8" in shell
    assert f"bash {plan.smoke_script}" in shell
    assert "loss_fn.use_cispo=true" in shell
