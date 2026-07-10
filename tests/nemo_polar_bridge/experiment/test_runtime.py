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
QWEN35_PROFILE = (
    REPO_ROOT / "examples" / "experiments" / "profiles" / "two_spark_qwen35_9b_safe.yaml"
)
QWEN35_TOP_P_SPEC = (
    REPO_ROOT
    / "examples"
    / "experiments"
    / "qwen3p5-9b-reasoning_gym-cispo-fp8-top-p095-replay-smoke.yaml"
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


def test_qwen35_profile_pins_live_roce_hca_and_gid() -> None:
    profile = RuntimeProfile.from_yaml(QWEN35_PROFILE)

    assert profile.extra_env["NCCL_TRANSPORT"] == "roce"
    assert profile.extra_env["NCCL_SOCKET_IFNAME"] == "enp1s0f0np0"
    assert profile.extra_env["NCCL_IB_HCA"] == "rocep1s0f0"
    assert profile.extra_env["NCCL_IB_GID_INDEX"] == "3"
    assert profile.extra_env["NRL_REFIT_SKIP_OPT_OFFLOAD"] == "1"
    assert profile.extra_env["NRL_FP8_BROADCAST"] == "0"
    assert profile.extra_env["NRL_VALIDATION_MODE"] == "infrastructure"


def test_qwen35_top_p_replay_spec_composes_fail_closed_runtime_contract() -> None:
    spec = ExperimentSpec.from_yaml(QWEN35_TOP_P_SPEC)
    plan = compose_launch(spec, RuntimeProfile.from_yaml(QWEN35_PROFILE))

    assert plan.env["POLAR_MODEL_TOP_P"] == "0.95"
    assert plan.env["NRL_TOP_P_SUPPORT_REPLAY"] == "1"
    assert plan.env["NRL_TOP_P_SUPPORT_SIZE"] == "256"
    assert plan.env["NRL_ALLOW_NAIVE_TOP_P"] == "0"


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
    # The smoke still receives the compatibility env vars above, while these
    # authoritative copies reach vLLM's constructor through vllm_kwargs.
    assert plan.extra_overrides == [
        "++policy.generation.vllm_kwargs.max_num_seqs=256",
        "++policy.generation.vllm_kwargs.max_num_batched_tokens=16384",
    ]


def test_long_context_text_only_runtime_reaches_launch_command():
    spec = grpo_spec()
    spec.model.max_total_sequence_length = 4096
    spec.vllm_runtime.max_model_len = 131072
    spec.vllm_runtime.max_num_seqs = 8
    spec.vllm_runtime.max_num_batched_tokens = 65536
    spec.vllm_runtime.enable_prefix_caching = True
    spec.vllm_runtime.enable_chunked_prefill = True
    spec.vllm_runtime.mamba_cache_mode = "align"
    spec.vllm_runtime.language_model_only = True
    spec.vllm_runtime.quantization_ignored_layer_kws = [
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
    ]

    plan = compose_launch(spec, RuntimeProfile.two_spark())
    assert "++policy.generation.vllm_cfg.max_model_len=131072" in plan.extra_overrides
    assert "++policy.generation.vllm_kwargs.max_num_seqs=8" in plan.extra_overrides
    assert (
        "++policy.generation.vllm_kwargs.max_num_batched_tokens=65536"
        in plan.extra_overrides
    )
    assert "++policy.generation.vllm_cfg.enable_prefix_caching=true" in plan.extra_overrides
    assert "++policy.generation.vllm_kwargs.enable_chunked_prefill=true" in plan.extra_overrides
    assert "++policy.generation.vllm_kwargs.mamba_cache_mode=align" in plan.extra_overrides
    assert "++policy.generation.vllm_kwargs.language_model_only=true" in plan.extra_overrides
    assert (
        "++policy.generation.vllm_cfg.quantization_ignored_layer_kws="
        '["linear_attn.in_proj_a","linear_attn.in_proj_b"]'
        in plan.extra_overrides
    )


def test_single_spark_lora_trainer_fit_reaches_launch_command():
    spec = grpo_spec()
    spec.trainer.activation_checkpointing = True
    spec.trainer.freeze_vision_tower = True
    spec.trainer.freeze_audio_tower = True
    spec.trainer.lora.enabled = True
    spec.trainer.lora.dim = 32
    spec.trainer.lora.alpha = 32
    spec.trainer.lora.target_modules = ["model.language_model.*proj*"]

    plan = compose_launch(spec, RuntimeProfile.two_spark())
    assert "policy.dtensor_cfg.activation_checkpointing=true" in plan.extra_overrides
    assert (
        "++policy.dtensor_cfg.automodel_kwargs.freeze_config."
        "freeze_vision_tower=true" in plan.extra_overrides
    )
    assert (
        "++policy.dtensor_cfg.automodel_kwargs.freeze_config."
        "freeze_audio_tower=true" in plan.extra_overrides
    )
    assert "policy.dtensor_cfg.lora_cfg.enabled=true" in plan.extra_overrides
    assert "policy.dtensor_cfg.lora_cfg.dim=32" in plan.extra_overrides
    assert "policy.dtensor_cfg.lora_cfg.alpha=32" in plan.extra_overrides
    assert (
        "policy.dtensor_cfg.lora_cfg.target_modules="
        '["model.language_model.*proj*"]' in plan.extra_overrides
    )


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
