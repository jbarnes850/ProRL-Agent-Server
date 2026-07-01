"""Experiment-as-code spec -> NeMo override compiler.

These tests pin the compiler against the known-good two-Spark Async GRPO run
documented in the lab skill (Qwen3-1.7B, nemo_gym/basic_arithmetic,
exact_answer/single_turn_chat, 8x4x2). The expected experiment-level overrides
are taken verbatim from the proven smoke launch
(scripts/smoke/run_nemo_polar_external_collector_spark_smoke.sh:464-523).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_polar_bridge.experiment import ExperimentSpec, SpecCompileError, compile_spec

REPO_ROOT = Path(__file__).resolve().parents[3]
PROVEN_RUN_YAML = (
    REPO_ROOT / "examples" / "experiments" / "qwen3-1p7b-basic_arith-grpo-8x4.yaml"
)

SGLANG_BASE = "examples/configs/recipes/llm/grpo-qwen3-0.6b-1n8g-sglang.yaml"
QWEN3_1P7B_NEMO_PATH = (
    "/host-hf/hub/models--Qwen--Qwen3-1.7B/snapshots/"
    "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
)


def proven_run_spec() -> ExperimentSpec:
    """The re-expressed proven Qwen3-1.7B basic_arithmetic GRPO run."""
    return ExperimentSpec(
        id="qwen3-1p7b-basic_arith-grpo-8x4",
        model={
            "name": "Qwen/Qwen3-1.7B",
            "snapshot": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "nemo_path": QWEN3_1P7B_NEMO_PATH,
            "max_total_sequence_length": 4096,
        },
        dataset={
            "family": "nemo_gym",
            "source": "nvidia/Nemotron-RL-ReasoningGym-v1",
            "subset": "basic_arithmetic",
        },
        verifier={"type": "exact_answer", "execution": "single_turn_chat"},
        algorithm={"name": "grpo"},
        async_grpo={"lag": 1},
        rollout={"prompts_per_step": 8, "generations_per_prompt": 4, "max_steps": 2},
    )


def test_proven_grpo_run_compiles_to_known_good_overrides():
    compiled = compile_spec(proven_run_spec())
    assert compiled.base_config == SGLANG_BASE
    ov = compiled.nemo_overrides_dict()

    # async / two-Spark disaggregation
    assert ov["grpo.async_grpo.enabled"] is True
    assert ov["grpo.async_grpo.max_trajectory_age_steps"] == 1
    # in_flight_weight_updates/recompute_kv_cache_after_weight_updates are
    # emitted only when they diverge from the smoke script's own hardcoded
    # defaults (true/false); the baseline spec uses those defaults, so
    # neither key appears here (see compile.py and runtime.py's
    # SMOKE_COVERED_KEYS comment for why).
    assert "grpo.async_grpo.in_flight_weight_updates" not in ov
    assert "grpo.async_grpo.recompute_kv_cache_after_weight_updates" not in ov
    assert ov["policy.generation.colocated.enabled"] is False
    assert ov["policy.generation.colocated.resources.gpus_per_node"] == 1
    assert ov["policy.generation.colocated.resources.num_nodes"] == 1
    assert ov["cluster.num_nodes"] == 2

    # rollout shape
    assert ov["grpo.num_prompts_per_step"] == 8
    assert ov["grpo.num_generations_per_prompt"] == 4
    assert ov["grpo.max_num_steps"] == 2
    assert ov["grpo.max_num_epochs"] == 1

    # GRPO baseline objective: async needs IS correction, RLVR drops KL penalty
    assert ov["loss_fn.use_importance_sampling_correction"] is True
    assert ov["loss_fn.reference_policy_kl_penalty"] == 0.0
    assert ov["grpo.use_dynamic_sampling"] is False
    assert ov["grpo.reward_shaping.enabled"] is False
    # baseline must NOT touch the beyond-GRPO loss flags
    assert "loss_fn.use_cispo" not in ov
    assert "grpo.adv_estimator.name" not in ov

    # model + generation
    assert ov["policy.model_name"] == QWEN3_1P7B_NEMO_PATH
    assert ov["policy.tokenizer.name"] == QWEN3_1P7B_NEMO_PATH
    assert ov["policy.max_total_sequence_length"] == 4096
    assert ov["policy.generation.backend"] == "vllm"
    assert ov["policy.generation.vllm_cfg.async_engine"] is True
    assert ov["policy.generation.vllm_cfg.precision"] == "bfloat16"
    assert ov["policy.generation.vllm_cfg.kv_cache_dtype"] == "auto"
    assert ov["policy.generation.vllm_cfg.max_model_len"] == 4096

    # verifier -> processor/env mapping
    assert ov["data.default.processor"] == "math_hf_data_processor"
    assert ov["data.default.env_name"] == "math"
    assert ov["data_plane.enabled"] is False


def test_dataset_family_rejects_unknown_value():
    # A typo'd dataset.family previously validated fine as a bare str and
    # would only surface as a silent fallthrough deep in prepare_dataset.py's
    # adapter dispatch. It must fail fast at spec-construction time instead.
    with pytest.raises(ValidationError, match="family"):
        ExperimentSpec(
            id="qwen3-1p7b-basic_arith-grpo-8x4",
            model={
                "name": "Qwen/Qwen3-1.7B",
                "snapshot": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                "nemo_path": QWEN3_1P7B_NEMO_PATH,
                "max_total_sequence_length": 4096,
            },
            dataset={"family": "reasoning_gm"},
            verifier={"type": "exact_answer", "execution": "single_turn_chat"},
            algorithm={"name": "grpo"},
            async_grpo={"lag": 1},
            rollout={"prompts_per_step": 8, "generations_per_prompt": 4, "max_steps": 2},
        )


def test_in_flight_weight_updates_false_emits_only_the_delta():
    spec = proven_run_spec()
    spec.async_grpo.in_flight_weight_updates = False
    ov = compile_spec(spec).nemo_overrides_dict()
    assert ov["grpo.async_grpo.in_flight_weight_updates"] is False
    # matches the smoke default, so still not declared
    assert "grpo.async_grpo.recompute_kv_cache_after_weight_updates" not in ov


def test_recompute_kv_cache_true_emits_only_the_delta():
    spec = proven_run_spec()
    spec.async_grpo.recompute_kv_cache = True
    ov = compile_spec(spec).nemo_overrides_dict()
    assert ov["grpo.async_grpo.recompute_kv_cache_after_weight_updates"] is True
    # matches the smoke default, so still not declared
    assert "grpo.async_grpo.in_flight_weight_updates" not in ov


def test_cispo_axis_sets_required_loss_flags():
    spec = proven_run_spec()
    spec.id = "qwen3-1p7b-basic_arith-cispo-8x4"
    spec.parent = "qwen3-1p7b-basic_arith-grpo-8x4"
    spec.algorithm.name = "cispo"  # the one-axis change vs the GRPO baseline
    ov = compile_spec(spec).nemo_overrides_dict()
    assert ov["loss_fn.use_cispo"] is True
    assert ov["loss_fn.token_level_loss"] is True
    assert ov["loss_fn.sequence_level_importance_ratios"] is False
    assert ov["loss_fn.use_importance_sampling_correction"] is True
    # Laguna parity: (c_low, c_high) = (1, 4) -> NeMo clamp [1-1, 1+4] = [0, 5]
    assert ov["loss_fn.ratio_clip_min"] == 1.0
    assert ov["loss_fn.ratio_clip_max"] == 4.0


def test_fp8_rollout_precision_sets_vllm_only_not_policy_precision():
    spec = proven_run_spec()
    spec.id = "qwen3-4b-instruct-basic_arith-grpo-fp8-rollout-8x4"
    spec.model.name = "Qwen/Qwen3-4B-Instruct-2507"
    spec.model.snapshot = "cdbee75f17c01a7cc42f958dc650907174af0554"
    spec.model.nemo_path = (
        "/host-hf/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/"
        "cdbee75f17c01a7cc42f958dc650907174af0554"
    )
    spec.precision.rollout = "fp8"
    spec.precision.kv_cache_dtype = "auto"
    ov = compile_spec(spec).nemo_overrides_dict()

    assert ov["policy.generation.vllm_cfg.precision"] == "fp8"
    assert ov["policy.generation.vllm_cfg.kv_cache_dtype"] == "auto"
    assert "policy.precision" not in ov


def test_drgrpo_objective_disables_std_norm_only():
    """Dr. GRPO = GRPO baseline minus group-std normalization (one-key delta).

    Verified against NeMo c236061b: advantage_estimator.py:72-78 (the only
    std-division guard), loss_functions.py:589-604 (TOKEN_LEVEL flat token
    mean). The leave-one-out mean baseline must stay on (base default); the
    compiler must never emit it false.
    """
    spec = proven_run_spec()
    spec.id = "qwen3-1p7b-basic_arith-drgrpo-8x4"
    spec.parent = "qwen3-1p7b-basic_arith-grpo-8x4"
    spec.algorithm.name = "drgrpo"
    ov = compile_spec(spec).nemo_overrides_dict()
    assert ov["grpo.normalize_rewards"] is False
    # baseline retained: compiler must never override the LOO baseline off
    assert "grpo.adv_estimator.use_leave_one_out_baseline" not in ov
    assert "grpo.use_leave_one_out_baseline" not in ov
    # length-debiased: token_level stays the base default, not flipped to seq-level
    assert "loss_fn.sequence_level_importance_ratios" not in ov
    assert "loss_fn.token_level_loss" not in ov
    # pure normalization change: not a different objective or estimator
    assert "loss_fn.use_cispo" not in ov
    assert "grpo.adv_estimator.name" not in ov


def test_drgrpo_requires_grpo_advantage():
    spec = proven_run_spec()
    spec.algorithm.name = "drgrpo"
    spec.algorithm.advantage = "reinforce_plus_plus"
    with pytest.raises(SpecCompileError, match="drgrpo"):
        compile_spec(spec)


def test_async_plus_dapo_dynamic_sampling_is_rejected():
    spec = proven_run_spec()
    spec.algorithm.name = "dapo"
    spec.algorithm.dynamic_sampling = True
    with pytest.raises(SpecCompileError, match="async"):
        compile_spec(spec)


def test_gdpo_requires_multi_reward_verifier():
    spec = proven_run_spec()
    spec.algorithm.advantage = "gdpo"
    with pytest.raises(SpecCompileError, match="reward_components"):
        compile_spec(spec)


def test_proven_run_yaml_matches_inline_spec():
    """The committed YAML artifact compiles identically to the inline spec."""
    from_yaml = compile_spec(ExperimentSpec.from_yaml(PROVEN_RUN_YAML))
    inline = compile_spec(proven_run_spec())
    assert from_yaml.base_config == inline.base_config
    assert from_yaml.nemo_overrides_dict() == inline.nemo_overrides_dict()
