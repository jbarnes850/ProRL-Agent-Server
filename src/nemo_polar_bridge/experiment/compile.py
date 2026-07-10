"""Compile an :class:`ExperimentSpec` into a NeMo base config + dotted overrides.

The compiler owns the experiment-level overrides only. Every key it emits
already exists in the merged base config (``grpo_math_1B_sglang.yaml`` ->
``grpo_math_1B.yaml``), so NeMo's loader applies them under ``set_struct(True)``
without an append (``+``) prefix. Grounding for each mapping lives in the
NeMo RL revision pinned in the lab skill:

- one ``ClippedPGLossFn`` for the whole PG family (loss_functions.py:161-205)
- async staleness via ``grpo.async_grpo.*`` (grpo.py:125-136)
- advantage estimator dispatch ``grpo.adv_estimator.name`` (grpo.py:1445)
- async excludes DAPO data-side features (run_grpo.py:155-160)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .spec import Algorithm, ExperimentSpec

# Base recipe the experiment delta is applied over.
# Shared by the proven c236 rollback and current NeMo main. The former
# Qwen3-specific recipe was removed by the SGLang rollout refactor in current
# main, while this base retains the same inheritance point on both revisions.
SGLANG_BASE = "examples/configs/grpo_math_1B_sglang.yaml"

# Laguna parity for CISPO: (c_low, c_high) = (1, 4) -> NeMo clamp [1-1, 1+4]=[0,5].
_CISPO_DEFAULT_CLIP = (1.0, 4.0)
# DAPO Clip-Higher reference (dapo recipe: ratio_clip_max=0.28).
_DAPO_DEFAULT_CLIP = (0.2, 0.28)

# CISPO uses NeMo's standard (unweighted) RLOO leave-one-out baseline, not
# Laguna's length-weighted variant (tech report eq. 2); exact parity deferred.

# verifier.type -> (data.default.processor, data.default.env_name)
_VERIFIER_DATA = {
    "exact_answer": ("math_hf_data_processor", "math"),
}


class SpecCompileError(ValueError):
    """Raised when a spec is structurally valid but expresses a combination NeMo
    RL cannot run (e.g. async + DAPO data-side features)."""


@dataclass
class CompiledExperiment:
    spec_id: str
    base_config: str
    nemo_overrides: list[tuple[str, object]]
    warnings: list[str] = field(default_factory=list)

    def nemo_overrides_dict(self) -> dict[str, object]:
        return {k: v for k, v in self.nemo_overrides}

    def nemo_override_args(self) -> list[str]:
        """Dotted ``key=value`` args in Hydra syntax for ``--config <base> ...``."""
        return [f"{k}={_hydra_value(v)}" for k, v in self.nemo_overrides]


def _hydra_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        # Compact JSON is valid Hydra list syntax and survives the smoke
        # launcher's shell boundary without whitespace splitting.
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def compile_spec(spec: ExperimentSpec) -> CompiledExperiment:
    _validate(spec)

    ov: list[tuple[str, object]] = []
    a = spec.algorithm
    ag = spec.async_grpo
    r = spec.rollout
    m = spec.model
    t = spec.topology
    trainer = spec.trainer

    # async / two-Spark disaggregation
    ov.append(("grpo.async_grpo.enabled", ag.enabled))
    ov.append(("grpo.async_grpo.max_trajectory_age_steps", ag.lag))
    # Emit only on divergence from the smoke script's hardcoded defaults
    # (in_flight=true / recompute=false); these route through positional
    # EXTRA_OVERRIDES, so emitting the delta keeps the baseline's
    # zero-positional-extras invariant.
    if ag.in_flight_weight_updates is not True:
        ov.append(("grpo.async_grpo.in_flight_weight_updates", ag.in_flight_weight_updates))
    if ag.recompute_kv_cache is not False:
        ov.append(
            ("grpo.async_grpo.recompute_kv_cache_after_weight_updates", ag.recompute_kv_cache)
        )

    # rollout shape
    ov.append(("grpo.num_prompts_per_step", r.prompts_per_step))
    ov.append(("grpo.num_generations_per_prompt", r.generations_per_prompt))
    ov.append(("grpo.max_num_steps", r.max_steps))
    ov.append(("grpo.max_num_epochs", r.max_epochs))

    # DAPO data-side flags (default off; only the dapo objective turns them on)
    ov.append(("grpo.use_dynamic_sampling", a.dynamic_sampling))
    ov.append(("grpo.reward_shaping.enabled", a.reward_shaping))

    # advantage estimator: emit only when it differs from the grpo default
    if a.advantage != "grpo":
        ov.append(("grpo.adv_estimator.name", a.advantage))

    # objective (loss_fn)
    ov.append(("loss_fn.reference_policy_kl_penalty", a.kl_penalty))
    if ag.enabled:
        # async is off-policy: NeMo requires the IS correction (run_grpo.py:155)
        ov.append(("loss_fn.use_importance_sampling_correction", True))
    _emit_objective(a, ov)

    # model
    ov.append(("policy.model_name", m.train_path))
    ov.append(("policy.tokenizer.name", m.train_path))
    ov.append(("policy.max_total_sequence_length", m.max_total_sequence_length))

    # Trainer fit. Qwen3.5-9B full fine-tuning cannot fit a single GB10 once
    # FP32 master weights, gradients, and Adam moments coexist. NeMo DTensor-v2
    # merges LoRA into the streamed base tensors at refit, so the rollout still
    # receives ordinary FP8 weights rather than an adapter-aware serving path.
    if trainer.activation_checkpointing:
        ov.append(("policy.dtensor_cfg.activation_checkpointing", True))
    if trainer.freeze_vision_tower is not None:
        ov.append(
            (
                "++policy.dtensor_cfg.automodel_kwargs.freeze_config.freeze_vision_tower",
                trainer.freeze_vision_tower,
            )
        )
    if trainer.freeze_audio_tower is not None:
        ov.append(
            (
                "++policy.dtensor_cfg.automodel_kwargs.freeze_config.freeze_audio_tower",
                trainer.freeze_audio_tower,
            )
        )
    if trainer.lora.enabled:
        lora = trainer.lora
        ov.extend(
            [
                ("policy.dtensor_cfg.lora_cfg.enabled", True),
                ("policy.dtensor_cfg.lora_cfg.dim", lora.dim),
                ("policy.dtensor_cfg.lora_cfg.alpha", lora.alpha),
                ("policy.dtensor_cfg.lora_cfg.target_modules", lora.target_modules),
                ("policy.dtensor_cfg.lora_cfg.exclude_modules", lora.exclude_modules),
                (
                    "policy.dtensor_cfg.lora_cfg.match_all_linear",
                    lora.match_all_linear,
                ),
                ("policy.dtensor_cfg.lora_cfg.dropout", lora.dropout),
                (
                    "policy.dtensor_cfg.lora_cfg.dropout_position",
                    lora.dropout_position,
                ),
                ("policy.dtensor_cfg.lora_cfg.lora_A_init", lora.lora_A_init),
                ("policy.dtensor_cfg.lora_cfg.use_triton", lora.use_triton),
            ]
        )

    # generation / topology shape
    ov.append(("policy.generation.backend", spec.precision.rollout_backend))
    ov.append(("policy.generation.vllm_cfg.async_engine", True))
    ov.append(("policy.generation.vllm_cfg.precision", spec.precision.rollout))
    ov.append(("policy.generation.vllm_cfg.kv_cache_dtype", spec.precision.kv_cache_dtype))
    if spec.vllm_runtime.max_model_len is None:
        ov.append(("policy.generation.vllm_cfg.max_model_len", m.max_total_sequence_length))
    else:
        # The smoke launcher owns this key through POLAR_MODEL_MAX_MODEL_LEN.
        # Use an additive positional override so a serving-only context can be
        # decoupled from the smaller trainer smoke sequence length.
        ov.append(
            (
                "++policy.generation.vllm_cfg.max_model_len",
                spec.vllm_runtime.max_model_len,
            )
        )
    # The pinned NeMo base declares vllm_kwargs as an empty structured mapping
    # and predates enable_prefix_caching in vllm_cfg. These runtime keys
    # therefore require Hydra's additive `++` form. Keeping them in the
    # compiled experiment delta makes the serving contract lineage-visible and
    # routes them as positional overrides after the smoke launcher's defaults.
    runtime = spec.vllm_runtime
    # NeMo only forwards free-form engine constructor arguments from
    # generation.vllm_kwargs. The similarly named keys under vllm_cfg are
    # accepted by Hydra but are not passed to vLLM, which silently leaves the
    # engine at its default (observed as a 2,048-token chunk in the live
    # Qwen3.5 smoke). Emit the authoritative copies here; they are appended
    # after the smoke launcher's compatibility overrides.
    if runtime.max_num_seqs is not None:
        ov.append(
            (
                "++policy.generation.vllm_kwargs.max_num_seqs",
                runtime.max_num_seqs,
            )
        )
    if runtime.max_num_batched_tokens is not None:
        ov.append(
            (
                "++policy.generation.vllm_kwargs.max_num_batched_tokens",
                runtime.max_num_batched_tokens,
            )
        )
    if runtime.enable_prefix_caching is not None:
        ov.append(
            (
                "++policy.generation.vllm_cfg.enable_prefix_caching",
                runtime.enable_prefix_caching,
            )
        )
    if runtime.enable_chunked_prefill is not None:
        ov.append(
            (
                "++policy.generation.vllm_kwargs.enable_chunked_prefill",
                runtime.enable_chunked_prefill,
            )
        )
    if runtime.mamba_cache_mode is not None:
        ov.append(
            (
                "++policy.generation.vllm_kwargs.mamba_cache_mode",
                runtime.mamba_cache_mode,
            )
        )
    if runtime.language_model_only is not None:
        ov.append(
            (
                "++policy.generation.vllm_kwargs.language_model_only",
                runtime.language_model_only,
            )
        )
    if runtime.quantization_ignored_layer_kws:
        ov.append(
            (
                "++policy.generation.vllm_cfg.quantization_ignored_layer_kws",
                runtime.quantization_ignored_layer_kws,
            )
        )
    ov.append(("policy.generation.temperature", r.temperature))
    ov.append(("policy.generation.top_p", r.top_p))
    ov.append(("policy.generation.colocated.enabled", t.colocated))
    ov.append(("policy.generation.colocated.resources.gpus_per_node", t.gpus_per_node))
    ov.append(("policy.generation.colocated.resources.num_nodes", t.gen_nodes))
    ov.append(("cluster.gpus_per_node", t.gpus_per_node))
    ov.append(("cluster.num_nodes", t.num_nodes))

    # verifier -> NeMo data processor/env
    processor, env_name = _VERIFIER_DATA.get(
        spec.verifier.type, ("math_hf_data_processor", "math")
    )
    ov.append(("data.default.processor", processor))
    ov.append(("data.default.env_name", env_name))
    ov.append(("data_plane.enabled", False))

    return CompiledExperiment(
        spec_id=spec.id, base_config=SGLANG_BASE, nemo_overrides=ov
    )


def _emit_objective(a: Algorithm, ov: list[tuple[str, object]]) -> None:
    """Emit the loss_fn flags that distinguish the objective from GRPO baseline.

    GRPO baseline emits nothing here (it is the base config default). Each other
    objective is a config point in the single ClippedPGLossFn.
    """
    if a.name == "grpo":
        return
    if a.name == "drgrpo":
        # Dr. GRPO (Liu et al.): drop group-std normalization only
        # (advantage_estimator.py:72-78 guard); keep leave-one-out mean baseline
        # and flat token-level loss. One-key delta over GRPO.
        ov.append(("grpo.normalize_rewards", False))
        return
    if a.name == "rloo":
        ov.append(("loss_fn.disable_ppo_ratio", True))
        return
    if a.name == "cispo":
        low, high = a.clip or _CISPO_DEFAULT_CLIP
        ov.append(("loss_fn.use_cispo", True))
        ov.append(("loss_fn.token_level_loss", True))
        ov.append(("loss_fn.sequence_level_importance_ratios", False))
        ov.append(("loss_fn.ratio_clip_min", float(low)))
        ov.append(("loss_fn.ratio_clip_max", float(high)))
        return
    if a.name == "gspo":
        ov.append(("loss_fn.sequence_level_importance_ratios", True))
        ov.append(("loss_fn.token_level_loss", False))
        return
    if a.name == "dapo":
        low, high = a.clip or _DAPO_DEFAULT_CLIP
        ov.append(("loss_fn.ratio_clip_min", float(low)))
        ov.append(("loss_fn.ratio_clip_max", float(high)))
        ov.append(("loss_fn.token_level_loss", True))
        return
    raise SpecCompileError(f"unknown objective: {a.name!r}")  # pragma: no cover


def _validate(spec: ExperimentSpec) -> None:
    a = spec.algorithm
    ag = spec.async_grpo
    lora = spec.trainer.lora

    if lora.enabled:
        if lora.target_modules and lora.exclude_modules:
            raise SpecCompileError(
                "LoRA target_modules and exclude_modules are mutually exclusive."
            )
        if lora.match_all_linear and (lora.target_modules or lora.exclude_modules):
            raise SpecCompileError(
                "LoRA match_all_linear=true requires empty target_modules and "
                "exclude_modules."
            )

    # async GRPO excludes DAPO data-side features (run_grpo.py:155-160).
    if ag.enabled and (a.dynamic_sampling or a.reward_shaping):
        raise SpecCompileError(
            "async GRPO does not support DAPO data-side features "
            "(dynamic_sampling / reward_shaping); see run_grpo.py:155-160. "
            "Run these on the sync path or disable async."
        )

    # drgrpo is a GRPO-family normalization preset; coherent only with the grpo advantage estimator.
    if a.name == "drgrpo" and a.advantage != "grpo":
        raise SpecCompileError(
            "drgrpo is a GRPO-family normalization preset and requires "
            f"advantage='grpo'; got {a.advantage!r}."
        )

    # GDPO needs a multi-component reward (advantage_estimator.py:113-119).
    if a.advantage == "gdpo" and spec.verifier.reward_components < 2:
        raise SpecCompileError(
            "gdpo advantage estimator requires a multi-reward verifier "
            f"(reward_components >= 2); got {spec.verifier.reward_components}."
        )

    # lag>1 requires in-flight weight updates (grpo.py:2930-2935).
    if ag.enabled and ag.lag > 1 and not ag.in_flight_weight_updates:
        raise SpecCompileError(
            "max_trajectory_age_steps > 1 requires in_flight_weight_updates=true "
            "(grpo.py:2930)."
        )

    # CISPO incompatibilities (loss_functions.py:245-265).
    if a.name == "cispo":
        if a.advantage not in ("grpo", "reinforce_plus_plus", "gdpo"):
            raise SpecCompileError(  # pragma: no cover - guarded by Literal
                "cispo requires a GRPO-family advantage estimator."
            )
