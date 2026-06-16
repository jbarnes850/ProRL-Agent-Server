"""Compile an :class:`ExperimentSpec` into a NeMo base config + dotted overrides.

The compiler owns the experiment-level overrides only. Every key it emits
already exists in the merged base config (``grpo-qwen3-0.6b-1n8g-sglang.yaml`` ->
``grpo_math_1B.yaml``), so NeMo's loader applies them under ``set_struct(True)``
without an append (``+``) prefix. Grounding for each mapping lives in the
NeMo RL revision pinned in the lab skill:

- one ``ClippedPGLossFn`` for the whole PG family (loss_functions.py:161-205)
- async staleness via ``grpo.async_grpo.*`` (grpo.py:125-136)
- advantage estimator dispatch ``grpo.adv_estimator.name`` (grpo.py:1445)
- async excludes DAPO data-side features (run_grpo.py:155-160)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .spec import Algorithm, ExperimentSpec

# Base recipe the proven two-Spark Async GRPO run inherits from. The spec
# carries the experiment delta over this base.
SGLANG_BASE = "examples/configs/recipes/llm/grpo-qwen3-0.6b-1n8g-sglang.yaml"

# Laguna parity for CISPO: (c_low, c_high) = (1, 4) -> NeMo clamp [1-1, 1+4]=[0,5].
_CISPO_DEFAULT_CLIP = (1.0, 4.0)
# DAPO Clip-Higher reference (dapo recipe: ratio_clip_max=0.28).
_DAPO_DEFAULT_CLIP = (0.2, 0.28)

# Scope limitation (decision 2026-06-16): the advantage baseline is NeMo's
# standard (unweighted) RLOO leave-one-out (grpo.use_leave_one_out_baseline),
# NOT Laguna's length-weighted leave-one-out (tech report eq. 2). Exact Laguna
# CISPO parity would need a small advantage_estimator.py change; deferred. CISPO
# here is the standard RLOO baseline with CISPO clipping.

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
    return str(value)


def compile_spec(spec: ExperimentSpec) -> CompiledExperiment:
    _validate(spec)

    ov: list[tuple[str, object]] = []
    a = spec.algorithm
    ag = spec.async_grpo
    r = spec.rollout
    m = spec.model
    t = spec.topology

    # async / two-Spark disaggregation
    ov.append(("grpo.async_grpo.enabled", ag.enabled))
    ov.append(("grpo.async_grpo.max_trajectory_age_steps", ag.lag))
    ov.append(("grpo.async_grpo.in_flight_weight_updates", ag.in_flight_weight_updates))
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

    # generation / topology shape
    ov.append(("policy.generation.backend", spec.precision.rollout_backend))
    ov.append(("policy.generation.vllm_cfg.async_engine", True))
    ov.append(("policy.generation.vllm_cfg.max_model_len", m.max_total_sequence_length))
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
        # Dr. GRPO (Liu et al., "Understanding R1-Zero-Like Training"): drop only
        # the group-std normalization; keep the leave-one-out mean baseline and
        # the flat token-level loss. Verified against NeMo c236061b -- the only
        # std-division guard is advantage_estimator.py:72-78, and TOKEN_LEVEL is a
        # flat global-token mean (loss_functions.py:589-604); adv_estimator.
        # normalize_rewards interpolates from grpo.normalize_rewards. One-key
        # delta over GRPO; token_level_loss=true is already the base default and
        # the leave-one-out baseline must stay on (never emitted false here).
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

    # async GRPO excludes DAPO data-side features (run_grpo.py:155-160).
    if ag.enabled and (a.dynamic_sampling or a.reward_shaping):
        raise SpecCompileError(
            "async GRPO does not support DAPO data-side features "
            "(dynamic_sampling / reward_shaping); see run_grpo.py:155-160. "
            "Run these on the sync path or disable async."
        )

    # drgrpo is a GRPO-family normalization preset (group-std off, mean baseline
    # kept); it is only coherent with the grpo advantage estimator.
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
