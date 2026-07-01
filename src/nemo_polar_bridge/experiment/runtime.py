"""Runtime profile + launch composer for the experiment-as-code engine.

The composer does not reimplement the docker/ray/polar orchestration; it reuses
the proven smoke launcher
(``scripts/smoke/run_nemo_polar_external_collector_spark_smoke.sh``), which is
parameterized by env vars (``NEMO_GRPO_*``, ``POLAR_MODEL_*``, ``POLAR_DATASET_*``,
``MODEL_HOST``/``MODEL_CONT``) and takes ``$1``=slug plus positional
``EXTRA_OVERRIDES`` appended last.

The spec's experiment knobs that the smoke already parameterizes map to its env
contract; the algorithm-axis keys the smoke does not set become positional
overrides. ``SMOKE_COVERED_KEYS`` is that seam: any compiled override outside it
is the experiment's novel contribution and is emitted positionally.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .compile import compile_spec
from .spec import ExperimentSpec

DEFAULT_SMOKE_SCRIPT = "scripts/smoke/run_nemo_polar_external_collector_spark_smoke.sh"
DEFAULT_HF_HUB_HOST_ROOT = "/home/jarrodbarnes/.cache/huggingface/hub"


def _default_hf_hub_host_root() -> str:
    """Overridable so a clone off Jarrod's machine can point at its own HF cache."""

    return os.environ.get("NEMO_POLAR_HF_HUB_HOST_ROOT", DEFAULT_HF_HUB_HOST_ROOT)

# The smoke-covered keys the compiler can also emit: a curated subset of the keys
# the smoke launcher sets (run_nemo_polar_external_collector_spark_smoke.sh:
# 464-523), omitting the logger.* keys and the `+`-append keys the compiler never
# emits. The GRPO baseline's compiled overrides are all within this set, so the
# baseline composes to env-only with zero positional extras; anything the compiler
# emits outside it (the beyond-GRPO loss/advantage keys) is appended as a
# positional override. Those positionals carry no `+` prefix, so under
# set_struct(True) they must already exist in the merged base (true on c236061b;
# a future objective needing a base-absent key would require `+`-prefix support
# the compiler does not have today).
SMOKE_COVERED_KEYS = frozenset(
    {
        "grpo.async_grpo.enabled",
        "grpo.async_grpo.max_trajectory_age_steps",
        # NOT in_flight_weight_updates / recompute_kv_cache_after_weight_updates:
        # unlike every other key here, the smoke script hardcodes literal
        # `=true`/`=false` with no env-var hook (verified against
        # run_nemo_polar_external_collector_spark_smoke.sh:546-547), so
        # treating them as "covered" silently dropped a spec's differing
        # value. compile.py only emits these two when they diverge from the
        # smoke script's hardcoded defaults (see compile.py), so the GRPO
        # baseline (which uses those defaults) still composes to zero
        # positional extras; a spec that flips one reaches the launch
        # command as a positional override, appended after the smoke
        # script's own hardcoded args, so Hydra's last-override-wins
        # semantics apply -- the same mechanism the CISPO loss_fn.* keys
        # already use.
        "grpo.num_prompts_per_step",
        "grpo.num_generations_per_prompt",
        "grpo.max_num_steps",
        "grpo.max_num_epochs",
        "grpo.val_at_start",
        "grpo.val_at_end",
        "grpo.val_period",
        "grpo.overlong_filtering",
        "grpo.use_dynamic_sampling",
        "grpo.reward_scaling.enabled",
        "grpo.reward_shaping.enabled",
        "loss_fn.reference_policy_kl_penalty",
        "loss_fn.use_importance_sampling_correction",
        "policy.model_name",
        "policy.tokenizer.name",
        "policy.max_total_sequence_length",
        "policy.sequence_packing.enabled",
        "policy.dynamic_batching.enabled",
        "policy.train_global_batch_size",
        "policy.train_micro_batch_size",
        "policy.logprob_batch_size",
        "policy.generation_batch_size",
        "policy.generation.backend",
        "policy.generation.max_new_tokens",
        "policy.generation.temperature",
        "policy.generation.top_p",
        "policy.generation.top_k",
        "policy.generation.port_range_low",
        "policy.generation.port_range_high",
        "policy.generation.vllm_cfg.async_engine",
        "policy.generation.vllm_cfg.enable_vllm_metrics_logger",
        "policy.generation.vllm_cfg.precision",
        "policy.generation.vllm_cfg.kv_cache_dtype",
        "policy.generation.vllm_cfg.max_model_len",
        "policy.generation.colocated.enabled",
        "policy.generation.colocated.resources.gpus_per_node",
        "policy.generation.colocated.resources.num_nodes",
        "data.train.dataset_name",
        "data.train.split_validation_size",
        "data.default.processor",
        "data.default.env_name",
        "env.math.num_workers",
        "data_plane.enabled",
        "checkpointing.enabled",
        "cluster.gpus_per_node",
        "cluster.num_nodes",
    }
)


class RuntimeProfile(BaseModel):
    """Spark/infra environment the spec deliberately does not own.

    Defaults reproduce the two-Spark setup; fields left ``None`` fall back to the
    smoke launcher's own defaults.
    """

    model_config = ConfigDict(extra="forbid")

    smoke_script: str = DEFAULT_SMOKE_SCRIPT
    hf_hub_host_root: str = Field(default_factory=_default_hf_hub_host_root)
    model_cont_root: str = "/host-hf/hub"
    matrix_name: str = "smoke"
    image: Optional[str] = None
    nemo_rl_ref: Optional[str] = None
    head_ip: Optional[str] = None
    worker_ip: Optional[str] = None
    run_root: Optional[str] = None
    extra_env: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def two_spark(cls) -> "RuntimeProfile":
        return cls()

    @classmethod
    def from_yaml(cls, path) -> "RuntimeProfile":
        with open(path) as handle:
            return cls.model_validate(yaml.safe_load(handle) or {})


@dataclass
class LaunchPlan:
    spec_id: str
    slug: str
    smoke_script: str
    env: dict[str, str]
    extra_overrides: list[str]

    def argv(self) -> list[str]:
        return ["bash", self.smoke_script, self.slug, *self.extra_overrides]

    def shell(self) -> str:
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
        command = " ".join(shlex.quote(part) for part in self.argv())
        return f"{env_prefix} {command}".strip()


def _hf_snapshot_subpath(name: str, snapshot: str) -> str:
    return f"models--{name.replace('/', '--')}/snapshots/{snapshot}"


def _env_bool(value: bool) -> str:
    return "true" if value else "false"


def compose_launch(
    spec: ExperimentSpec,
    profile: RuntimeProfile,
    *,
    slug: Optional[str] = None,
    matrix_cell: str = "",
) -> LaunchPlan:
    compiled = compile_spec(spec)
    slug = slug or spec.id
    model = spec.model

    if not model.snapshot and not model.nemo_path:
        raise ValueError(
            "model.snapshot (or model.nemo_path) is required to resolve the "
            "container/host model paths for launch"
        )

    env: dict[str, str] = {}
    # rollout / async shape
    env["NEMO_GRPO_NUM_PROMPTS_PER_STEP"] = str(spec.rollout.prompts_per_step)
    env["NEMO_GRPO_NUM_GENERATIONS_PER_PROMPT"] = str(spec.rollout.generations_per_prompt)
    env["NEMO_GRPO_MAX_NUM_STEPS"] = str(spec.rollout.max_steps)
    env["NEMO_GRPO_MAX_TRAJECTORY_AGE_STEPS"] = str(spec.async_grpo.lag)
    # model
    env["POLAR_MODEL_NAME"] = model.name
    env["POLAR_MODEL_MAX_TOTAL_SEQUENCE_LENGTH"] = str(model.max_total_sequence_length)
    env["POLAR_MODEL_MAX_MODEL_LEN"] = str(model.max_total_sequence_length)
    env["POLAR_MODEL_TEMPERATURE"] = str(spec.rollout.temperature)
    env["POLAR_MODEL_TOP_P"] = str(spec.rollout.top_p)
    env["NEMO_VLLM_PRECISION"] = spec.precision.rollout
    env["NEMO_VLLM_KV_CACHE_DTYPE"] = spec.precision.kv_cache_dtype
    if spec.vllm_runtime.gpu_memory_utilization is not None:
        env["NEMO_VLLM_GPU_MEMORY_UTILIZATION"] = str(
            spec.vllm_runtime.gpu_memory_utilization
        )
    if spec.vllm_runtime.enforce_eager is not None:
        env["NEMO_VLLM_ENFORCE_EAGER"] = _env_bool(spec.vllm_runtime.enforce_eager)
    if spec.vllm_runtime.max_num_seqs is not None:
        env["NEMO_VLLM_MAX_NUM_SEQS"] = str(spec.vllm_runtime.max_num_seqs)
    if spec.vllm_runtime.max_num_batched_tokens is not None:
        env["NEMO_VLLM_MAX_NUM_BATCHED_TOKENS"] = str(
            spec.vllm_runtime.max_num_batched_tokens
        )
    if model.snapshot:
        subpath = _hf_snapshot_subpath(model.name, model.snapshot)
        env["MODEL_HOST"] = f"{profile.hf_hub_host_root}/{subpath}"
        env["MODEL_CONT"] = model.nemo_path or f"{profile.model_cont_root}/{subpath}"
    else:
        env["MODEL_CONT"] = model.nemo_path  # type: ignore[assignment]
    # dataset / verifier
    env["POLAR_DATASET_FAMILY"] = spec.dataset.family
    env["POLAR_VERIFIER_TYPE"] = spec.verifier.type
    env["POLAR_EXECUTION_TYPE"] = spec.verifier.execution
    if spec.dataset.source:
        env["POLAR_DATASET_ID"] = spec.dataset.source
    if spec.dataset.subset:
        env["POLAR_SOURCE_DATASETS"] = spec.dataset.subset
    if spec.dataset.local_jsonl:
        env["POLAR_DATASET_LOCAL_JSONL"] = spec.dataset.local_jsonl
    # matrix
    env["POLAR_MATRIX_NAME"] = profile.matrix_name
    if matrix_cell:
        env["POLAR_MATRIX_CELL"] = matrix_cell
    # optional infra overrides (else smoke defaults apply)
    if profile.image:
        env["IMAGE"] = profile.image
    if profile.nemo_rl_ref:
        env["NEMO_RL_REF"] = profile.nemo_rl_ref
    if profile.head_ip:
        env["HEAD_IP"] = profile.head_ip
    if profile.worker_ip:
        env["WORKER_IP"] = profile.worker_ip
    if profile.run_root:
        env["RUN_DIR"] = f"{profile.run_root.rstrip('/')}/{slug}"
    env.update(profile.extra_env)

    # algorithm-axis overrides: anything the smoke does not already set
    extra_overrides = [
        arg
        for (key, _), arg in zip(compiled.nemo_overrides, compiled.nemo_override_args())
        if key not in SMOKE_COVERED_KEYS
    ]

    return LaunchPlan(
        spec_id=spec.id,
        slug=slug,
        smoke_script=profile.smoke_script,
        env=env,
        extra_overrides=extra_overrides,
    )
