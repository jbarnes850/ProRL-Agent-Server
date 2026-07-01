"""Declarative experiment-as-code spec for the two-Spark NeMo + Polar RL lab.

A spec captures only the portable *experiment* knobs (model, dataset, verifier,
algorithm, async staleness, rollout shape, generation backend, topology shape).
Infrastructure (node IPs, ports, data paths, NCCL env, logger/checkpoint
toggles) is supplied separately by a runtime profile, not by the spec. This is
the EaC specification/execution boundary: the spec is portable and versioned;
the runtime profile is environment-specific.

The spec compiles (see ``compile.py``) to a NeMo base config plus a list of
dotted overrides applied through NeMo RL's own loader, exactly the surface used
by ``run_grpo_with_polar --config <base> <overrides>``.
"""

from __future__ import annotations

from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

Objective = Literal["grpo", "drgrpo", "dapo", "cispo", "gspo", "rloo"]
Advantage = Literal["grpo", "gdpo", "reinforce_plus_plus"]

_STRICT = ConfigDict(extra="forbid")


class Model(BaseModel):
    """Policy model. ``nemo_path`` is the trainer-visible (container) path; when
    absent the HF ``name`` is used directly."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str
    snapshot: Optional[str] = None
    nemo_path: Optional[str] = None
    max_total_sequence_length: int = 4096

    @property
    def train_path(self) -> str:
        return self.nemo_path or self.name


class Dataset(BaseModel):
    model_config = _STRICT

    family: str = "nemo_gym"
    source: Optional[str] = None
    subset: Optional[str] = None
    local_jsonl: Optional[str] = None


class Verifier(BaseModel):
    model_config = _STRICT

    type: str = "exact_answer"
    execution: str = "single_turn_chat"
    reward_components: int = 1


class Algorithm(BaseModel):
    """The swappable algorithm axis. ``name`` selects the policy-loss objective
    (all share NeMo's single ``ClippedPGLossFn``); ``advantage`` selects the
    estimator within the GRPO family. ``clip`` is the additive NeMo convention
    ``(ratio_clip_min, ratio_clip_max)`` -> clamp ``[1-min, 1+max]``."""

    model_config = _STRICT

    name: Objective = "grpo"
    advantage: Advantage = "grpo"
    clip: Optional[tuple[float, float]] = None
    kl_penalty: float = 0.0
    dynamic_sampling: bool = False
    reward_shaping: bool = False


class AsyncConfig(BaseModel):
    """NeMo native Async GRPO staleness window. ``lag`` maps to
    ``grpo.async_grpo.max_trajectory_age_steps``."""

    model_config = _STRICT

    enabled: bool = True
    lag: int = 1
    in_flight_weight_updates: bool = True
    recompute_kv_cache: bool = False


class Rollout(BaseModel):
    model_config = _STRICT

    prompts_per_step: int
    generations_per_prompt: int
    max_steps: int
    max_epochs: int = 1
    temperature: float = 1.0
    top_p: float = 1.0


class Topology(BaseModel):
    """Disaggregation *shape* (not addresses). Defaults reproduce the two-Spark
    non-colocated split: ``num_nodes`` total (train + inference), ``gen_nodes``
    of them dedicated to generation."""

    model_config = _STRICT

    colocated: bool = False
    gpus_per_node: int = 1
    num_nodes: int = 2
    gen_nodes: int = 1


class Precision(BaseModel):
    model_config = _STRICT

    train: str = "bfloat16"
    rollout_backend: Literal["vllm", "sglang"] = "vllm"
    rollout: str = "bfloat16"
    kv_cache_dtype: str = "auto"


class VllmRuntime(BaseModel):
    model_config = _STRICT

    gpu_memory_utilization: Optional[float] = None
    enforce_eager: Optional[bool] = None
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str
    parent: Optional[str] = None
    model: Model
    dataset: Dataset
    verifier: Verifier
    algorithm: Algorithm = Field(default_factory=Algorithm)
    async_grpo: AsyncConfig = Field(default_factory=AsyncConfig)
    rollout: Rollout
    topology: Topology = Field(default_factory=Topology)
    precision: Precision = Field(default_factory=Precision)
    vllm_runtime: VllmRuntime = Field(default_factory=VllmRuntime)

    @classmethod
    def from_yaml(cls, path) -> "ExperimentSpec":
        with open(path) as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data)
