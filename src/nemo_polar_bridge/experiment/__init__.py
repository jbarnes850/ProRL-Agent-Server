"""Experiment-as-code layer for the two-Spark NeMo + Polar RL lab.

A declarative :class:`ExperimentSpec` compiles to a NeMo base config plus dotted
overrides validated by NeMo RL's own loader. This is the specification layer of
the experiment-as-code stack; lineage and orchestration build on top.
"""

from .compile import CompiledExperiment, SpecCompileError, compile_spec
from .lineage import (
    LineageRecord,
    LineageStore,
    compiled_digest,
    register_experiment,
    spec_digest,
)
from .runtime import LaunchPlan, RuntimeProfile, compose_launch
from .spec import (
    Algorithm,
    AsyncConfig,
    Dataset,
    ExperimentSpec,
    Model,
    Precision,
    Rollout,
    Topology,
    Verifier,
)

__all__ = [
    "Algorithm",
    "AsyncConfig",
    "CompiledExperiment",
    "Dataset",
    "ExperimentSpec",
    "LaunchPlan",
    "LineageRecord",
    "LineageStore",
    "Model",
    "Precision",
    "Rollout",
    "RuntimeProfile",
    "SpecCompileError",
    "Topology",
    "Verifier",
    "compile_spec",
    "compiled_digest",
    "compose_launch",
    "register_experiment",
    "spec_digest",
]
